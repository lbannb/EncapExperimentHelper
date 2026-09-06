#=
    joint_sampling_common.jl — shared setup for the joint-sampling benchmark,
    rewritten against EncapExperimentHelper.

    This is the acceptance example for the package: the same file as
    Finite_GfPEPS_sampling_paper_results/Hubbard_model/diagnosis/
    joint_sampling_benchmark_pi_flux_Hubbard/joint_sampling_common.jl,
    with the encap boilerplate replaced by library calls. The physics is untouched.

    Included by the encap entry script (see joint_sampling_benchmark.jl in this
    folder), which defines, BEFORE the include:

        ctx        :: EEH.EncapContext — the run's encap context
        EXPERIMENT :: String  — file-name prefix, shared by the evolve/compute
                                pair of one experiment
        FLAT_START :: Bool    — true:  flat starting PEPS (all ones)
                                false: random starting PEPS with multiplied spectrum
        TRIAL_MODE :: Symbol  — :none | :simultaneous | :frozen |
                                :frozen_simultaneous | :slater_only
        KIND       :: Symbol  — :optimize_trial_state | :evolve | :compute_exp_values

    All file names are generated from the run parameters via `tag`, so these scripts
    double as templates: change the params and the names follow.

    The three *_body.jl files need two changes to run against this:
        save(save_file, d)  ->  EEH.save_checkpoint(save_file, d)
        archive_log()       ->  EEH.archive_log(log)
=#

import EncapExperimentHelper as EEH

using QuantumNaturalfPEPS
using QuantumNaturalGradient
using LinearAlgebra
using Random
using ITensors, ITensorMPS
using JLD2
using Distributed
using TimerOutputs

QNG = QuantumNaturalGradient
QNfP = QuantumNaturalfPEPS

# ---- Distributed setup: one single-threaded worker per CPU thread --------------
# cpus_per_task falls back to Sys.CPU_THREADS outside a SLURM allocation, so the
# script also runs on a login node or in the REPL.
nr_procs = EEH.cpus_per_task()
@show nr_procs

addprocs(nr_procs, exeflags="-t 1")
@everywhere begin
    using QuantumNaturalfPEPS
    using LinearAlgebra
    BLAS.set_num_threads(1) # Each worker should only use one BLAS thread
end
BLAS.set_num_threads(nr_procs) # For computing the double layers

@show ctx.run_name ctx.instance EXPERIMENT FLAT_START TRIAL_MODE KIND

#=
    Settings
=#

params = Dict{Symbol, Any}(
    :T => Float64,
    :seed => 30062026 + ctx.instance, # random seed; ctx.instance = ENCAP_PROCID + 1
    :Lx => 4,
    :Ly => 4,
    :bdim => 2,
    :sample_nr => 2500,           # samples per iteration during the evolution
    :sample_nr_exp_vals => 25000, # samples for the expectation-value run
    :lr => 0.05,
    :t => 1.0,
    :U => 1.0,                    # U/t = 1.0 for the whole benchmark
    :eigencut => 1e-4,
    # approximate (non-exact) contraction settings
    :contract_cutoff => 1e-6,
    :contract_dim => 300,
    :sample_cutoff => 5e-4,
    :maxiter => 400,              # evolution iterations
    :maxiter_preopt => 200,       # trial-state optimization iterations
    :α_init => 2.0,               # spectrum multiplication exponent (random start only)
    :parity_sector => 0,          # 0=even, 1=odd
    :target_state => 0,           # target state, 0 for ground state, 1 for first excited, ...
)

# ---- command-line overrides: any "--param value" pair overrides params[:param] --
# e.g.  julia joint_sampling_benchmark.jl --eigencut 1e-3 --lr 0.01 --maxiter 500
# e.g.  encap rerun joint_sampling_benchmark.jl -sl_i 4 -n "joint_flat_frozen_evolve" \
#           -a " --eigencut 1e-3 --lr 0.01 --maxiter 500"   <- note the leading space
EEH.apply_cli_overrides!(params)
@show params

# does the trial state carry variational parameters (optimized with the PEPS)?
const TRIAL_VARIATIONAL = TRIAL_MODE in (:simultaneous, :frozen_simultaneous)
# does the run start from the shared pre-optimized trial state?
const TRIAL_LOAD_FROZEN = TRIAL_MODE in (:frozen, :frozen_simultaneous)

N = params[:Lx] * params[:Ly]

# ---- generic file naming from the run parameters --------------------------------
# The trial-state optimization is independent of the bond dimension, so its files
# must not be tagged with D — that is what `skip` is for.
const TAG = ["L"   => p -> "$(p[:Lx])x$(p[:Ly])",
             "N"   => :sample_nr,
             "U_t" => p -> round(p[:U] / p[:t]; digits=4),
             "D"   => :bdim]

no_D = KIND == :optimize_trial_state
tag = EEH.run_tag(params, TAG; skip = no_D ? ["D"] : ())

# Shared pre-optimized trial state: written by the optimize_trial_state run, loaded
# by every experiment with TRIAL_MODE = :frozen. It lives in the shared data/ folder
# at the experiment root, outside any run folder, which is what lets other runs read
# it — and it is never tagged with D.
base_tag = EEH.run_tag(params, TAG; skip = ["D"])
trial_state_file = EEH.data_file(ctx, "Hubbard_benchmark_trial_state", base_tag)

data_dir = ctx.data_dir                                  # created by encap_context
checkpoint_file(name) = EEH.checkpoint_file(ctx, name, tag)
load_checkpoint(file) = EEH.load_checkpoint(file)

# ---- log handling ----------------------------------------------------------------
# Renames the encap log (enumerated by ENCAP_PROCID) to a stable, parameter-tagged
# name; the bodies call EEH.archive_log(log) at the very end of the run, which
# appends it to logs/ so restarted runs accumulate instead of overwriting.
log = EEH.adopt_log!(ctx, tag)

# ---- model ------------------------------------------------------------------------
Random.seed!(params[:seed])

hilbert = siteinds("Fermion", params[:Lx], params[:Ly])
hilbert_peps = siteinds("S=1/2", params[:Lx], params[:Ly])

ham_op = QNfP.TensorOperatorSum(
    QNfP.hamiltonian_hubbard_pi_flux_hopping(params[:t], params[:U], params[:Lx], params[:Ly]), hilbert)

# ---- state constructors -------------------------------------------------------------
# flat PEPS tensors: every entry equal to one
function constant_peps_tensor(::Type{S}, incoming, outgoing) where {S<:Number}
    inds = (incoming..., outgoing...)
    data = ones(S, map(dim, inds)...)
    return ITensor(data, inds...)
end

# Starting PEPS: flat (all ones) or random unitaries with the spectrum
# multiplied by a power-law factor as described in arXiv/2503.12557.
function build_peps(; bdim=params[:bdim], flat=FLAT_START)
    peps = PEPS(params[:T], hilbert_peps; bond_dim=bdim,
        contract_cutoff=params[:contract_cutoff],
        contract_dim=params[:contract_dim],
        sample_cutoff=params[:sample_cutoff],
        tensor_init=(flat ? constant_peps_tensor : QNfP.random_unitary))
    if !flat
        spectrum = collect(1:bdim) .^ (-params[:α_init])
        QNfP.multiply_spectrum!(peps, spectrum)
    end
    return peps
end

# π-flux Gaussian (Slater) trial-state parameters (the starting guess for this run):
#   χᵢ,ᵢ₊ₓ = -χ,  χᵢ,ᵢ₊y = -χ·(-1)^{i_x},  Δᵢⱼ = 0,  a₀(i) = 0  →  π flux per plaquette
function pi_flux_η(χ=1.0)
    n_MF = QNfP.get_max_num_MF_params_NN(params[:Lx], params[:Ly])
    n_x  = QNfP.get_max_num_hopping_x_NN(params[:Lx], params[:Ly])
    n_y  = QNfP.get_max_num_hopping_y_NN(params[:Lx], params[:Ly])
    η = zeros(n_MF)
    η[N + 1 : N + n_x] .= -χ                                        # x-hoppings: uniform χ
    signs = [(-1)^i for j in 1:(params[:Ly]-1) for i in 1:params[:Lx]]  # (-1)^{i_x} per y-bond
    η[N + n_x + 1 : N + n_x + n_y] .= -χ .* signs                   # y-hoppings: staggered sign
    return η
end

build_trial_state(η) = QNfP.GaussianState(QNfP.build_general_H_BdG_2D_NN, N; η=η,
    parity_sector=params[:parity_sector], target_state=params[:target_state])
