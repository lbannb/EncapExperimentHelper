#=
    joint_sampling_benchmark.jl — encap entry point, rewritten against
    EncapExperimentHelper.

    One entry script for all experiments: the encap run name (-n) selects the
    experiment. Start with

        encap rerun joint_sampling_benchmark.jl -sl_i 1 -n "optimize_trial_state"
        encap rerun joint_sampling_benchmark.jl -sl_i 4 -n "<experiment>_evolve"
        encap rerun joint_sampling_benchmark.jl -sl_i 4 -n "<experiment>_compute_exp_values"

    Order: optimize_trial_state first (single instance; every :frozen experiment
    loads its converged η from the shared file in data/), then the *_evolve runs
    (ENCAP_PROCID selects the bond dimension), then *_compute_exp_values.

    Any extra "--param value" pairs override the matching entry in the params dict
    (see joint_sampling_common.jl). With encap, pass them via -a/--args — note the
    leading space, which keeps encap from parsing the value as one of its own flags:

        encap rerun joint_sampling_benchmark.jl -sl_i 4 -n "joint_flat_frozen_evolve" \
            -a " --eigencut 1e-3 --lr 0.01 --maxiter 500"
=#

import EncapExperimentHelper as EEH

# ---- where am I? ------------------------------------------------------------------
# encap copies this script into <root>/<script stem>/<run name>/ and runs it there, so
# @__DIR__ is the run folder and the shared files are one level up; run in place, they
# are in the <script stem>/ subfolder. encap_context resolves both, and creates this
# run's save/ and logs/ folders and the shared data/ folder.
ctx = EEH.encap_context()

# ---- experiment selection: the encap run folder name is the -n argument ------------
const EXPERIMENTS = Dict(
    #                                                   starting PEPS   trial state
    "optimize_trial_state"                                => (flat=true,  trial=:slater_only),
    "joint_flat_frozen_evolve"                            => (flat=true,  trial=:frozen),
    "joint_flat_frozen_compute_exp_values"                => (flat=true,  trial=:frozen),
    "joint_flat_simultaneous_evolve"                      => (flat=true,  trial=:simultaneous),
    "joint_flat_simultaneous_compute_exp_values"          => (flat=true,  trial=:simultaneous),
    "joint_random_frozen_evolve"                          => (flat=false, trial=:frozen),
    "joint_random_frozen_compute_exp_values"              => (flat=false, trial=:frozen),
    "joint_random_simultaneous_evolve"                    => (flat=false, trial=:simultaneous),
    "joint_random_simultaneous_compute_exp_values"        => (flat=false, trial=:simultaneous),
    "joint_random_frozen_simultaneous_evolve"             => (flat=false, trial=:frozen_simultaneous),
    "joint_random_frozen_simultaneous_compute_exp_values" => (flat=false, trial=:frozen_simultaneous),
    "peps_only_evolve"                                    => (flat=false, trial=:none),
    "peps_only_compute_exp_values"                        => (flat=false, trial=:none),
)

settings = EEH.select_run(EXPERIMENTS, ctx)

const FLAT_START = settings.flat  # true: flat PEPS of ones, false: random + multiplied spectrum
const TRIAL_MODE = settings.trial # :none | :simultaneous | :frozen | :frozen_simultaneous | :slater_only
const KIND = TRIAL_MODE == :slater_only ? :optimize_trial_state :
             (endswith(ctx.run_name, "_evolve") ? :evolve : :compute_exp_values)

# file-name prefix, shared by the evolve/compute pair of one experiment
const EXPERIMENT = "Hubbard_benchmark_" * (KIND == :optimize_trial_state ? "trial_state" :
    replace(ctx.run_name, r"_(evolve|compute_exp_values)$" => ""))

# ---- shared setup + the body matching this run kind --------------------------------
include(EEH.shared_path(ctx, "joint_sampling_common.jl"))
include(EEH.shared_path(ctx,
    KIND == :optimize_trial_state ? "optimize_trial_state_body.jl" :
    KIND == :evolve               ? "joint_sampling_evolve_body.jl" :
                                    "joint_sampling_exp_values_body.jl"))
