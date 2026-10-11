# EncapExperimentHelper

Helper functions for experiments started and monitored with
[encap](https://github.com/computational-quantum-science/encap).

encap runs a script by copying it into `<experiment root>/<run name>/` and launching it
there, one instance per `ENCAP_PROCID`. That leaves every script to solve the same handful
of problems: find its shared code now that `@__DIR__` has moved, tell its log apart from its
siblings' and not lose it on a restart, take parameter overrides from the command line, name
its files consistently, and resume from a checkpoint instead of starting the optimisation
over.

Nothing is exported — use it as `EEH.foo`.

```julia
import EncapExperimentHelper as EEH

ctx = EEH.encap_context()          # run_name, run_dir, shared_dir, data_dir, procid, instance

params = Dict{Symbol,Any}(:lr => 0.05, :maxiter => 400, :bdim => 2)
EEH.apply_cli_overrides!(params)   # --lr 0.01 --maxiter 500

tag = EEH.run_tag(params, [:bdim, :maxiter])
log = EEH.adopt_log!(ctx, tag)

save_file = EEH.checkpoint_file(ctx, "save", tag)
ckpt = EEH.load_checkpoint(save_file)             # nothing on a fresh run
θ = ckpt === nothing ? initial_guess() : ckpt["θ"]

for i in 1:params[:maxiter]
    θ = step(θ)
    EEH.save_checkpoint(save_file, Dict("θ" => θ))  # atomic: a killed job cannot truncate it
end

EEH.archive_log(log)
```

`examples/` holds the joint-sampling benchmark's entry script and shared setup file written
against the package.

## What it provides

### Context — `encap_context`, `shared_path`, `find_up`, `sweep`, `cpus_per_task`

`encap_context()` resolves where the run lives and creates the folders the rest of the
package writes into:

| field                     |                                                          |
|:--------------------------|:---------------------------------------------------------|
| `run_name`                | `basename(run_dir)` — the name passed to encap as `-n`   |
| `run_dir`                 | this run's encap folder; holds `log`, `logs/`, `save/`   |
| `shared_dir`              | experiment root: shared `*.jl` and `data/` live here     |
| `data_dir`                | `<shared_dir>/data`, shared between all runs             |
| `procid`, `instance`      | `ENCAP_PROCID` (0 outside encap) and `procid + 1`        |
| `under_encap`             | whether this looks like an encap run folder              |

`shared_dir` is resolved for both ways the same script gets run — inside an encap run folder
(recognised by the `.encap.conf` encap drops there) the root is the parent; run in place, it
is the folder named after the script, taken from `PROGRAM_FILE` rather than hardcoded.

```julia
include(EEH.shared_path(ctx, "common.jl"))                       # code shared by one experiment
include(joinpath(EEH.find_up("lib"; from = ctx.run_dir), "peps.jl"))  # ... by several
params[:bdim] = EEH.sweep(ctx, [2, 3, 4, 6])                     # ENCAP_PROCID picks the value
addprocs(EEH.cpus_per_task())                                    # SLURM_CPUS_PER_TASK, with a fallback
```

`shared_path` and `find_up` return absolute paths, so they are also safe to hand to
`@everywhere include(...)` — workers do not share the driver's working directory.

### Parameters — `apply_cli_overrides!`

Overrides a defaults dict from `--key value` pairs, parsing each value to the type of the
default it replaces, and echoing every override so the log records what the run used.
Unknown keys, missing values and unparseable types throw rather than being ignored.

Through encap, the `-a` value needs a **leading space**, which stops encap from parsing
`--lr` as one of its own flags:

```
encap rerun run.jl -sl_i 4 -n "peps_only_evolve" -a " --lr 0.01 --maxiter 500"
```

`apply_env_overrides!` does the same from environment variables `<prefix><KEY>`, which
reach a job without encap's argument parsing (`sbatch --export`, a wrapper script). Call it
first, so the command line wins. Besides the types above, both accept `Rational`s
(`1/2`), `yes/no` and `on/off` for `Bool`s, and a default of `nothing` for an optional
parameter (`none` keeps it unset; anything else is read as a number):

```julia
EEH.apply_env_overrides!(params; prefix = "PPEPS_")   # PPEPS_LX=8 PPEPS_SZ_SECTOR=1/2
EEH.apply_cli_overrides!(params)
Random.seed!(EEH.instance_seed(ctx, 12_500))           # 12_500 + 10_000 * ENCAP_PROCID
```

### Workers — `setup_workers`

Starts `procs` workers with `threads` Julia threads each, loads `modules` on them, pins
their BLAS to one thread and gives the driver `blas_threads` (default: the whole
allocation). This is the layout the QuantumNaturalfPEPS samplers expect. Workers start in
the driver's active project, so a script with its own environment gets the same package
versions everywhere:

```julia
EEH.setup_workers(threads = 8, modules = [:QuantumNaturalfPEPS])  # cpus_per_task() ÷ 8 workers
```

### Tags — `run_tag`

One string derived from the run parameters, shared by the log, the checkpoints and the data
files, so every file of a run lines up and changing a parameter makes the names follow.
Parts are `Symbol`s, or `label => key` / `label => f(params)` pairs for renamed and derived
quantities; `skip` drops a part for a run kind that does not depend on it.

```julia
TAG = ["L" => p -> "$(p[:Lx])x$(p[:Ly])", "N" => :sample_nr, "D" => :bdim]
EEH.run_tag(params, TAG)                  # "L=4x4_N=2500_D=2"
EEH.run_tag(params, TAG; skip = ["D"])    # "L=4x4_N=2500"    — for a D-independent run
```

Anywhere the package takes a tag it also takes a plain `String`, so you can build your own.

### Logs — `adopt_log!`, `archive_log`

encap writes a run's stdout to `<run_dir>/log` for instance 0 and `<run_dir>/log_<procid>`
for the rest, so parallel instances produce logs you cannot tell apart, and a restart
overwrites the previous one.

`adopt_log!` renames the live log to `log_<tag>` at startup; `archive_log` appends it to
`<run_dir>/logs/log_<tag>` at the end, so restarts accumulate. A live log left behind by a
job killed before it could archive is archived rather than clobbered.

The rename works because encap's launcher holds the log open by name for the lifetime of the
job, so the write end is a file descriptor pointing at the inode and output keeps flowing
into the renamed file. **This is a POSIX property** — it does not hold on Windows, where an
open handle blocks the rename.

### Checkpoints — `checkpoint_file`, `data_file`, `load_checkpoint`, `save_checkpoint`, `append_data`

`checkpoint_file` names a per-run, parameter-tagged file under `<run_dir>/save/`;
`data_file` names one in the shared `data/` folder, outside any run folder, which is what
lets one run consume another's results.

`save_checkpoint` writes atomically by default — to a temporary file beside the target, then
`rename` onto it. Checkpoints are typically written every iteration, so a job killed
mid-write would otherwise leave a truncated file where the only resume point should be.

`append_data` concatenates named entries onto what is already stored (`cols` are `hcat`ed,
`vecs` are `vcat`ed), so a restarted run extends the history of the previous attempt instead
of truncating it:

```julia
EEH.append_data(results, Dict("params" => params, "θ_history" => θ_hist, "E_history" => E);
                cols = ["θ_history"], vecs = ["E_history"])
```

### Snapshots and compatibility — `snapshot_file`, `list_snapshots`, `check_compatible`

The checkpoint is overwritten every iteration; snapshots are kept, so a later run can
re-measure the state along the optimisation with many more samples. They go to
`<data_dir>/<name>_<tag>_snapshots/iteration_<n>.jld2`, in the shared folder, because the
run that measures them is not the run that wrote them. `list_snapshots` returns them sorted
by iteration number.

`check_compatible(saved, current, keys)` throws, listing every mismatch, unless a saved
state and the current run agree on `keys` — the guard before resuming or starting from
another run's result:

```julia
EEH.save_checkpoint(EEH.snapshot_file(ctx, "peps", tag, niter), Dict("θ" => θ, "params" => params))
for (iteration, path) in EEH.list_snapshots(EEH.snapshot_dir(ctx, "peps", tag))
    snapshot = EEH.load_checkpoint(path)
    EEH.check_compatible(snapshot["params"], params, [:Lx, :Ly, :bdim])
    # ... measure ...
end
```

### Dispatch — `select_run`

Looks up a run's settings in a table keyed by run name, so one entry script can serve every
experiment of a benchmark. An unknown name throws immediately, listing the valid ones,
instead of being noticed once the job has queued, started and crashed somewhere unrelated.

```julia
settings = EEH.select_run(EXPERIMENTS, ctx)
```

## Tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```
