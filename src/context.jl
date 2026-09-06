# ---- encap run context ------------------------------------------------------------
# Everything about "where am I running" that can be known before the parameters exist.

"""
    EncapContext

Where a run lives in encap's folder structure.

encap copies the entry script into `<experiment root>/<run_name>/` and runs it there
(see `slurm_files/run.sh` in any run folder), so the script's own `@__DIR__` is the
run folder, not the folder the script was submitted from.

| field         |                                                                  |
|:--------------|:-----------------------------------------------------------------|
| `run_name`    | `basename(run_dir)` — the name passed to encap as `-n`           |
| `run_dir`     | this run's encap folder; holds `log`, `logs/`, `save/`           |
| `shared_dir`  | the experiment root: shared `*.jl` files and `data/` live here   |
| `data_dir`    | `<shared_dir>/data` — shared between all runs of the experiment  |
| `procid`      | `ENCAP_PROCID`, 0 when the script runs outside encap             |
| `instance`    | `procid + 1`, for 1-based indexing into a parameter sweep        |
| `under_encap` | whether this looks like an encap run folder                      |

Build one with [`encap_context`](@ref).
"""
struct EncapContext
    run_name::String
    run_dir::String
    shared_dir::String
    data_dir::String
    procid::Int
    instance::Int
    under_encap::Bool
end

function Base.show(io::IO, ::MIME"text/plain", ctx::EncapContext)
    println(io, "EncapContext(\"", ctx.run_name, "\")")
    println(io, "  run_dir    = ", ctx.run_dir)
    println(io, "  shared_dir = ", ctx.shared_dir)
    println(io, "  data_dir   = ", ctx.data_dir)
    print(io,   "  procid = ", ctx.procid, ", instance = ", ctx.instance,
                ", under_encap = ", ctx.under_encap)
end

"""
    encap_procid() -> Int

`ENCAP_PROCID` as an `Int`, or `0` when the variable is unset (i.e. the script is
being run by hand rather than by encap).
"""
encap_procid() = parse(Int, get(ENV, "ENCAP_PROCID", "0"))

"""
    script_stem() -> String

File name of the running script without its extension, or `""` in the REPL. encap
names the experiment root folder after the entry script, so this is what identifies
it — no need to hardcode the folder name in the script.
"""
script_stem() = isempty(PROGRAM_FILE) ? "" : splitext(basename(PROGRAM_FILE))[1]

default_run_dir() = isempty(PROGRAM_FILE) ? pwd() : dirname(abspath(PROGRAM_FILE))

"""
    shared_dir(run_dir) -> String

The experiment root for a script whose `@__DIR__` is `run_dir`, resolved for both
ways the same script gets run:

* under encap, `run_dir` is `<root>/<script stem>/<run name>` and the root is its
  parent (recognised by the `.encap.conf` encap drops in every run folder);
* run in place, the script sits next to a `<script stem>/` folder, which is the root.

Falls back to `run_dir` when neither applies (a script run in place before encap has
ever created its folder).
"""
function shared_dir(run_dir::AbstractString)
    run_dir = abspath(run_dir)
    stem = script_stem()
    if !isempty(stem) && isdir(joinpath(run_dir, stem))
        return joinpath(run_dir, stem)
    elseif isfile(joinpath(run_dir, ".encap.conf"))
        return dirname(run_dir)
    else
        return run_dir
    end
end

"""
    encap_context([run_dir]; data_subdir="data", create=true) -> EncapContext

Resolve the run's [`EncapContext`](@ref). `run_dir` defaults to the directory of the
running script, which is what you want when this is called from the encap entry
script; pass `@__DIR__` explicitly from anywhere else.

With `create=true` (the default) the `save/` and `logs/` folders of the run and the
shared `data/` folder are created if missing.

```julia
ctx = EncapExperimentHelper.encap_context()
```
"""
function encap_context(run_dir::AbstractString = default_run_dir();
                       data_subdir::AbstractString = "data",
                       create::Bool = true)
    run_dir = abspath(run_dir)
    shared = shared_dir(run_dir)
    data = joinpath(shared, data_subdir)
    procid = encap_procid()
    under = isfile(joinpath(run_dir, ".encap.conf")) || haskey(ENV, "ENCAP_PROCID")
    if create
        mkpath(data)
        mkpath(joinpath(run_dir, "save"))
        mkpath(joinpath(run_dir, "logs"))
    end
    return EncapContext(basename(run_dir), run_dir, shared, data, procid, procid + 1, under)
end

"""
    shared_path(ctx, parts...) -> String

Path to a file in the experiment root, e.g. a shared include:

```julia
include(EncapExperimentHelper.shared_path(ctx, "joint_sampling_common.jl"))
```

The path is absolute, so it is also safe to hand to `@everywhere include(...)` —
workers do not share the driver's working directory.
"""
shared_path(ctx::EncapContext, parts::AbstractString...) = joinpath(ctx.shared_dir, parts...)

"""
    find_up(name; from=pwd(), stop=nothing) -> String

Walk up from `from` until a directory contains an entry called `name`, and return the
absolute path of that entry. Use it to reach code shared by several experiments, which
lives above the experiment root and so is not covered by `ctx.shared_dir`:

```julia
include(joinpath(EncapExperimentHelper.find_up("lib"; from=ctx.run_dir), "peps_helpers.jl"))
```

Stops at the filesystem root, or at `stop` if given. Throws if `name` is never found,
naming every directory that was searched.
"""
function find_up(name::AbstractString; from::AbstractString = pwd(), stop = nothing)
    dir = abspath(from)
    stopdir = stop === nothing ? nothing : abspath(stop)
    searched = String[]
    while true
        candidate = joinpath(dir, name)
        ispath(candidate) && return candidate
        push!(searched, dir)
        (stopdir !== nothing && dir == stopdir) && break
        parent = dirname(dir)
        parent == dir && break   # filesystem root
        dir = parent
    end
    error("Could not find \"$name\" above $(abspath(from)); searched:\n  " *
          join(searched, "\n  "))
end

"""
    sweep(ctx, values)

The entry of `values` belonging to this encap instance, i.e. `values[ctx.instance]`.
encap starts instances `0 … i-1` (the `i:` key in `.encap.conf`) and exports the index
as `ENCAP_PROCID`; this maps that onto one swept parameter.

```julia
params[:bdim] = EncapExperimentHelper.sweep(ctx, [2, 3, 4, 6])
```

Throws a message naming the run and instance if the sweep is too short, which is
otherwise a `BoundsError` a long way from its cause.
"""
function sweep(ctx::EncapContext, values)
    n = length(values)
    1 <= ctx.instance <= n || error(
        "Run \"$(ctx.run_name)\": instance $(ctx.instance) (ENCAP_PROCID=$(ctx.procid)) " *
        "is out of range for a sweep of $n value(s) — start encap with -sl_i $n or fewer.")
    return values[begin + ctx.instance - 1]
end

"""
    cpus_per_task(default=Sys.CPU_THREADS) -> Int

`SLURM_CPUS_PER_TASK` as an `Int`, falling back to `default` outside a SLURM
allocation, so the same script also runs on a login node or in the REPL.
"""
function cpus_per_task(default::Integer = Sys.CPU_THREADS)
    haskey(ENV, "SLURM_CPUS_PER_TASK") || return Int(default)
    return parse(Int, ENV["SLURM_CPUS_PER_TASK"])
end
