# ---- checkpoints and result files ----------------------------------------------------

"""
    checkpoint_file(ctx, name, tag) -> String

Path of a checkpoint in this run's `save/` folder: `<run_dir>/save/<name>_<tag>.jld2`.
Per-run and parameter-tagged, so a rerun of the same experiment finds its own checkpoint
and continues instead of starting the optimisation over.
"""
checkpoint_file(ctx::EncapContext, name::AbstractString, tag) =
    joinpath(ctx.run_dir, "save", string(name, "_", run_tag(tag), ".jld2"))

"""
    data_file(ctx, name, tag; suffix="") -> String

Path of a result file in the shared `data/` folder at the experiment root:
`<data_dir>/<name>_<tag><suffix>.jld2`. Unlike [`checkpoint_file`](@ref) this sits
*outside* any run folder, which is what lets one run consume another's results — a
pre-optimised state written by one experiment and loaded by several others.
"""
data_file(ctx::EncapContext, name::AbstractString, tag; suffix::AbstractString = "") =
    joinpath(ctx.data_dir, string(name, "_", run_tag(tag), suffix, ".jld2"))

"""
    snapshot_file(ctx, name, tag, iteration) -> String

Path of the snapshot taken at `iteration`:
`<data_dir>/<name>_<tag>_snapshots/iteration_<iteration>.jld2`.

Unlike the checkpoint, which every iteration overwrites, snapshots are kept, so a later
run can re-evaluate the state along the optimisation, e.g. with many more samples than
one optimisation step can afford. They sit in the shared `data/` folder for the same
reason as [`data_file`](@ref): the run that measures them is not the run that wrote them.
"""
snapshot_file(ctx::EncapContext, name::AbstractString, tag, iteration::Integer) =
    joinpath(snapshot_dir(ctx, name, tag), "iteration_$(iteration).jld2")

"""
    snapshot_dir(ctx, name, tag) -> String

The folder holding the snapshots of [`snapshot_file`](@ref).
"""
snapshot_dir(ctx::EncapContext, name::AbstractString, tag) =
    joinpath(ctx.data_dir, string(name, "_", run_tag(tag), "_snapshots"))

"""
    list_snapshots(dir) -> Vector{Tuple{Int,String}}

The `iteration_<n>.jld2` files in `dir` as `(n, path)` pairs, sorted by iteration number
(sorting by name would put `iteration_100` before `iteration_25`). Empty when the folder
does not exist yet.
"""
function list_snapshots(dir::AbstractString)
    snapshots = Tuple{Int,String}[]
    isdir(dir) || return snapshots
    for file in readdir(dir)
        m = match(r"^iteration_(\d+)\.jld2$", file)
        m === nothing || push!(snapshots, (parse(Int, m.captures[1]), joinpath(dir, file)))
    end
    return sort!(snapshots; by = first)
end

"""
    check_compatible(saved, current, keys; what="saved state") -> nothing

Throw unless `saved` and `current` agree on every key in `keys` — the guard before
continuing from a checkpoint or starting from another run's result, where a state saved
for another lattice or coupling would otherwise be picked up without complaint. Both may
be dicts (with `Symbol` or `String` keys) or named tuples; the error lists every
mismatch, not just the first.

```julia
EncapExperimentHelper.check_compatible(ckpt["params"], params, [:Lx, :Ly, :J1, :J2, :bdim])
```
"""
function check_compatible(saved, current, keys; what::AbstractString = "saved state")
    mismatches = String[]
    for key in keys
        a, b = lookup_param(saved, key), lookup_param(current, key)
        isequal(a, b) || push!(mismatches, "  $key: $(repr(a)) in the $what, $(repr(b)) now")
    end
    isempty(mismatches) ||
        error("The $what does not match the current run:\n" * join(mismatches, "\n"))
    return nothing
end

lookup_param(x::AbstractDict, key) =
    haskey(x, Symbol(key)) ? x[Symbol(key)] : get(x, string(key), missing)
lookup_param(x::NamedTuple, key) = get(x, Symbol(key), missing)

"""
    load_checkpoint(path) -> Dict{String,Any} or nothing

Contents of the checkpoint at `path`, or `nothing` when there is none — so a run can
resume or start fresh with one branch:

```julia
ckpt = EncapExperimentHelper.load_checkpoint(save_file)
if ckpt !== nothing
    theta = copy(ckpt["theta"])
end
```
"""
load_checkpoint(path::AbstractString) = isfile(path) ? JLD2.load(path) : nothing

"""
    save_checkpoint(path, data; atomic=true) -> path

Write `data` to `path`, by default atomically: to a temporary file next to it, then
`rename` onto the target. Checkpoints are typically written every iteration, so a job
killed mid-write — a SLURM timeout, an OOM kill — would otherwise leave a truncated file
where the only resume point should be. The rename is atomic as long as the temporary
lands on the same filesystem, which it does since it is written alongside the target.
"""
function save_checkpoint(path::AbstractString, data::AbstractDict; atomic::Bool = true)
    mkpath(dirname(path))
    payload = Dict{String,Any}(string(k) => v for (k, v) in data)
    if !atomic
        write_jld2(path, payload)
        return path
    end
    base, ext = splitext(path)
    tmp = string(base, ".tmp.", getpid(), ext)   # same folder, so the rename is atomic
    try
        write_jld2(tmp, payload)
        mv(tmp, path; force = true)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    return path
end

# Write through jldopen rather than JLD2.save, which goes via FileIO and dispatches on
# the file extension — the temporary file of an atomic write does not have the usual one.
function write_jld2(path::AbstractString, payload::AbstractDict)
    JLD2.jldopen(path, "w") do file
        for (key, value) in payload
            file[key] = value
        end
    end
    return path
end

"""
    append_data(path, data; cols=(), vecs=(), atomic=true) -> path

Write `data` to `path`, concatenating the named entries onto what is already stored
there: `cols` are `hcat`ed (per-iteration column vectors, e.g. a parameter history),
`vecs` are `vcat`ed (per-iteration scalars, e.g. an energy history). Everything else is
replaced.

This is what makes a restarted run extend the record of the previous attempt rather than
truncate it to whatever the last attempt happened to compute:

```julia
EncapExperimentHelper.append_data(results, Dict(
        "params"    => params,
        "E_history" => misc["history"][!, :energy]);
    vecs = ["E_history"])
```

Keys listed in `cols`/`vecs` that are absent from either `data` or the existing file are
left alone, so the first run of an experiment needs no special case.
"""
function append_data(path::AbstractString, data::AbstractDict;
                     cols = (), vecs = (), atomic::Bool = true)
    existing = isfile(path) ? JLD2.load(path) : Dict{String,Any}()
    out = Dict{String,Any}(string(k) => v for (k, v) in data)
    concat_into!(out, existing, cols, hcat)
    concat_into!(out, existing, vecs, vcat)
    return save_checkpoint(path, out; atomic)
end

function concat_into!(out, existing, keys, combine)
    for key in keys
        k = string(key)
        (haskey(out, k) && haskey(existing, k)) || continue
        out[k] = combine(existing[k], out[k])
    end
    return out
end
