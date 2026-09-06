# ---- run tags ----------------------------------------------------------------------
# One string derived from the run parameters, shared by the log, the checkpoints and
# the data files of a run, so every file of an experiment lines up and changing a
# parameter makes the file names follow.

"""
    run_tag(params, keys; skip=(), sep="_", joiner="=") -> String
    run_tag(str) -> String

Build a file-name tag from the run parameters.

Each element of `keys` is either

* a `Symbol` — looked up in `params`, labelled with the key itself;
* a `label => key` pair — looked up in `params`, labelled with `label`;
* a `label => f` pair with `f` callable — labelled with `label`, valued `f(params)`,
  for quantities that are derived rather than stored (`"U_t" => p -> p[:U]/p[:t]`).

```julia
const TAG = [:Lx, :Ly, :sample_nr, :bdim]
tag = EncapExperimentHelper.run_tag(params, TAG)        # "Lx=4_Ly=4_sample_nr=2500_bdim=2"
```

`skip` drops parts by label, for a run kind that is independent of a parameter and
must not carry it in its file names — the trial-state optimisation in the joint
sampling benchmark does not depend on the bond dimension, and its files would
otherwise claim to:

```julia
tag = EncapExperimentHelper.run_tag(params, TAG; skip = independent_of_D ? [:bdim] : ())
```

Path separators and whitespace in values are replaced by `-`. Anywhere this package
takes a tag it also takes a plain `String`, so you can bypass this and build your own.
"""
run_tag(str::AbstractString) = String(str)

function run_tag(params::AbstractDict, keys; skip = (), sep::AbstractString = "_",
                 joiner::AbstractString = "=")
    skipset = Set{String}(string(s) for s in skip)
    parts = String[]
    for k in keys
        label, value = tag_part(params, k)
        label in skipset && continue
        push!(parts, string(label, joiner, sanitize_tag(value)))
    end
    return join(parts, sep)
end

function tag_part(params::AbstractDict, key::Symbol)
    haskey(params, key) || error(
        "run_tag: no parameter :$key; valid: " * join(sorted_keys(params), ", "))
    return string(key), params[key]
end

function tag_part(params::AbstractDict, pair::Pair)
    label, rhs = string(first(pair)), last(pair)
    rhs isa Symbol && return label, tag_part(params, rhs)[2]
    return label, rhs(params)
end

tag_part(::AbstractDict, other) = error(
    "run_tag: expected a Symbol or a label => key/function pair, got $(repr(other))")

sanitize_tag(value) = replace(string(value), r"[\/\s]" => "-")
