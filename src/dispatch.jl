# ---- run-name dispatch ----------------------------------------------------------------

"""
    select_run(table, run_name; what="experiment") -> value
    select_run(table, ctx::EncapContext; what="experiment") -> value

Look up the settings of a run in a table keyed by run name. A run's identity is the name
it was started with (`encap -n <name>`, recovered as `basename(@__DIR__)`), which lets
one entry script serve every experiment of a benchmark:

```julia
const EXPERIMENTS = Dict(
    "peps_only_evolve"         => (flat = false, trial = :none),
    "joint_flat_frozen_evolve" => (flat = true,  trial = :frozen),
)
settings = EncapExperimentHelper.select_run(EXPERIMENTS, ctx)
```

A typo in `-n` is otherwise only noticed once the job has been queued, started and
crashed somewhere unrelated, so an unknown name throws immediately, listing the valid
ones.
"""
function select_run(table::AbstractDict, run_name::AbstractString;
                    what::AbstractString = "experiment")
    haskey(table, run_name) && return table[run_name]
    error("Unknown $what \"$run_name\" — start via encap with -n <name>, " *
          "where <name> is one of:\n  " *
          join(sort!(String[string(k) for k in keys(table)]), "\n  "))
end

select_run(table::AbstractDict, ctx::EncapContext; kwargs...) =
    select_run(table, ctx.run_name; kwargs...)
