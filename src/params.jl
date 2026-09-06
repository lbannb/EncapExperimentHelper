# ---- command-line parameter overrides ---------------------------------------------

"""
    apply_cli_overrides!(params, args=ARGS; io=stdout) -> params

Override entries of a defaults dict from `--key value` pairs on the command line, in
place. Each value is parsed to the type of the default it replaces, and every override
is echoed to `io` so the log records what the run actually used.

```julia
params = Dict{Symbol,Any}(:lr => 0.05, :maxiter => 400, :verbose => false)
EncapExperimentHelper.apply_cli_overrides!(params)
```
```
julia run.jl --lr 0.01 --maxiter 500
```

Passing them through encap needs a **leading space** in the `-a` value, which is what
stops encap from parsing `--lr` as one of its own flags:

```
encap rerun run.jl -sl_i 4 -n "peps_only_evolve" -a " --lr 0.01 --maxiter 500"
```

Errors — rather than silently ignoring — on an argument that is not `--key`, on an
unknown key (listing the valid ones), on a missing value, and on a key whose default
has a type that cannot be parsed from a string (see [`parse_as`](@ref)).
"""
function apply_cli_overrides!(params::AbstractDict{Symbol}, args = ARGS; io::IO = stdout)
    i = firstindex(args)
    while i <= lastindex(args)
        arg = args[i]
        startswith(arg, "--") || error("Expected --<param> <value>, got \"$arg\"")
        key = Symbol(arg[3:end])
        haskey(params, key) || error(
            "Unknown parameter --$key; valid: " * join(sorted_keys(params), ", "))
        i + 1 <= lastindex(args) || error("Missing value for --$key")
        old = params[key]
        params[key] = parse_as(old, args[i+1], key)
        println(io, "Command-line override: $key = $(params[key]) (was $old)")
        i += 2
    end
    return params
end

sorted_keys(params::AbstractDict) = sort!(String[string(k) for k in keys(params)])

"""
    parse_as(default, str, key) -> value

Parse the command-line string `str` to the type of `default`. Handles `Bool`, any
`Integer`, any `AbstractFloat`, `AbstractString` and `Symbol`; anything else (a
`DataType`-valued parameter such as `:T => Float64`, say) throws, since there is no
sensible way to build it from a string.

Add a method to support your own parameter type.
"""
parse_as(::Bool, str::AbstractString, key::Symbol) = parse(Bool, str)
parse_as(default::Integer, str::AbstractString, key::Symbol) = parse(typeof(default), str)
parse_as(default::AbstractFloat, str::AbstractString, key::Symbol) = parse(typeof(default), str)
parse_as(::AbstractString, str::AbstractString, key::Symbol) = String(str)
parse_as(::Symbol, str::AbstractString, key::Symbol) = Symbol(str)
parse_as(default, str::AbstractString, key::Symbol) = error(
    "Parameter --$key (of type $(typeof(default))) cannot be set from the command line")
