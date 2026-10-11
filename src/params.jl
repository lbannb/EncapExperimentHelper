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
    apply_env_overrides!(params; prefix="", env=ENV, io=stdout) -> params

Override entries of a defaults dict from environment variables named `<prefix><KEY>`,
the key upper-cased, in place. Values are parsed by the same rules as
[`apply_cli_overrides!`](@ref), and every override is echoed to `io`.

Environment variables reach a job without going through encap's argument parsing — from
`sbatch --export`, a wrapper script, or an interactive shell. The PPEPS experiment
scripts are configured this way. Call it before `apply_cli_overrides!`, so the command
line has the last word:

```julia
params = Dict{Symbol,Any}(:Lx => 6, :J2 => 0.125, :sz_sector => nothing)
EncapExperimentHelper.apply_env_overrides!(params; prefix = "PPEPS_")  # PPEPS_LX=8 PPEPS_J2=0
EncapExperimentHelper.apply_cli_overrides!(params)
```

Unlike the command line, variables without a matching key are ignored, since the
environment holds much else besides.
"""
function apply_env_overrides!(params::AbstractDict{Symbol}; prefix::AbstractString = "",
                              env = ENV, io::IO = stdout)
    for key in sort!(collect(keys(params)); by = string)
        name = string(prefix, uppercase(string(key)))
        haskey(env, name) || continue
        old = params[key]
        params[key] = parse_as(old, env[name], key)
        println(io, "Environment override: $key = $(params[key]) (was $old, from $name)")
    end
    return params
end

"""
    parse_as(default, str, key) -> value

Parse the command-line string `str` to the type of `default`. Handles `Bool`
(`true/false`, `1/0`, `yes/no`, `on/off`), any `Integer`, any `AbstractFloat`, `Rational`
(`1/2` or `1//2`), `AbstractString` and `Symbol`. A default of `nothing` marks an optional
parameter: `nothing`/`none` keep it unset, anything else is read as an integer, a
rational or a float, in that order. Anything else (a `DataType`-valued parameter such as
`:T => Float64`, say) throws, since there is no sensible way to build it from a string.

Add a method to support your own parameter type.
"""
function parse_as(::Bool, str::AbstractString, key::Symbol)
    s = lowercase(strip(str))
    s in ("true", "1", "yes", "on") && return true
    s in ("false", "0", "no", "off") && return false
    throw(ArgumentError("Parameter --$key expects true/false, 1/0, yes/no or on/off, got \"$str\""))
end
parse_as(default::Integer, str::AbstractString, key::Symbol) = parse(typeof(default), str)
parse_as(default::AbstractFloat, str::AbstractString, key::Symbol) = parse(typeof(default), str)
parse_as(default::Rational, str::AbstractString, key::Symbol) = parse_rational(typeof(default), str, key)
parse_as(::AbstractString, str::AbstractString, key::Symbol) = String(str)
parse_as(::Symbol, str::AbstractString, key::Symbol) = Symbol(str)
function parse_as(::Nothing, str::AbstractString, key::Symbol)
    s = strip(str)
    lowercase(s) in ("nothing", "none") && return nothing
    value = tryparse(Int, s)
    value === nothing || return value
    occursin('/', s) && return parse_rational(Rational{Int}, s, key)
    value = tryparse(Float64, s)
    value === nothing || return value
    throw(ArgumentError("Parameter --$key: cannot read \"$str\" as nothing, an integer, " *
                        "a rational or a number"))
end
parse_as(default, str::AbstractString, key::Symbol) = error(
    "Parameter --$key (of type $(typeof(default))) cannot be set from the command line")

function parse_rational(::Type{Rational{T}}, str::AbstractString, key::Symbol) where {T}
    parts = split(replace(strip(str), "//" => "/"), "/")
    length(parts) in (1, 2) || throw(ArgumentError("Parameter --$key: \"$str\" is not a rational"))
    numerator = parse(T, parts[1])
    denominator = length(parts) == 2 ? parse(T, parts[2]) : one(T)
    iszero(denominator) && throw(ArgumentError("Parameter --$key: zero denominator in \"$str\""))
    return numerator // denominator
end
