# ---- worker processes ----------------------------------------------------------------

"""
    setup_workers(; threads=1, procs=cpus_per_task() ÷ threads,
                  blas_threads=cpus_per_task(), modules=Symbol[]) -> Vector{Int}

Start `procs` worker processes with `threads` Julia threads each, load `modules` on them
and pin their BLAS to one thread, then give the driver `blas_threads` BLAS threads.
Returns the new worker ids.

This is the layout the QuantumNaturalfPEPS samplers expect. Samples are spread over the
workers, and each worker samples with its Julia threads, so BLAS inside the workers must
stay single-threaded. The driver, meanwhile, uses every core for the dense contractions
between sampling rounds (the double-layer environments).

The default — one single-threaded worker per CPU of the allocation — is the
joint-sampling benchmark's. The PPEPS scripts use fewer workers with 8 threads each:

```julia
EncapExperimentHelper.setup_workers(threads = 8, modules = [:QuantumNaturalfPEPS])
```

Workers start in the driver's active project. A script that activates its own
environment therefore gets the same package versions on every worker, which a bare
`addprocs` does not guarantee.
"""
function setup_workers(; threads::Integer = 1,
                       procs::Integer = max(1, cpus_per_task() ÷ threads),
                       blas_threads::Integer = cpus_per_task(),
                       modules = Symbol[])
    project = Base.active_project()
    flags = project === nothing ? `-t $threads` : `-t $threads --project=$(dirname(project))`
    ws = Distributed.addprocs(procs; exeflags = flags)
    for m in modules
        Distributed.remotecall_eval(Main, ws, Expr(:using, Expr(:., Symbol(m))))
    end
    Distributed.remotecall_eval(Main, ws,
        :(import LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)))
    LinearAlgebra.BLAS.set_num_threads(blas_threads)
    return ws
end
