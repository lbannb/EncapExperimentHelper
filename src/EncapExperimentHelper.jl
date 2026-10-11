"""
    EncapExperimentHelper

Helpers for experiment scripts started and monitored with
[encap](https://github.com/computational-quantum-science/encap).

encap runs a script by copying it into `<experiment root>/<run name>/` and launching it
there, one instance per `ENCAP_PROCID`. That leaves every script to solve the same
handful of problems: find its shared code now that `@__DIR__` has moved, tell its log
apart from its siblings' and not lose it on a restart, accept parameter overrides from
the command line, name its files consistently, and resume from a checkpoint rather than
start the optimisation over. This package does those.

Nothing is exported; the intended use is

```julia
import EncapExperimentHelper as EEH

ctx    = EEH.encap_context()                 # run_name, run_dir, shared_dir, data_dir, instance
params = Dict{Symbol,Any}(:lr => 0.05, :maxiter => 400, :bdim => 2)
EEH.apply_cli_overrides!(params)             # --lr 0.01 --maxiter 500
tag    = EEH.run_tag(params, [:bdim, :maxiter])
log    = EEH.adopt_log!(ctx, tag)

save_file = EEH.checkpoint_file(ctx, "save", tag)
ckpt      = EEH.load_checkpoint(save_file)   # nothing on a fresh run
# ... run, calling EEH.save_checkpoint(save_file, state) as you go ...

EEH.archive_log(log)
```

See `examples/joint_sampling_common.jl` for a full setup file written against it.
"""
module EncapExperimentHelper

import Distributed
import JLD2
import LinearAlgebra

include("context.jl")
include("params.jl")
include("tags.jl")
include("logs.jl")
include("checkpoints.jl")
include("dispatch.jl")
include("workers.jl")

end # module
