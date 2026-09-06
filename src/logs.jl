# ---- log handling -------------------------------------------------------------------
# encap writes a run's stdout to <run_dir>/log for instance 0 and <run_dir>/log_<procid>
# for the rest, so parallel instances of one script produce logs you cannot tell apart,
# and a restart overwrites the previous one. Rename the live log to a parameter-tagged
# name at startup, then append it to <run_dir>/logs/ at the end so restarts accumulate.

"""
    LogHandle

The two paths of a run's log: `live`, the tagged file encap's output is flowing into,
and `archive`, the accumulating file in `<run_dir>/logs/` it is appended to at the end
of the run. Produced by [`adopt_log!`](@ref), consumed by [`archive_log`](@ref).
"""
struct LogHandle
    live::String
    archive::String
end

function Base.show(io::IO, ::MIME"text/plain", h::LogHandle)
    println(io, "LogHandle")
    println(io, "  live    = ", h.live)
    print(io,   "  archive = ", h.archive)
end

"""
    adopt_log!(ctx, tag; prefix="log") -> LogHandle

Rename this run's encap log to a stable, parameter-tagged name and return the
[`LogHandle`](@ref) to close it out with at the end of the run.

encap's launcher opens the log by name and appends to it for the lifetime of the job
(the `julia ... &>> \$log` line in `slurm_files/run.sh`), so the write end is an open
file descriptor pointing at the inode. Renaming the file therefore does not interrupt
logging — output keeps flowing into it under the new name. This is a POSIX property and
does not hold on Windows, where an open handle blocks the rename.

If an untidied log from a previous, unarchived run of the same tag is already there — a
job killed by a SLURM timeout before it could call [`archive_log`](@ref) — it is
archived first rather than clobbered.

```julia
log = EncapExperimentHelper.adopt_log!(ctx, tag)
# ... the run ...
EncapExperimentHelper.archive_log(log)
```
"""
function adopt_log!(ctx::EncapContext, tag; prefix::AbstractString = "log")
    name = string(prefix, "_", run_tag(tag))
    encap_log = joinpath(ctx.run_dir,
                         ctx.procid == 0 ? String(prefix) : "$(prefix)_$(ctx.procid)")
    handle = LogHandle(joinpath(ctx.run_dir, name), joinpath(ctx.run_dir, "logs", name))
    isfile(encap_log) || return handle                  # not started by encap
    abspath(encap_log) == abspath(handle.live) && return handle
    isfile(handle.live) && archive_log(handle)          # leftover of a killed run
    mv(encap_log, handle.live; force = true)
    return handle
end

"""
    archive_log(h::LogHandle) -> LogHandle

Append this run's log to `<run_dir>/logs/` and remove the live file. Call it at the very
end of a run. Appending — rather than moving — is what lets a restarted run add to the
record of the previous attempts instead of replacing it.

A no-op when the live log is gone, so it is safe to call twice, and safe in a script
that was not started by encap.
"""
function archive_log(h::LogHandle)
    isfile(h.live) || return h
    mkpath(dirname(h.archive))
    open(h.archive, "a") do out
        open(h.live, "r") do input
            write(out, input)
        end
    end
    rm(h.live; force = true)
    return h
end
