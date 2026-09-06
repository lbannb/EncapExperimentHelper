using Test
import EncapExperimentHelper as EEH

# Build a temporary encap run folder: <root>/<stem>/<run_name>, with the .encap.conf
# marker encap drops in every run folder.
function fake_run_dir(run_name = "peps_only_evolve"; stem = "benchmark")
    root = mktempdir()
    run_dir = joinpath(root, stem, run_name)
    mkpath(run_dir)
    touch(joinpath(run_dir, ".encap.conf"))
    return root, run_dir
end

# Run `f` with ENCAP_PROCID set to `value`, restoring the environment afterwards.
function with_procid(f, value)
    had = haskey(ENV, "ENCAP_PROCID")
    old = get(ENV, "ENCAP_PROCID", "")
    value === nothing ? delete!(ENV, "ENCAP_PROCID") : (ENV["ENCAP_PROCID"] = string(value))
    try
        f()
    finally
        had ? (ENV["ENCAP_PROCID"] = old) : delete!(ENV, "ENCAP_PROCID")
    end
end

@testset "EncapExperimentHelper" begin

@testset "apply_cli_overrides!" begin
    defaults() = Dict{Symbol,Any}(
        :lr => 0.05, :maxiter => 400, :flat => false, :label => "run", :mode => :evolve,
        :T => Float64)

    # type-preserving parsing
    io = IOBuffer()
    p = EEH.apply_cli_overrides!(defaults(),
        ["--lr", "0.01", "--maxiter", "500", "--flat", "true",
         "--label", "sweep", "--mode", "compute"]; io)
    @test p[:lr] === 0.01
    @test p[:maxiter] === 500
    @test p[:flat] === true
    @test p[:label] == "sweep"
    @test p[:mode] === :compute

    # every override is echoed, showing old and new value
    log = String(take!(io))
    @test occursin("Command-line override: lr = 0.01 (was 0.05)", log)
    @test count("Command-line override", log) == 5

    # untouched defaults survive
    @test EEH.apply_cli_overrides!(defaults(), String[])[:lr] === 0.05

    # unknown key lists the valid ones
    err = try
        EEH.apply_cli_overrides!(defaults(), ["--lrate", "0.01"]; io = devnull)
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Unknown parameter --lrate", err.msg)
    @test occursin("maxiter", err.msg)

    @test_throws ErrorException EEH.apply_cli_overrides!(defaults(), ["--lr"]; io = devnull)
    @test_throws ErrorException EEH.apply_cli_overrides!(defaults(), ["lr", "0.01"]; io = devnull)
    @test_throws ErrorException EEH.apply_cli_overrides!(defaults(), ["-lr", "0.01"]; io = devnull)

    # a parameter whose default is a type cannot come from the command line
    @test_throws ErrorException EEH.apply_cli_overrides!(defaults(), ["--T", "Float32"]; io = devnull)

    # a bad value is a parse error, not a silent default
    @test_throws ArgumentError EEH.apply_cli_overrides!(defaults(), ["--maxiter", "abc"]; io = devnull)
end

@testset "run_tag" begin
    params = Dict{Symbol,Any}(:Lx => 4, :Ly => 4, :sample_nr => 2500, :bdim => 2,
                              :U => 1.0, :t => 1.0, :label => "a b/c")

    @test EEH.run_tag(params, [:Lx, :Ly, :bdim]) == "Lx=4_Ly=4_bdim=2"
    @test EEH.run_tag(params, [:bdim]; joiner = "_", sep = "-") == "bdim_2"

    # skip drops a part for a run kind that is independent of that parameter
    keys_all = [:Lx, :sample_nr, :bdim]
    @test EEH.run_tag(params, keys_all; skip = [:bdim]) == "Lx=4_sample_nr=2500"
    @test EEH.run_tag(params, keys_all; skip = ()) == "Lx=4_sample_nr=2500_bdim=2"

    # relabelled and derived parts
    @test EEH.run_tag(params, ["N" => :sample_nr]) == "N=2500"
    @test EEH.run_tag(params, ["U_t" => p -> round(p[:U] / p[:t]; digits = 4)]) == "U_t=1.0"
    @test EEH.run_tag(params, ["L" => p -> "$(p[:Lx])x$(p[:Ly])"]) == "L=4x4"
    @test EEH.run_tag(params, ["L" => p -> "$(p[:Lx])x$(p[:Ly])"]; skip = ["L"]) == ""

    # values that would otherwise break the file name
    @test EEH.run_tag(params, [:label]) == "label=a-b-c"

    # a plain string passes through, so callers can bypass the builder entirely
    @test EEH.run_tag("my_own_tag") == "my_own_tag"

    # a missing parameter names the valid keys instead of throwing a KeyError
    err = try
        EEH.run_tag(params, [:nope])
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("no parameter :nope", err.msg)
    @test occursin("sample_nr", err.msg)
end

@testset "context" begin
    root, run_dir = fake_run_dir("joint_flat_frozen_evolve")

    # inside an encap run folder the experiment root is the parent
    @test EEH.shared_dir(run_dir) == dirname(run_dir)

    # run in place, the root is the folder named after the script
    stem = EEH.script_stem()
    if !isempty(stem)
        inplace = mktempdir()
        mkpath(joinpath(inplace, stem))
        @test EEH.shared_dir(inplace) == joinpath(inplace, stem)
    end

    # neither: fall back to the directory itself
    bare = mktempdir()
    @test EEH.shared_dir(bare) == abspath(bare)

    with_procid(3) do
        ctx = EEH.encap_context(run_dir)
        @test ctx.run_name == "joint_flat_frozen_evolve"
        @test ctx.run_dir == abspath(run_dir)
        @test ctx.shared_dir == abspath(dirname(run_dir))
        @test ctx.data_dir == joinpath(abspath(dirname(run_dir)), "data")
        @test ctx.procid == 3
        @test ctx.instance == 4
        @test ctx.under_encap
        # create=true makes the folders the rest of the package writes into
        @test isdir(ctx.data_dir)
        @test isdir(joinpath(ctx.run_dir, "save"))
        @test isdir(joinpath(ctx.run_dir, "logs"))
    end

    # outside encap the proc id defaults to 0
    with_procid(nothing) do
        @test EEH.encap_procid() == 0
        @test EEH.encap_context(run_dir).instance == 1
    end

    # create=false touches nothing
    _, untouched = fake_run_dir()
    with_procid(nothing) do
        EEH.encap_context(untouched; create = false)
    end
    @test !isdir(joinpath(untouched, "save"))

    @test EEH.shared_path(EEH.encap_context(run_dir), "common.jl") ==
          joinpath(abspath(dirname(run_dir)), "common.jl")
end

@testset "sweep and cpus_per_task" begin
    _, run_dir = fake_run_dir()
    with_procid(2) do
        ctx = EEH.encap_context(run_dir)
        @test EEH.sweep(ctx, [2, 3, 4, 6]) == 4
        err = try
            EEH.sweep(ctx, [2, 3])
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("out of range for a sweep of 2", err.msg)
        @test occursin("ENCAP_PROCID=2", err.msg)
    end

    had = haskey(ENV, "SLURM_CPUS_PER_TASK")
    old = get(ENV, "SLURM_CPUS_PER_TASK", "")
    try
        delete!(ENV, "SLURM_CPUS_PER_TASK")
        @test EEH.cpus_per_task(7) == 7          # no allocation: fall back, do not throw
        ENV["SLURM_CPUS_PER_TASK"] = "48"
        @test EEH.cpus_per_task(7) == 48
    finally
        had ? (ENV["SLURM_CPUS_PER_TASK"] = old) : delete!(ENV, "SLURM_CPUS_PER_TASK")
    end
end

@testset "find_up" begin
    root = mktempdir()
    mkpath(joinpath(root, "lib"))
    deep = joinpath(root, "a", "b", "c")
    mkpath(deep)

    @test EEH.find_up("lib"; from = deep) == joinpath(root, "lib")
    @test EEH.find_up("b"; from = deep) == joinpath(root, "a", "b")

    err = try
        EEH.find_up("definitely_not_here"; from = deep, stop = root)
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("definitely_not_here", err.msg)
    @test occursin(abspath(deep), err.msg)
end

@testset "select_run" begin
    table = Dict("peps_only_evolve" => (flat = false,), "joint_flat_frozen_evolve" => (flat = true,))
    @test EEH.select_run(table, "peps_only_evolve").flat === false

    _, run_dir = fake_run_dir("joint_flat_frozen_evolve")
    with_procid(nothing) do
        @test EEH.select_run(table, EEH.encap_context(run_dir)).flat === true
    end

    err = try
        EEH.select_run(table, "peps_only_evolv")
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("peps_only_evolve", err.msg)
    @test occursin("joint_flat_frozen_evolve", err.msg)
end

@testset "logs" begin
    # instance 0 writes to "log", the others to "log_<procid>"
    _, run_dir = fake_run_dir()
    with_procid(0) do
        ctx = EEH.encap_context(run_dir)
        write(joinpath(ctx.run_dir, "log"), "first run\n")
        h = EEH.adopt_log!(ctx, "D=2")
        @test h.live == joinpath(ctx.run_dir, "log_D=2")
        @test h.archive == joinpath(ctx.run_dir, "logs", "log_D=2")
        @test isfile(h.live)
        @test !isfile(joinpath(ctx.run_dir, "log"))
        @test read(h.live, String) == "first run\n"

        EEH.archive_log(h)
        @test !isfile(h.live)
        @test read(h.archive, String) == "first run\n"
        @test EEH.archive_log(h) === h        # idempotent
    end

    _, run_dir2 = fake_run_dir()
    with_procid(3) do
        ctx = EEH.encap_context(run_dir2)
        write(joinpath(ctx.run_dir, "log_3"), "instance three\n")
        h = EEH.adopt_log!(ctx, "D=4")
        @test read(h.live, String) == "instance three\n"
        @test !isfile(joinpath(ctx.run_dir, "log_3"))
    end

    # a restart appends to the archive instead of overwriting it
    _, run_dir3 = fake_run_dir()
    with_procid(0) do
        ctx = EEH.encap_context(run_dir3)
        for text in ("attempt one\n", "attempt two\n")
            write(joinpath(ctx.run_dir, "log"), text)
            EEH.archive_log(EEH.adopt_log!(ctx, "D=2"))
        end
        @test read(joinpath(ctx.run_dir, "logs", "log_D=2"), String) ==
              "attempt one\nattempt two\n"
    end

    # a live log left behind by a killed job is archived, not clobbered
    _, run_dir4 = fake_run_dir()
    with_procid(0) do
        ctx = EEH.encap_context(run_dir4)
        write(joinpath(ctx.run_dir, "log_D=2"), "killed before archiving\n")
        write(joinpath(ctx.run_dir, "log"), "the restart\n")
        h = EEH.adopt_log!(ctx, "D=2")
        @test read(h.archive, String) == "killed before archiving\n"
        @test read(h.live, String) == "the restart\n"
        EEH.archive_log(h)
        @test read(h.archive, String) == "killed before archiving\nthe restart\n"
    end

    # no encap log at all (script run by hand): no error, nothing created
    _, run_dir5 = fake_run_dir()
    with_procid(nothing) do
        ctx = EEH.encap_context(run_dir5)
        h = EEH.adopt_log!(ctx, "D=2")
        @test !isfile(h.live)
        @test !isfile(h.archive)
    end
end

@testset "checkpoints" begin
    _, run_dir = fake_run_dir()
    with_procid(0) do
        ctx = EEH.encap_context(run_dir)

        save_file = EEH.checkpoint_file(ctx, "save", "D=2")
        @test save_file == joinpath(ctx.run_dir, "save", "save_D=2.jld2")
        @test EEH.data_file(ctx, "bench", "D=2") == joinpath(ctx.data_dir, "bench_D=2.jld2")
        @test EEH.data_file(ctx, "bench", "D=2"; suffix = "_Nexp_25000") ==
              joinpath(ctx.data_dir, "bench_D=2_Nexp_25000.jld2")

        @test EEH.load_checkpoint(save_file) === nothing

        EEH.save_checkpoint(save_file, Dict("theta" => [1.0, 2.0], "n" => 3))
        ckpt = EEH.load_checkpoint(save_file)
        @test ckpt["theta"] == [1.0, 2.0]
        @test ckpt["n"] == 3

        # the atomic write leaves no temporary behind
        @test isempty(filter(f -> occursin(".tmp.", f), readdir(joinpath(ctx.run_dir, "save"))))

        # symbol keys are accepted and stored as strings
        EEH.save_checkpoint(save_file, Dict(:theta => [9.0]))
        @test EEH.load_checkpoint(save_file)["theta"] == [9.0]

        EEH.save_checkpoint(save_file, Dict("theta" => [7.0]); atomic = false)
        @test EEH.load_checkpoint(save_file)["theta"] == [7.0]
    end
end

@testset "append_data" begin
    _, run_dir = fake_run_dir()
    with_procid(0) do
        ctx = EEH.encap_context(run_dir)
        path = EEH.data_file(ctx, "bench", "D=2")

        # first run: nothing to append to
        EEH.append_data(path, Dict("params" => Dict(:lr => 0.05),
                                   "theta_history" => [1.0 2.0; 3.0 4.0],
                                   "E_history" => [-1.0, -2.0]);
                        cols = ["theta_history"], vecs = ["E_history"])
        first = EEH.load_checkpoint(path)
        @test size(first["theta_history"]) == (2, 2)
        @test first["E_history"] == [-1.0, -2.0]

        # restart: histories extend, everything else is replaced
        EEH.append_data(path, Dict("params" => Dict(:lr => 0.01),
                                   "theta_history" => reshape([5.0, 6.0], 2, 1),
                                   "E_history" => [-3.0]);
                        cols = ["theta_history"], vecs = ["E_history"])
        second = EEH.load_checkpoint(path)
        @test second["theta_history"] == [1.0 2.0 5.0; 3.0 4.0 6.0]
        @test second["E_history"] == [-1.0, -2.0, -3.0]
        @test second["params"][:lr] == 0.01
    end
end

end
