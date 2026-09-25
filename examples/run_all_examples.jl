# Run from this directory: julia --project=. run_all_examples.jl
examples_dir = @__DIR__

for filename in readdir(examples_dir)
    endswith(filename, ".jl") && filename != basename(@__FILE__) || continue
    script = joinpath(examples_dir, filename)
    println("\nRunning ", filename)
    flush(stdout)
    # A fresh process releases each example's models and plotting memory.
    command = `$(Base.julia_cmd()) --project=$examples_dir --threads=1 $script`
    run(addenv(command, "OPENBLAS_NUM_THREADS" => "1",
                        "JULIA_NUM_PRECOMPILE_TASKS" => "1",
                        "GKSwstype" => "100"))
end

println("\nAll examples completed. Figures are in ", joinpath(examples_dir, "res"))
