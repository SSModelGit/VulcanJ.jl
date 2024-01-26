using VulcanJ
using Test

@testset "VulcanJ.jl" begin
    # Write your tests here.
    #= @testset "Trial Tests" begin
        include("trial_test.jl")
    end =#

    #= @testset "Underwater Volcano Search Test" begin
        include("volcano_search.jl")
    end =#
end

include("volcano_search.jl")
