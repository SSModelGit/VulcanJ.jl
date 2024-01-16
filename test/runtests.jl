using VulcanJ
using Test

@testset "VulcanJ.jl" begin
    # Write your tests here.
    @testset "Trial Tests" begin
        include("trial_test.jl")
    end
end
