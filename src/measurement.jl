using GaussianProcesses

struct VulcanStateGP
    state_history::Matrix{Int64} # path history of (x,y) states: 2x(t+k) matrix
    info_history::Vector{Float64} # history of information states
    gp::GPE # Gaussian Process that has been trained on all (Gaussian) stored historical data
end

function add_measurement(curr_state_gp::VulcanStateGP, X::Matrix{Int64}, info::Float64)
    return vcat(curr_state_gp.state_history, X)
end