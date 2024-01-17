module EnvironmentGP

using GaussianProcesses

struct EnvNode
    δmi::Float64 # Mutual information gained by entering the current information node
    gp::GPE # Gaussian Process that has been trained on all (Gaussian) stored historical data
end

EnvNode(;dim=2) = EnvNode(0.0,GPE(Matrix{Float64}(undef,dim,0),Float64[],MeanZero(),SE(zeros(dim),0.0)))

update_observations(env::EnvNode, X::Matrix, info::Float64) = GP(hcat(env.gp.x,float(X)),hcat(gp.env.y, info),env.gp.mean,env.gp.kernel)

predict_env(env::EnvNode, env_sites::Matrix{Float64}) = predict_f(env.gp, env_sites)

export EnvNode, update_observations, predict_env

end