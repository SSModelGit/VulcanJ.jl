module EnvironmentGP

using GaussianProcesses

struct EnvNode
    δmi::Float64 # Mutual information gained by entering the current information node
    gp::GPE # Gaussian Process that has been trained on all (Gaussian) stored historical data
end

EnvNode(;dim=2) = EnvNode(0.0,GPE(Matrix{Float64}(undef,dim,0),Float64[],MeanZero(),SE(zeros(dim),0.0)))

function update_observations(gp::GPE, X::Matrix, info::Float64)
    let gp=GP(hcat(gp.x,float(X)),vcat(gp.y, info),gp.mean,gp.kernel)
        optimize!(gp)
        gp
    end
end
update_observations(env::EnvNode, X::Matrix, info::Float64) = update_observations(env.gp, X, info)

predict_gp(gp::GPE, env_sites::Matrix{Float64}) = predict_f(gp, env_sites)

export EnvNode, update_observations, predict_gp

end