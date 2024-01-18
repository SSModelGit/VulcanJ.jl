using VulcanJ
using GaussianProcesses

gen_prob(xlim::Integer,ylim::Integer) = InfoProblem(sxy=[xlim,ylim])
init_state(x::Matrix{T}) where T<:Integer = InitialNode(x)
gen_init_env() = VulcanJ.EnvNode(dim=2);

function gen_pseudo_obs(nobs::Integer, p::InfoProblem, f::Function)
    obs_locs = stack([float.(rand(1:p.Xlims[1],nobs)), float.(rand(1:p.Xlims[2],nobs))], dims=1)
    obs = vec(mapslices(f,obs_locs,dims=1))
    (obs_locs, obs)
end

simple_gp_train(obs_locs::Matrix{Float64}, obs::Vector{Float64}, gp::GPE) = GP(obs_locs, obs, gp.mean, gp.kernel)

simple_gp_train(gen_pseudo_obs(3,gen_prob(3, 3), x->x[1]^2+x[2])...,gen_init_env().gp)

# gen_init_state(coord::Vector{Integer}) = InfoNode(coord, NaN, )

# sitelist = cellsites(prob)
# sitelist = stack([[float(x),float(y)] for x in 1:xlim for y in 1:ylim], dims=2);

#= den=VulcanJ.EnvNode(dim=2);

nobs=3;
obs_locs = stack([float.(rand(1:xlim,nobs)), float.(rand(1:ylim,nobs))]);
obs = vec(mapslices(x->x[1]^2+x[2],obs_locs,dims=2));

if size(obs_locs)[2]==size(obs)
    @show gpn=GP(obs_locs, obs, gpe.mean, gpe.kernel);
else
    @show gpn=GP(obs_locs', obs, gpe.mean, gpe.kernel);
end

μ,Σ=predict_f(gpn,sitelist); =#