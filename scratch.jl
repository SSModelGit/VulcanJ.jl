using VulcanJ
using GaussianProcesses

xlim = 5
ylim = 3

prob = InfoProblem(sxy=[xlim,ylim])
sitelist = cellsites(prob)

# sitelist = stack([[float(x),float(y)] for x in 1:xlim for y in 1:ylim], dims=2);

den=VulcanJ.EnvNode(dim=2);
# δmi_den=VulcanJ.Δmutual_info(den, sitelist);

gpe=GPE();

nobs=3;
obs_locs = stack([float.(rand(1:xlim,nobs)), float.(rand(1:ylim,nobs))]);
obs = vec(mapslices(x->x[1]^2+x[2],obs_locs,dims=2));

if size(obs_locs)[2]==size(obs)
    @show gpn=GP(obs_locs, obs, gpe.mean, gpe.kernel);
else
    @show gpn=GP(obs_locs', obs, gpe.mean, gpe.kernel);
end

μ,Σ=predict_f(gpn,sitelist);

