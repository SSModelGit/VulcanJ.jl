using POMDPs, POMDPTools, MCTS

####################
# Ground Truth Model
####################

function single_volcanic_process(loc::Array{Float64}, center::Matrix{Float64}, lambd::Dict{Symbol, Any})
    exp(-lambd[:l1]*(loc[1]-center[1])^2-lambd[:l2]*(loc[2]-center[2])^2)
end

function composed_caldera_process(loc::Array{Float64}, center::Matrix{Float64}, scale_params::Dict{Symbol, Any})
    mapreduce(x->single_volcanic_process(loc, x, scale_params), +, [center+scale_params[:caldera_r].*[cos(i);sin(i);;] for i in 0:2π/12:2π])
end

function volcanic_process(loc::Array{Float64}, center::Matrix{Float64}, scale_params::Dict{Symbol, Any})
    scale_params[:volcano_h]*single_volcanic_process(loc, center, scale_params) + scale_params[:rel_caldera_h]*composed_caldera_process(loc, center, scale_params)
end

generate_prior_data(locs::Matrix{Float64}; proc::Function) = mapslices(proc, locs, dims=1)


#########
# Example
#########

function ex_set_params()
    Dict(:lims=>[10,10],:N=>100,:center=>[5.;5.;;],:volcano_h=>50., :caldera_r=>4., :rel_caldera_h=>1., :l1=>0.01, :l2=>0.01)
end

function ex_priors(params::Dict{Symbol, Any})
    let locs=stack([float(rand(1:params[:lims][1],params[:N])),float(rand(1:params[:lims][2],params[:N]))], dims=1)
        Dict(:X=>locs, :y=>vec(generate_prior_data(locs; proc=x->volcanic_process(x,params[:center],params))+randn(1,params[:N])))
    end
end

function setup(;prior_samples=100)
    let params=ex_set_params(), prior=ex_priors(params), Xinit=[1;1;;], obs=single_volcanic_process(float(Xinit), params[:center], params)
        InfoProblem(initial_info=Dict(:prior=>prior,:Xinit=>Xinit,:obs=>obs))
    end
end

estimator(mdp, s, remaining_depth) = s.env.δmi

function run_mcts()
    let p = setup()
        solver = DPWSolver(n_iterations=1000,depth=5, estimate_value=estimator)
        policy = solve(solver, p)
        action(policy, p.Xinit)
        println("Tree")
        # @show tree = policy.tree
        policy.tree
    end
end