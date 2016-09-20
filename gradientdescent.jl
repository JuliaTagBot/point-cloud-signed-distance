module GradientDescent

using MathProgBase
using ForwardDiff

type UnconstrainedOptimization{F <: Function} <: MathProgBase.AbstractNLPEvaluator
    func::F
end

function MathProgBase.initialize(d::UnconstrainedOptimization, requested_features::Vector{Symbol})
end

MathProgBase.features_available(d::UnconstrainedOptimization) = [:Grad]

MathProgBase.jac_structure(d::UnconstrainedOptimization) = (tuple(), tuple())

function MathProgBase.eval_grad_f(d::UnconstrainedOptimization, g, x)
    ForwardDiff.gradient!(g, d.func, x)
end

function MathProgBase.eval_f(d::UnconstrainedOptimization, x)
    d.func(x)
end

end
