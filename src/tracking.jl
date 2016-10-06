module Tracking

import Flash
import Flash: Manipulator, num_states
import Flash.GradientDescent: CostFunctor
import SimpleGradientDescent: getmodel, setwarmstart!, optimize!, NaiveSolver, getsolution

function estimate_state{PointType}(manipulator::Manipulator,
                                   sensed_points::Vector{PointType},
                                   x_estimated::Vector{Float64};
                                   callback::Function=x -> (),
                                   solver=NaiveSolver(num_states(manipulator),
                                                      rate=0.1,
                                                      max_step=0.5,
                                                      iteration_limit=30))
    cost = CostFunctor(manipulator, sensed_points)
    function wrapped_cost{T}(x::AbstractVector{T})
        c = cost(x)
        callback(x, c)
        c / length(sensed_points)
    end
    num_vars = num_states(manipulator)
    opt = getmodel(wrapped_cost, num_vars, solver)
    setwarmstart!(opt, x_estimated)
    optimize!(opt)
    getsolution(opt)
end

end
