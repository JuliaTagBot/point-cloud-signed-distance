const deformation_cost_weight = 10

function Base.flatten(state::Flash.ManipulatorState)
    vcat(state.mechanism_state.q, state.deformation_data)
end

function unflatten!{T}(state::Flash.ManipulatorState, data::AbstractVector{T})
    nq = num_positions(state.mechanism_state)
    set_configuration!(state.mechanism_state, data[1:nq])
    state.deformation_data[:] = data[(nq + 1):end]
end

function cost(state::ManipulatorState, sensed_points::AbstractArray)
    skin = Flash.skin(state)
    c = sum(point -> skin(point)^2, sensed_points)
    for deformation_set in values(state.deformations)
        for deformation in deformation_set
            c += deformation_cost_weight * sum(deformation .^ 2)
        end
    end
    c
end

type CostFunctor{ParamType, PointType} <: Function
    manipulator::Manipulator{ParamType}
    sensed_points::Vector{PointType}
    state::Nullable{ManipulatorState}
end

CostFunctor(manipulator::Manipulator, sensed_points::AbstractVector) = CostFunctor(manipulator, sensed_points, Nullable{ManipulatorState}())

function (functor::CostFunctor{ParamType}){ParamType, T}(x::AbstractVector{T})
    if isnull(functor.state) || typeof(get(functor.state)) != ManipulatorState{ParamType, T, T}
        functor.state = Nullable{ManipulatorState}(ManipulatorState(functor.manipulator, T, T))
    end
    state::ManipulatorState{ParamType, T, T} = get(functor.state)
    unflatten!(state, x)
    # @code_warntype cost(state, functor.sensed_points)
    cost(state, functor.sensed_points)
end

function CostAndGradientFunctor(cost::Function)
    (g, x) -> begin
        out = ForwardDiff.GradientResult(first(x), g)
        ForwardDiff.gradient!(out, cost, x)
        ForwardDiff.value(out)
    end
end
