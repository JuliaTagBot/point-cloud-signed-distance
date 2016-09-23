module Arms

import DrakeVisualizer: contour_mesh, draw, Visualizer, Link, GeometryData
using RigidBodyDynamics
import RigidBodyDynamics: set_configuration!, zero_configuration
using LCMGL
import GeometryTypes: HomogenousMesh
import SpatialFields: InterpolatingSurface, XSquaredLogX
import StaticArrays: SVector, @SVector
using CoordinateTransformations
using Rotations
import ColorTypes
import ForwardDiff: value
import DataStructures: OrderedDict
import Base: convert, flatten

convert(::Type{AffineMap}, T::Transform3D) = AffineMap(RigidBodyDynamics.rotationmatrix_normalized_fsa(T.rot), T.trans)

value{T}(x::AbstractArray{T}) = map(value, x)
value(tform::AbstractAffineMap) = AffineMap(value(transform_deriv(tform)), value(tform(SVector{3, Float64}(0, 0, 0))))

abstract BodyGeometryType

type BodyGeometry{T}
    surface_points::Vector{Point3D{T}}
    skeleton_points::Vector{Point3D{T}}
    geometry_type::BodyGeometryType
end

immutable DeformableGeometry <: BodyGeometryType end
immutable RigidGeometry <: BodyGeometryType end

# type Limb
#     surface_points::Vector{Point3D}
#     skeleton_points::Vector{Point3D}
# end
#

type Manipulator{T}
    mechanism::Mechanism{T}
    geometries::OrderedDict{RigidBody{T}, BodyGeometry{T}}
    surface_groups::Vector{Vector{RigidBody{T}}}
end

typealias View{T} SubArray{T, 1, Array{T, 1}, Tuple{UnitRange{Int64}}, true}

immutable ManipulatorState{ParamType, ConfigurationType, DeformationType}
    manipulator::Manipulator{ParamType}
    mechanism_state::MechanismState{ConfigurationType}
    deformations::Dict{BodyGeometry{ParamType}, Vector{View{DeformationType}}}
    deformation_data::Vector{DeformationType}
end

num_deformations(geom::BodyGeometry, geom_type::DeformableGeometry) = length(geom.surface_points)
num_deformations(geom::BodyGeometry, geom_type::RigidGeometry) = 0
num_deformations(geom::BodyGeometry) = num_deformations(geom, geom.geometry_type)

function ManipulatorState{T}(manipulator::Manipulator{T}, ConfigurationType::DataType=Float64,
                          DeformationType::DataType=Float64)
    typealias DeformationView View{DeformationType}

    mechanism_state = MechanismState(ConfigurationType, manipulator.mechanism)
    deformation_data = Vector{DeformationType}()
    deformations = Dict{BodyGeometry{T}, Vector{DeformationView}}()
    offset = 0
    for (body, geometry) in manipulator.geometries
        body_deformations = Vector{DeformationView}()
        for i in 1:num_deformations(geometry)
            append!(deformation_data, zeros(DeformationType, 3))
            push!(body_deformations, view(deformation_data, offset+(1:3)))
            offset += 3
        end
        deformations[geometry] = body_deformations
    end
    ManipulatorState{T, ConfigurationType, DeformationType}(manipulator,
        mechanism_state,
        deformations,
        deformation_data)
end

# function ManipulatorState{C, D}(model::Manipulator, joint_angles::Vector{C},
#                                 deformations::Vector{Vector{SVector{3, D}}})
#     mechanism_state = MechanismState(C, model.mechanism)
#     set_configuration!(mechanism_state, joint_angles)
#     zero_velocity!(mechanism_state)
#     deformations_in_frames = Vector{Vector{FreeVector3D{D}}}()
#     for (i, (body, limb)) in enumerate(model.limbs)
#         limb_deformations = deformations[i]
#         limb_deformations_in_frame = Vector{FreeVector3D{D}}()
#         for d in limb_deformations
#             push!(limb_deformations_in_frame, FreeVector3D(body.frame, d))
#         end
#         push!(deformations_in_frames, limb_deformations_in_frame)
#     end
#     ManipulatorState{C, D}(mechanism_state, deformations_in_frames)
# end

# function zero_configuration(model::Manipulator, ConfigurationType=Float64, DeformationType=Float64)
#     joint_angles = Vector{ConfigurationType}()
#     for vertex in model.mechanism.toposortedTree[2:end]
#         joint = vertex.edgeToParentData
#         append!(joint_angles, RigidBodyDynamics.zero_configuration(joint, ConfigurationType))
#     end
#     deformations = Vector{SVector{3, DeformationType}}[SVector{3, DeformationType}[0 for i in 1:num_deformations(geometry)] for (body, geometry) in model.geometries]
#     joint_angles, deformations
# end


# function ManipulatorState(model::Manipulator, ConfigurationType::DataType=Float64,
#                           DeformationType::DataType=Float64)
#     joint_angles, deformations = zero_configuration(model, ConfigurationType, DeformationType)
#     ManipulatorState(model, joint_angles, deformations)
# end

function set_configuration!{P, C, D}(state::ManipulatorState{P, C, D}, joint_angles::AbstractVector{C})
    set_configuration!(state.mechanism_state, joint_angles)
end

function link_origins(arm::Manipulator, state::MechanismState)
    transforms = [convert(AffineMap, transform_to_root(state, body.frame)) for body in keys(arm.limbs)]
end

link_origins(arm::Manipulator, state::ManipulatorState) = link_origins(arm, state.mechanism_state)

function link_origins{T}(arm::Manipulator, joint_angles::AbstractVector{T})
    state = MechanismState(T, arm.mechanism)
    set_configuration!(state, joint_angles)
    link_origins(arm, state)
end

function surface_points{P, C, D}(state::ManipulatorState{P, C, D}, geometry::BodyGeometry{P})
    typealias T promote_type(C, D)
    surface_points = Vector{SVector{3, T}}(length(geometry.surface_points))
    root = root_frame(state.manipulator.mechanism)
    deformations = state.deformations[geometry]
    for (i, point) in enumerate(geometry.surface_points)
        deformation = FreeVector3D(point.frame,
            convert(SVector{3, T}, deformations[i]))
        surface_points[i] = RigidBodyDynamics.transform(state.mechanism_state,
                                          point + deformation,
                                          root).v
    end
    surface_points
end

function skeleton_points{P, C, D}(state::ManipulatorState{P, C, D}, geometry::BodyGeometry{P})
    typealias T promote_type(C, D)
    skeleton_points = Vector{SVector{3, T}}(length(geometry.skeleton_points))
    root = root_frame(state.manipulator.mechanism)
    for (i, point) in enumerate(geometry.skeleton_points)
        skeleton_points[i] = RigidBodyDynamics.transform(state.mechanism_state,
                                          point,
                                          root).v
    end
    skeleton_points
end

function skin{P, C, D}(state::ManipulatorState{P, C, D}, bodies::Vector{RigidBody{P}})
    surface = collect(flatten(map(body ->
        surface_points(state, state.manipulator.geometries[body]), bodies)))
    skeleton = collect(flatten(map(body ->
        skeleton_points(state, state.manipulator.geometries[body]), bodies)))

    points = vcat(surface, skeleton)
    values = vcat(zeros(length(surface)), -1 + zeros(length(skeleton)))
    skin = InterpolatingSurface(points, values, XSquaredLogX())
end

function skin(state::ManipulatorState)
    map(bodies -> skin(state, bodies), state.manipulator.surface_groups)
end

function draw{D, C}(state::ManipulatorState{D, C}, draw_skin::Bool=true)
    LCMGLClient("points") do lcmgl
        for geometry in values(state.manipulator.geometries)
            point_size(lcmgl, 10)
            color(lcmgl, 1, 0, 0)
            begin_mode(lcmgl, LCMGL.POINTS)
            for point in surface_points(state, geometry)
                vertex(lcmgl, map(value, point)...)
            end
            end_mode(lcmgl)
            color(lcmgl, 0, 0, 1)
            begin_mode(lcmgl, LCMGL.POINTS)
            for point in skeleton_points(state, geometry)
                vertex(lcmgl, map(value, point)...)
            end
            end_mode(lcmgl)
        end
        switch_buffer(lcmgl)
    end

    if draw_skin
        links = Link[]
        surfaces = skin(state)
        for (i, surface) in enumerate(surfaces)
            geometries = []
            for iso_level = [0.0]
                lb = @SVector [minimum(p[i] for p in surface.points) for i in 1:3]
                ub = @SVector [maximum(p[i] for p in surface.points) for i in 1:3]
                widths = ub - lb
                push!(geometries, GeometryData(contour_mesh(surface, lb - 0.5 * widths, ub + 0.5 * widths, iso_level, 0.1)))
            end
            push!(links, Link(geometries, "skin_$(i)"))
        end
        Visualizer(links)
    end
end

include("models.jl")

end
