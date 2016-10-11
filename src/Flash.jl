module Flash

import DrakeVisualizer: contour_mesh, draw, Visualizer, Link, GeometryData
using RigidBodyDynamics
import RigidBodyDynamics: set_configuration!
using LCMGL
import GeometryTypes
import GeometryTypes: HomogenousMesh, Vec, vertices
import SpatialFields: InterpolatingSurface, XSquaredLogX, XCubed
import StaticArrays: SVector, @SVector
using CoordinateTransformations
using Rotations
import ColorTypes
import ForwardDiff
import ForwardDiff: value
import DataStructures: OrderedDict
import Base: convert, flatten, show

convert(::Type{AffineMap}, T::Transform3D) = AffineMap(RigidBodyDynamics.rotationmatrix_normalized_fsa(T.rot), T.trans)

value{T}(x::AbstractArray{T}) = map(value, x)
value(tform::AbstractAffineMap) = AffineMap(value(transform_deriv(tform)), value(tform(SVector{3, Float64}(0, 0, 0))))

type BodyGeometry{T}
    surface_points::Vector{Point3D{SVector{3, T}}}
    skeleton_points::Vector{Point3D{SVector{3, T}}}
end

abstract SurfaceType

immutable DeformableSkin <: SurfaceType end
immutable RigidSkin <: SurfaceType end
immutable RigidPolytope <: SurfaceType end

type Surface{T}
    geometries::OrderedDict{RigidBody{T}, BodyGeometry{T}}
    surface_type::SurfaceType
end

show(io::IO, surface::Surface) = print(io, "$(surface.surface_type) surface with $(length(surface.geometries)) geometries")

type Manipulator{T}
    mechanism::Mechanism{T}
    surfaces::Vector{Surface{T}}
end

show(io::IO, manip::Manipulator) = print(io, "Manipulator with $(length(bodies(manip.mechanism))) links and $(length(manip.surfaces)) surfaces")

typealias View{T} SubArray{T, 1, Array{T, 1}, Tuple{UnitRange{Int64}}, true}

immutable ManipulatorState{ParamType, ConfigurationType, DeformationType}
    manipulator::Manipulator{ParamType}
    mechanism_state::MechanismState{ConfigurationType}
    deformations::Dict{BodyGeometry{ParamType}, Vector{View{DeformationType}}}
    deformation_data::Vector{DeformationType}
end

num_deformations(geom::BodyGeometry, surface_type::DeformableSkin) = length(geom.surface_points)
num_deformations(geom::BodyGeometry, surface_type::RigidSkin) = 0
num_deformations(geom::BodyGeometry, surface_type::RigidPolytope) = 0

num_deformations(manip::Manipulator) = sum(surface -> sum(geometry -> 3 * num_deformations(geometry, surface.surface_type), values(surface.geometries)), manip.surfaces)
num_states(manip::Manipulator) = num_positions(manip.mechanism) + num_deformations(manip)

function ManipulatorState{T}(manipulator::Manipulator{T}, ConfigurationType::DataType=Float64,
                          DeformationType::DataType=Float64)
    typealias DeformationView View{DeformationType}

    mechanism_state = MechanismState(ConfigurationType, manipulator.mechanism)
    deformation_data = zeros(DeformationType, num_deformations(manipulator))
    deformations = Dict{BodyGeometry{T}, Vector{DeformationView}}()
    offset = 0
    for surface in manipulator.surfaces
        for geometry in values(surface.geometries)
            body_deformations = Vector{DeformationView}()
            for i in 1:num_deformations(geometry, surface.surface_type)
                push!(body_deformations, view(deformation_data, offset+(1:3)))
                offset += 3
            end
            deformations[geometry] = body_deformations
        end
    end
    ManipulatorState{T, ConfigurationType, DeformationType}(manipulator,
        mechanism_state,
        deformations,
        deformation_data)
end

function set_configuration!{P, C, D}(state::ManipulatorState{P, C, D}, joint_angles::AbstractVector{C})
    set_configuration!(state.mechanism_state, joint_angles)
end

function link_origins(state::MechanismState)
    transforms = [convert(AffineMap, transform_to_root(state, body.frame)) for body in bodies(state.mechanism)]
end

link_origins(state::ManipulatorState) = link_origins(state.mechanism_state)

function link_origins{T}(arm::Manipulator, joint_angles::AbstractVector{T})
    state = MechanismState(T, arm.mechanism)
    set_configuration!(state, joint_angles)
    link_origins(state)
end

function surface_points{P, C, D}(state::ManipulatorState{P, C, D}, geometry::BodyGeometry{P})
    typealias T promote_type(C, D)
    surface_points = Vector{SVector{3, T}}(length(geometry.surface_points))
    root = root_frame(state.manipulator.mechanism)
    deformations = state.deformations[geometry]
    if length(deformations) > 0
        for (i, point) in enumerate(geometry.surface_points)
            deformation = FreeVector3D(point.frame,
                convert(SVector{3, T}, deformations[i]))
            surface_points[i] = RigidBodyDynamics.transform(state.mechanism_state,
                                              point + deformation,
                                              root).v
        end
    else
        for (i, point) in enumerate(geometry.surface_points)
            surface_points[i] = RigidBodyDynamics.transform(state.mechanism_state,
                                              point,
                                              root).v
        end
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

function skin(state::ManipulatorState, surface::Surface, surf_type::Union{DeformableSkin, RigidSkin})
    surface_pts = collect(flatten(map(geometry ->
        surface_points(state, geometry), values(surface.geometries))))
    skeleton_pts = collect(flatten(map(geometry ->
        skeleton_points(state, geometry), values(surface.geometries))))
    points = vcat(surface_pts, skeleton_pts)
    signed_distances = vcat(zeros(length(surface_pts)), -1 + zeros(length(skeleton_pts)))
    InterpolatingSurface(points, signed_distances, XSquaredLogX())
end

any_inside{T}(points::Vector{Vec{3, T}}) = points[1]

immutable SimplexSurface{T} <: Function
    simplex::GeometryTypes.FlexibleSimplex{Vec{3, T}}
end

(surface::SimplexSurface{T}){T}(x) = GeometryTypes.gjk(surface.simplex, Vec{3, T}(x[1], x[2], x[3]))

function skin{P, T}(state::ManipulatorState{P, T, T}, surface::Surface, surf_type::RigidPolytope)
    surface_pts = flatten(map(geometry ->
        surface_points(state, geometry), values(surface.geometries)))
    simplex = GeometryTypes.FlexibleSimplex([Vec{3, T}(pt[1], pt[2], pt[3]) for pt in surface_pts])
    SimplexSurface(simplex)
end

skin(state::ManipulatorState, surface::Surface) = skin(state, surface, surface.surface_type)

function surfaces(state::ManipulatorState)
    map(surface -> skin(state, surface), state.manipulator.surfaces)
end

function skin(state::ManipulatorState)
    all_surfaces = surfaces(state)
    x -> minimum(s(x) for s in all_surfaces)
end

function drawing_region(surface::InterpolatingSurface)
    lb = @SVector [minimum(p[i] for p in surface.points) for i in 1:3]
    ub = @SVector [maximum(p[i] for p in surface.points) for i in 1:3]
    widths = ub - lb
    lb - 0.5 * widths, ub + 0.5 * widths
end

function drawing_region(surface::SimplexSurface)
    lb = SVector{3, Float64}(minimum(vertices(surface.simplex))...)
    ub = SVector{3, Float64}(maximum(vertices(surface.simplex))...)
    widths = ub - lb
    lb - 0.1 * widths, ub + 0.1 * widths
end

function draw{D, C}(state::ManipulatorState{D, C}, draw_skin::Bool=true)
    LCMGLClient("points") do lcmgl
        for surface in state.manipulator.surfaces
            for geometry in values(surface.geometries)
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
        end
        switch_buffer(lcmgl)
    end

    if draw_skin
        links = Link[]
        all_surfaces = surfaces(state)
        for (i, surface) in enumerate(all_surfaces)
            geometries = []
            for iso_level = [0.0]
                lb, ub = drawing_region(surface)
                push!(geometries, GeometryData(contour_mesh(surface, lb, ub, iso_level, 0.1)))
            end
            push!(links, Link(geometries, "skin_$(i)"))
        end
        Visualizer(links)
    end
end

include("depthsensors.jl")
include("depthdata.jl")
include("models.jl")
include("gradientdescent.jl")
include("tracking.jl")

end
