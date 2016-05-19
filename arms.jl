module Arms

import DrakeVisualizer: GeometryData, draw, Visualizer, Link
using RigidBodyDynamics
import RigidBodyDynamics: set_configuration!
using AffineTransforms
import GeometryTypes: HyperRectangle, HyperSphere, Vec, Point, HomogenousMesh
import SpatialFields: InterpolatingSurface, XSquaredLogX
import Quaternions: axis, angle
import ColorTypes
import ForwardDiff: value
import DataStructures: OrderedDict
import Base: convert

convert(::Type{AffineTransform}, T::Transform3D) = tformtranslate(convert(Vector, T.trans)) * tformrotate(axis(T.rot), angle(T.rot))

value(x::Real) = x

type Limb
    surface_points::Vector{Point3D}
    skeleton_points::Vector{Point3D}
end

type Model
    mechanism::Mechanism
    limbs::OrderedDict{RigidBody, Limb}
end

type ModelState{ConfigurationType, DeformationType}
    mechanism_state::MechanismState{ConfigurationType}
    limb_deformations::Vector{Vector{FreeVector3D{DeformationType}}}
end

ModelState{C, D}(mechanism_state::MechanismState{C}, deformations::Vector{Vector{FreeVector3D{D}}}) = ModelState{C, D}(mechanism_state, deformations)

function ModelState{C, D}(model::Model, joint_angles::Vector{C}, deformations::Vector{Vector{Vec{3, D}}})
    mechanism_state = MechanismState(C, model.mechanism)
    set_configuration!(mechanism_state, joint_angles)
    zero_velocity!(mechanism_state)
    deformations_in_frames = Vector{Vector{FreeVector3D{D}}}()
    for (i, (body, limb)) in enumerate(model.limbs)
        limb_deformations = deformations[i]
        limb_deformations_in_frame = Vector{FreeVector3D{D}}()
        for d in limb_deformations
            push!(limb_deformations_in_frame, FreeVector3D(body.frame, d))
        end
        push!(deformations_in_frames, limb_deformations_in_frame)
    end
    ModelState(mechanism_state, deformations_in_frames)
end

function ModelState(model::Model, ConfigurationType=Float64, DeformationType=Float64)
    joint_angles = zeros(ConfigurationType, num_positions(model.mechanism))
    deformations = Vector{Vec{3, DeformationType}}[Vec{3, DeformationType}[0 for point in limb.surface_points] for (body, limb) in model.limbs]
    ModelState(model, joint_angles, deformations)
end

function set_configuration!{C, D}(state::ModelState{C, D}, joint_angles::Vector{C})
    set_configuration!(state.mechanism_state, joint_angles)
end

function two_link_arm()
    limbs = OrderedDict{RigidBody, Limb}()

    link_length = 1.0
    radius = 0.1

    mechanism = Mechanism{Float64}("world")
    parent = root_body(mechanism)

    for i = 1:2
        joint = Joint("joint$(i)", Revolute(Vec(0.,0,1)))
        joint_to_parent = Transform3D(joint.frameBefore, parent.frame, Vec(link_length, 0., 0))
        body = RigidBody(rand(SpatialInertia{Float64}, CartesianFrame3D("body$(i)")))
        body_to_joint = Transform3D(Float64, body.frame, joint.frameAfter)
        attach!(mechanism, parent, joint, joint_to_parent, body, body_to_joint)
        parent = body

        surface_points = Vector{Point3D}()
        skeleton_points = Vector{Point3D}()
        for x = linspace(0.1*link_length, 0.9*link_length, 3)
            for y = [-radius; radius]
                for z = [-radius; radius]
                    push!(surface_points, Point3D(body.frame, Vec(x, y, z)))
                end
            end
        end
        if i == 1
            push!(surface_points, Point3D(body.frame, Vec(0., 0, 0)))
        elseif i == 2
            push!(surface_points, Point3D(body.frame, Vec(link_length, 0., 0)))
        end
        # surface_geometry = HomogenousMesh(surface_points, convex_hull(surface_points))
        # surface_geometry_data = GeometryData(surface_geometry, tformeye(3))

        for x = linspace(0.2*link_length, 0.8*link_length, 3)
            push!(skeleton_points, Point3D(body.frame, Vec(x, 0., 0)))
        end

        limbs[body] = Limb(surface_points, skeleton_points)
    end

    Model(mechanism, limbs)
end

# function convert(::Type{DrakeVisualizer.Robot}, arm::Model)
#     links = [DrakeVisualizer.Link([limb.surface_geometry], body.frame.name) for (body, limb) in arm.limbs]
#     return DrakeVisualizer.Robot(links)
# end
#
function draw(arm::Model, state::ModelState)
    links = Vector{Link}()
    for (i, (body, limb)) in enumerate(arm.limbs)
        geometries = Vector{GeometryData}()
        for (j, surface_point) in enumerate(limb.surface_points)
            pose = surface_point + state.limb_deformations[i][j]
            push!(geometries, GeometryData(HyperSphere(Point(0.,0,0), 0.01), tformtranslate(convert(Vector, pose.v)), ColorTypes.RGBA{Float64}(1.0, 0.0, 0.0, 0.5)))
        end
        for skeleton_point in limb.skeleton_points
            push!(geometries, GeometryData(HyperSphere(Point(0.,0,0), 0.01), tformtranslate(convert(Vector, skeleton_point.v)), ColorTypes.RGBA{Float64}(0.0, 0.0, 1.0, 0.5)))
        end
        push!(links, Link(geometries, body.frame.name))
    end

    surface = skin(arm, state)
    push!(links, Link([GeometryData(convert(HomogenousMesh, surface))], "skin"))

    vis = Visualizer(links)

    origins = link_origins(arm, state)
    push!(origins, tformeye(3));
    draw(vis, origins)
end


function link_origins(arm::Model, state::ModelState)
    # state = MechanismState(Float64, arm.mechanism)
    transforms = AffineTransform[transform_to_root(state.mechanism_state, body.frame) for body in keys(arm.limbs)]
end


function skin{D, C}(arm::Model, state::ModelState{D, C})
    surface_points = Point{3, promote_type(D, C)}[]
    skeleton_points = Point{3, promote_type(D, C)}[]

    for (i, (body, limb)) in enumerate(arm.limbs)
        for (j, point) in enumerate(limb.surface_points)
            push!(surface_points, RigidBodyDynamics.transform(state.mechanism_state, point + state.limb_deformations[i][j], root_frame(arm.mechanism)).v)
        end
    end
    for (i, (body, limb)) in enumerate(arm.limbs)
        for (j, point) in enumerate(limb.skeleton_points)
            push!(skeleton_points, RigidBodyDynamics.transform(state.mechanism_state, point, root_frame(arm.mechanism)).v)
        end
    end

    points = vcat(surface_points, skeleton_points)
    values = vcat(zeros(length(surface_points)), -1 + zeros(length(skeleton_points)))
    skin = InterpolatingSurface(points, values, XSquaredLogX())
end


end
