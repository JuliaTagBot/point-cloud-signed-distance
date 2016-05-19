module Arms

import DrakeVisualizer
import DrakeVisualizer: GeometryData, draw, Visualizer
using RigidBodyDynamics
import RigidBodyDynamics: set_configuration!
using AffineTransforms
using GeometryTypes
using PyCall
using SpatialFields
import Quaternions: axis, angle
using LCMGL
import ColorTypes
import ForwardDiff: value
import DataStructures: OrderedDict
import Base: convert

convert(::Type{AffineTransform}, T::Transform3D) = tformtranslate(convert(Vector, T.trans)) * tformrotate(axis(T.rot), angle(T.rot))

function __init__()
    const global spatial = pyimport("scipy.spatial")
end

value(x::Real) = x

function convex_hull{T}(nodes::Vector{Point{3, T}})
    hull = spatial[:ConvexHull](hcat(map(x -> convert(Vector, x), nodes)...)')
    simplices = hull[:simplices]
    simplices += 1

    faces = Face{3, Int, 0}[]
    # Reorient simplices so that normals always point out from the hull
    for i = 1:size(simplices, 1)
        v = Vector{T}[nodes[simplices[i, j]] for j in 1:size(simplices, 2)]

        # Given a face of a convex hull, all of the points in the body must be on the same side of that face. So in theory, we just need to pick one point not on the face and check which side it's on. But to attempt to be robust to numerical issues, we'll actually sum the dot product with the normal for every point in the body, and check whether that sum is positive or negative. If this becomes a performance bottleneck, we can revisit it later.
        n = Point{3, T}(cross(v[2] - v[1], v[3] - v[1]))
        b = dot(n, nodes[simplices[i, 1]])
        total = zero(T)
        for j = 1:length(nodes)
            total += dot(n, nodes[j]) - b
        end
        if total < 0
            # Then the normal is pointing the right way
            push!(faces, Face{3, Int, 0}((simplices[i,:])...))
        else
            # Otherwise the normal is pointing the wrong way, so we flip the face
            push!(faces, Face{3, Int, 0}((simplices[i,end:-1:1])...))
        end
    end
    return faces
end

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
    links = Vector{DrakeVisualizer.Link}()
    for (i, (body, limb)) in enumerate(arm.limbs)
        geometries = Vector{DrakeVisualizer.GeometryData}()
        for (j, surface_point) in enumerate(limb.surface_points)
            pose = surface_point + state.limb_deformations[i][j]
            push!(geometries, DrakeVisualizer.GeometryData(GeometryTypes.HyperSphere(Point(0.,0,0), 0.01), tformtranslate(convert(Vector, pose.v)), ColorTypes.RGBA{Float64}(1.0, 0.0, 0.0, 0.5)))
        end
        for skeleton_point in limb.skeleton_points
            push!(geometries, DrakeVisualizer.GeometryData(GeometryTypes.HyperSphere(Point(0.,0,0), 0.01), tformtranslate(convert(Vector, skeleton_point.v)), ColorTypes.RGBA{Float64}(0.0, 0.0, 1.0, 0.5)))
        end
        push!(links, DrakeVisualizer.Link(geometries, body.frame.name))
    end

    surface = skin(arm, state)
    push!(links, DrakeVisualizer.Link([DrakeVisualizer.GeometryData(convert(HomogenousMesh, surface))], "skin"))

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
    # LCMGLClient("points") do lcmgl
    #     color(lcmgl, 0, 0, 0)
    #     point_size(lcmgl, 5)
    #     begin_mode(lcmgl, LCMGL.POINTS)
    #     for (i, point) in enumerate(points)
    #         if values[i] == 0
    #             color(lcmgl, 0, 0, 0)
    #         else
    #             color(lcmgl, 0, 0, 1)
    #         end
    #         vertex(lcmgl, map(value, convert(Vector, point))...)
    #         # sphere(lcmgl, convert(Vector, point), 0.02, 10, 10)
    #     end
    #     end_mode(lcmgl)
    #     switch_buffer(lcmgl)
    # end
    skin = InterpolatingSurface(points, values, SpatialFields.XSquaredLogX())
end


end
