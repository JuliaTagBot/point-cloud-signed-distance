module Models

import StaticArrays: SVector, @SVector
import DataStructures: OrderedDict
using RigidBodyDynamics
import RigidBodyTreeInspector
import Flash
import Flash: BodyGeometry, Manipulator
import GeometryTypes: vertices, AbstractMesh
import CoordinateTransformations: IdentityTransformation,
                                  AbstractAffineMap,
                                  transform_deriv
import EnhancedGJK
import Rotations
import Quaternions

typealias Point3DS{T} Point3D{SVector{3, T}}

function two_link_arm(deformable::Bool=false)
    surfaces = Vector{BodyGeometry}()

    link_length = 1.0
    radius = 0.1

    mechanism = Mechanism(RigidBody{Float64}("world"))
    parent = root_body(mechanism)
    surface_points = Vector{Point3DS{Float64}}()
    skeleton_points = Vector{Point3DS{Float64}}()

    for i = 1:2
        joint = Joint("joint$(i)", Revolute(SVector{3,Float64}(0,0,1)))
        if i > 1
            joint_to_parent = Transform3D(joint.frameBefore, parent.frame, SVector{3,Float64}(link_length, 0., 0))
        else
            joint_to_parent = Transform3D(Float64, joint.frameBefore, parent.frame)
        end
        body = RigidBody(rand(SpatialInertia{Float64}, CartesianFrame3D("body$(i)")))
        body_to_joint = Transform3D(Float64, body.frame, joint.frameAfter)
        attach!(mechanism, parent, joint, joint_to_parent, body, body_to_joint)
        parent = body

        for x = linspace(0.3*link_length, 0.7*link_length, 3)
            for y = [-radius; radius]
                for z = [-radius; radius]
                    push!(surface_points, Point3D(body.frame, SVector(x, y, z)))
                end
            end
            y = 0.0
            for z = [-sqrt(2) * radius, sqrt(2) * radius]
                push!(surface_points, Point3D(body.frame, SVector(x, y, z)))
            end

        end
        if i == 1
            for z = [-sqrt(2) * radius, sqrt(2) * radius]
                push!(surface_points, Point3D(body.frame,
                    SVector(link_length, 0, z)))
            end
            push!(surface_points, Point3D(body.frame, SVector(0., 0, 0)))
        elseif i == 2
            push!(surface_points, Point3D(body.frame, SVector(link_length, 0., 0)))
        end

        for x = linspace(0.2*link_length, 0.8*link_length, 3)
            push!(skeleton_points, Point3D(body.frame, SVector(x, 0., 0)))
        end
    end
    push!(surfaces, Flash.RigidInterpolatingSkin(surface_points, skeleton_points))

    Manipulator(mechanism, surfaces)
end

function beanbag()
    # geometries = OrderedDict{RigidBody{Float64}, BodyGeometry{Float64}}()

    mechanism = Mechanism(RigidBody{Float64}("world"))
    parent = root_body(mechanism)

    joint = Joint("joint1", QuaternionFloating{Float64}())
    joint_to_parent = Transform3D(Float64, joint.frameBefore, parent.frame)
    body = RigidBody(rand(SpatialInertia{Float64}, CartesianFrame3D("body1")))
    body_to_joint = Transform3D(Float64, body.frame, joint.frameAfter)
    attach!(mechanism, parent, joint, joint_to_parent, body, body_to_joint)

    surface_points = Vector{Point3DS{Float64}}()
    skeleton_points = [Point3D(body.frame, @SVector [0.0, 0.0, 0.0])]
    for axis = 1:3
        for s = [-1; 1]
            x = [0., 0, 0]
            x[axis] = s
            push!(surface_points, Point3D(body.frame, SVector(x...)))
        end
    end
    # geometries[body] = BodyGeometry(surface_points, skeleton_points)
    surfaces = Flash.BodyGeometry[Flash.DeformableInterpolatingSkin(surface_points,
                                                  skeleton_points)]
    Manipulator(mechanism, surfaces)
end

function squishable()
    surfaces = Vector{BodyGeometry}()

    mechanism = Mechanism(RigidBody{Float64}("world"))
    parent = root_body(mechanism)

    joint = Joint("joint1", QuaternionFloating{Float64}())
    joint_to_parent = Transform3D(Float64, joint.frameBefore, parent.frame)
    body = RigidBody(rand(SpatialInertia{Float64}, CartesianFrame3D("squishable_body")))
    body_to_joint = Transform3D(Float64, body.frame, joint.frameAfter)
    attach!(mechanism, parent, joint, joint_to_parent, body, body_to_joint)

    surface_points = Vector{Point3DS{Float64}}()
    skeleton_points = [Point3D(body.frame, @SVector [0.0, 0.0, 0.0])]
    radii = @SVector [0.44/2, 0.40/2, 0.30/2]
    for axis = 1:3
        for i_sign = [-1, 1]
            for j_sign = [-1, 1]
                for theta = [pi/4]
                    v = [0., 0, 0]
                    v[axis] = 1
                    x = [0., 0, 0]
                    i = mod(axis, 3) + 1
                    j = mod(i, 3) + 1
                    a = radii[i] * 1.25
                    b = radii[j] * 1.25
                    x[i] = i_sign * sqrt(a^2 * b^2 / (a^2 * tan(theta)^2 + b^2))
                    x[j] = j_sign * sqrt(b^2 * (1 - b^2 / (a^2 * tan(theta)^2 + b^2)))
                    point = SVector(x...)
                    push!(surface_points, Point3D(body.frame, point))
                end
            end
        end
    end
    surfaces = BodyGeometry[Flash.DeformableInterpolatingSkin(surface_points, skeleton_points)]
    Manipulator(mechanism, surfaces)
end

to_quaternion(::UniformScaling) = Quaternions.Quaternion(1.0, 0, 0, 0)
to_quaternion(::IdentityTransformation) = Quaternions.Quaternion(1.0, 0, 0, 0)
to_quaternion(transform::AbstractAffineMap) = to_quaternion(transform_deriv(transform, SVector(0., 0., 0.)))
function to_quaternion(mat::AbstractMatrix)
    quat = Rotations.Quat(mat)
    Quaternions.Quaternion(quat.w, quat.x, quat.y, quat.z)
end

function extract_convex_surfaces(mechanism, vis_data)
    surfaces = Vector{Flash.BodyGeometry}()
    for (frame, link) in vis_data
        for geometrydata in link
            @assert geometrydata.transform == IdentityTransformation()
            push!(surfaces, Flash.ConvexGeometry(geometrydata.geometry, frame))
        end
    end
    surfaces
end

function load_urdf(filename; package_path::Vector=RigidBodyTreeInspector.ros_package_path())
    mechanism = RigidBodyDynamics.parse_urdf(Float64, filename);
    vis_data = RigidBodyTreeInspector.parse_urdf_visuals(filename, mechanism; package_path=package_path);
    surfaces = extract_convex_surfaces(mechanism, vis_data)
    Flash.Manipulator(mechanism, surfaces);
end

function merge!(manip1::Manipulator, manip2::Manipulator)
    RigidBodyDynamics.attach!(manip1.mechanism, RigidBodyDynamics.root_body(manip1.mechanism), manip2.mechanism)
    manip1.surfaces = vcat(manip1.surfaces, manip2.surfaces)
    manip1
end


end
