module Models

import StaticArrays: SVector, @SVector
import DataStructures: OrderedDict
using RigidBodyDynamics
import Flash: BodyGeometry, DeformableGeometry, RigidGeometry, Manipulator

function two_link_arm(deformable::Bool=false)
    geometries = OrderedDict{RigidBody{Float64}, BodyGeometry{Float64}}()

    link_length = 1.0
    radius = 0.1

    mechanism = Mechanism(RigidBody{Float64}("world"))
    parent = root_body(mechanism)

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

        surface_points = Vector{Point3D{Float64}}()
        skeleton_points = Vector{Point3D{Float64}}()
        for x = linspace(0.1*link_length, 0.9*link_length, 3)
            for y = [-radius; radius]
                for z = [-radius; radius]
                    push!(surface_points, Point3D(body.frame, SVector(x, y, z)))
                end
            end
        end
        if i == 1
            push!(surface_points, Point3D(body.frame, SVector(0., 0, 0)))
        elseif i == 2
            push!(surface_points, Point3D(body.frame, SVector(link_length, 0., 0)))
        end
        # surface_geometry = HomogenousMesh(surface_points, convex_hull(surface_points))
        # surface_geometry_data = GeometryData(surface_geometry, tformeye(3))

        for x = linspace(0.2*link_length, 0.8*link_length, 3)
            push!(skeleton_points, Point3D(body.frame, SVector(x, 0., 0)))
        end

        geometries[body] = BodyGeometry(surface_points, skeleton_points,
            deformable? DeformableGeometry() : RigidGeometry())
    end

    # surface_groups = [[g] for g in keys(geometries)]
    surface_groups = [collect(keys(geometries))]

    Manipulator(mechanism, geometries, surface_groups)
end

function beanbag()
    geometries = OrderedDict{RigidBody{Float64}, BodyGeometry{Float64}}()

    mechanism = Mechanism(RigidBody{Float64}("world"))
    parent = root_body(mechanism)

    joint = Joint("joint1", QuaternionFloating())
    joint_to_parent = Transform3D(Float64, joint.frameBefore, parent.frame)
    body = RigidBody(rand(SpatialInertia{Float64}, CartesianFrame3D("body1")))
    body_to_joint = Transform3D(Float64, body.frame, joint.frameAfter)
    attach!(mechanism, parent, joint, joint_to_parent, body, body_to_joint)

    surface_points = Vector{Point3D{Float64}}()
    skeleton_points = [Point3D(body.frame, @SVector [0.0, 0.0, 0.0])]
    for axis = 1:3
        for s = [-1; 1]
            x = [0., 0, 0]
            x[axis] = s
            push!(surface_points, Point3D(body.frame, SVector(x...)))
        end
    end
    geometries[body] = BodyGeometry(surface_points, skeleton_points, DeformableGeometry())

    Manipulator(mechanism, geometries)
end

function squishable()
    geometries = OrderedDict{RigidBody{Float64}, BodyGeometry{Float64}}()

    mechanism = Mechanism(RigidBody{Float64}("world"))
    parent = root_body(mechanism)

    joint = Joint("joint1", QuaternionFloating())
    joint_to_parent = Transform3D(Float64, joint.frameBefore, parent.frame)
    body = RigidBody(rand(SpatialInertia{Float64}, CartesianFrame3D("body1")))
    body_to_joint = Transform3D(Float64, body.frame, joint.frameAfter)
    attach!(mechanism, parent, joint, joint_to_parent, body, body_to_joint)

    surface_points = Vector{Point3D{Float64}}()
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
    geometries[body] = BodyGeometry(surface_points, skeleton_points, DeformableGeometry())
    Manipulator(mechanism, geometries)
end

end
