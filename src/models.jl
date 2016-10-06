module Models

import StaticArrays: SVector, @SVector
import DataStructures: OrderedDict
using RigidBodyDynamics
import RigidBodyTreeInspector
import Flash
import Flash: BodyGeometry, Manipulator
import GeometryTypes: vertices

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

        geometries[body] = BodyGeometry(surface_points, skeleton_points)
    end

    Manipulator(mechanism, [Flash.Surface(geometries, deformable? Flash.DeformableSkin() : Flash.RigidSkin())])
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
    geometries[body] = BodyGeometry(surface_points, skeleton_points)

    Manipulator(mechanism, [Flash.Surface(geometries, Flash.DeformableSkin())])
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
    geometries[body] = BodyGeometry(surface_points, skeleton_points)
    Manipulator(mechanism, [Flash.Surface(geometries, Flash.DeformableSkin())])
end

function extract_convex_surfaces(mechanism, vis_data)
    surfaces = Vector{Flash.Surface{Float64}}()
    for (i, node) in enumerate(mechanism.toposortedTree)
        body = node.vertexData
        mesh = vis_data[i].geometry_data[1].geometry
        tform = vis_data[i].geometry_data[1].transform
        verts = vertices(mesh)
        surface_points = [Point3D(body.frame, tform(SVector{3, Float64}(v...))) for v in verts]
        skeleton_points = Vector{Point3D{Float64}}()
        geometries = OrderedDict(body => Flash.BodyGeometry(surface_points, skeleton_points))
        push!(surfaces, Flash.Surface(geometries, Flash.RigidPolytope()))
    end
    surfaces
end

function load_urdf(filename; package_path::Vector=RigidBodyTreeInspector.ros_package_path())
    mechanism = RigidBodyDynamics.parse_urdf(Float64, filename);
    vis_data = RigidBodyTreeInspector.parse_urdf_visuals(filename, mechanism; package_path=package_path);
    surfaces = extract_convex_surfaces(mechanism, vis_data)
    Flash.Manipulator(mechanism, surfaces);
end


end
