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
