module Arms

import DrakeVisualizer
import DrakeVisualizer: GeometryData
using RigidBodyDynamics
using AffineTransforms
using GeometryTypes
using PyCall
using SpatialFields
using LCMGL
import ForwardDiff: value
import DataStructures: OrderedDict
import Base: convert

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

# type Link
#     length
#     surface_geometry::GeometryData
#     skeleton_points::Vector{Point{3, Float64}}
# end

type Limb
    surface_geometry::GeometryData
    skeleton_points::Vector{Point{3, Float64}}
end

type Model
    mechanism::Mechanism
    limbs::OrderedDict{RigidBody, Limb}
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

        surface_points = Point{3, Float64}[]
        skeleton_points = Point{3, Float64}[]
        for x = linspace(0.1*link_length, 0.9*link_length, 3)
            for y = [-radius; radius]
                for z = [-radius; radius]
                    push!(surface_points, Point{3, Float64}(x, y, z))
                end
            end
        end
        if i == 1
            push!(surface_points, Point(0, 0, 0))
        elseif i == 2
            push!(surface_points, Point(link_length, 0, 0))
        end
        surface_geometry = HomogenousMesh(surface_points, convex_hull(surface_points))
        surface_geometry_data = GeometryData(surface_geometry, tformeye(3))

        for x = linspace(0.2*link_length, 0.8*link_length, 3)
            push!(skeleton_points, Point{3, Float64}(x, 0, 0))
        end

        limbs[body] = Limb(surface_geometry_data, skeleton_points)
    end

    Model(mechanism, limbs)
end

# function link_origins(arm::PlanarArm, joint_angles)
#     transforms = Array{AffineTransform}(length(arm.links))
#     transforms[1] = tformrotate([0;0;1], joint_angles[1])
#     for i = 2:length(arm.links)
#         transforms[i] = transforms[i-1] * tformtranslate([arm.links[i-1].length; 0; 0]) * tformrotate([0.;0;1], joint_angles[i])
#     end
#     transforms
# end
#
# function convert(::Type{DrakeVisualizer.Robot}, arm::PlanarArm)
#     links = [DrakeVisualizer.Link([link.surface_geometry], "link_$(i)") for (i, link) in enumerate(arm.links)]
#     return DrakeVisualizer.Robot(links)
# end
#
# function skin{T}(arm::PlanarArm, joint_angles::Vector{T})
#     origins = link_origins(arm, joint_angles)
#     surface_points = Point{3, T}[]
#     skeleton_points = Point{3, T}[]
#     for (i, link) in enumerate(arm.links)
#         for vertex in vertices(link.surface_geometry.geometry)
#             push!(surface_points, origins[i] * convert(Vector, vertex))
#         end
#         for point in link.skeleton_points
#             push!(skeleton_points, origins[i] * convert(Vector, point))
#         end
#     end
#     points = vcat(surface_points, skeleton_points)
#     values = vcat(zeros(length(surface_points)), -1 + zeros(length(skeleton_points)))
#     LCMGLClient("points") do lcmgl
#         color(lcmgl, 0, 0, 0)
#         point_size(lcmgl, 5)
#         begin_mode(lcmgl, LCMGL.POINTS)
#         for (i, point) in enumerate(points)
#             if values[i] == 0
#                 color(lcmgl, 0, 0, 0)
#             else
#                 color(lcmgl, 0, 0, 1)
#             end
#             vertex(lcmgl, map(value, convert(Vector, point))...)
#             # sphere(lcmgl, convert(Vector, point), 0.02, 10, 10)
#         end
#         end_mode(lcmgl)
#         switch_buffer(lcmgl)
#     end
#     skin = InterpolatingSurface(points, values, SpatialFields.XSquaredLogX())
# end


end
