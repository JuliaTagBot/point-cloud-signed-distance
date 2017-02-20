module DepthSensors

using CoordinateTransformations
using StaticArrays: SVector
import Flash
using DrakeVisualizer: Visualizer, PointCloud, batch, setgeometry!
using ColorTypes: RGB
import ForwardDiff

export Kinect, raycast_depths, raycast_points

function generateKinectRays(rows, cols, vertical_fov=0.4682, horizontal_fov=0.5449)
    camera_cx = (cols + 1) / 2.
    camera_cy = (rows + 1) / 2.
    tan_vert_fov = tan(vertical_fov)
    tan_hor_fov = tan(horizontal_fov)

    rays = Array{SVector{3, Float64}}(rows, cols)

    for v in 1:rows
        for u in 1:cols
            ray = SVector{3, Float64}(
            (u - camera_cx) * tan_vert_fov / camera_cx,
            (v - camera_cy) * tan_hor_fov / camera_cy,
                    1.0
                )
            ray = normalize(ray)
            rays[v, u] = ray
        end
    end
    return rays
end

type DepthSensor
    rays::Array{SVector{3, Float64}, 2}
end

function draw_rays(vis::Visualizer, sensor::DepthSensor, tform::Transformation)
    camera_origin = tform(SVector(0., 0, 0))
    batch(vis) do v
        cloud = PointCloud(SVector{3, Float64}[])
        cloud.channels[:rgb] = RGB{Float64}[]
        push!(cloud.points, camera_origin)
        push!(cloud.channels[:rgb], RGB(1, 0, 0))
        for ray in rays_in_world(sensor, tform)
            push!(cloud.points, camera_origin + ray)
            push!(cloud.channels[:rgb], RGB(0, 1, 0))
        end
        setgeometry!(v, cloud)
    end
end

Kinect(rows, cols, vertical_fov=0.4682, horizontal_fov=0.5449) = DepthSensor(generateKinectRays(rows, cols, vertical_fov, horizontal_fov))

function doRaycast(camera_origin, camera_view_ray, field::Function)
    EPS = 1E-5
    SAFE_RATE = 0.4
    SAFE_ITER_LIMIT = 60
    dist = 0
    k = 0
    field_along_ray = x -> field(camera_origin + x * camera_view_ray)
    value = field_along_ray(dist)
    while (abs(value) > EPS && k < SAFE_ITER_LIMIT)
        derivative = ForwardDiff.derivative(field_along_ray, dist)
        step = value / -derivative
        step = sign(step) * min(SAFE_RATE, abs(step))
        dist += step
        value = field_along_ray(dist)
        # sample_point = camera_origin + dist * camera_view_ray
        # value = field(sample_point)
        # estimated_gradient = (value - last_value) / step
        # @show dist last_value value estimated_gradient step
        # last_value = value
        k += 1
    end
    if abs(field(camera_origin + dist*camera_view_ray)) > 1000*EPS
        return NaN
    else
        return dist
    end
end

function rays_in_world(sensor::DepthSensor, sensor_tform::Union{AbstractAffineMap, IdentityTransformation})
    rotation = LinearMap(transform_deriv(sensor_tform, SVector{3, Float64}(0,0,0)))
    SVector{3, Float64}[rotation(ray) for ray in sensor.rays]
end

function raycast_depths(surface::Function, sensor::DepthSensor, sensor_tform::Union{AbstractAffineMap, IdentityTransformation})
    sensor_origin_xyz = sensor_tform(SVector{3, Float64}(0, 0, 0))
    rays = rays_in_world(sensor, sensor_tform)
    distances = similar(rays, Float64)
    for i in eachindex(rays)
        camera_view_ray = normalize(rays[i])
        distances[i] = doRaycast(sensor_origin_xyz, camera_view_ray, surface)
    end
    distances
end

function raycast_points(surface::Function, sensor::DepthSensor, sensor_tform::Union{AbstractAffineMap, IdentityTransformation})
    distances = raycast_depths(surface, sensor, sensor_tform)
    points = SVector{3, Float64}[]
    for row in 1:size(distances, 1)
        for col in 1:size(distances, 2)
            if !isnan(distances[row, col])
                # @show distances[row, col]
                # @show distances[row, col] * normalize(sensor.rays[row, col])
                # @show sensor_tform(distances[row, col] * normalize(sensor.rays[row, col]))
                push!(points, sensor_tform(distances[row, col] * normalize(sensor.rays[row, col])))
            end
        end
    end
    points
end

function raycast(state::Flash.ManipulatorState, sensor::DepthSensor, sensor_tform::Union{AbstractAffineMap, IdentityTransformation})
    surface = Flash.skin(state)
    raycast_points(surface, sensor, sensor_tform)
end

end
