module DepthSensors

using CoordinateTransformations
using LCMGL
import StaticArrays: SVector
import Flash

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

function draw_rays(lcmgl::LCMGLClient, sensor::DepthSensor, tform::Transformation)
    camera_origin = tform(SVector(0., 0, 0))
    color(lcmgl, 0, 1, 0)
    begin_mode(lcmgl, LCMGL.LINES)
    for ray in rays_in_world(sensor, tform)
        vertex(lcmgl, camera_origin...)
        vertex(lcmgl, (camera_origin + ray)...)
    end
    end_mode(lcmgl)
end

function draw_rays(sensor::DepthSensor, tform::Transformation)
    LCMGLClient("sensor_rays") do lcmgl
        draw_rays(lcmgl, sensor, tform)
        switch_buffer(lcmgl)
    end
end

Kinect(rows, cols, vertical_fov=0.4682, horizontal_fov=0.5449) = DepthSensor(generateKinectRays(rows, cols, vertical_fov, horizontal_fov))

function doRaycast(camera_origin, camera_view_ray, field::Function)
    EPS = 1E-5
    SAFE_RATE = 0.5
    SAFE_ITER_LIMIT = 60
    dist = 0
    k = 0
    estimated_gradient = -1
    sample_point = camera_origin + dist*camera_view_ray
    last_value = field(sample_point)
    while (abs(last_value) > EPS && k < SAFE_ITER_LIMIT)
        step = -last_value / estimated_gradient
        step = sign(step) * min(SAFE_RATE, abs(step))
        dist += step
        sample_point = camera_origin + dist * camera_view_ray
        value = field(sample_point)
        estimated_gradient = (value - last_value) / step
        # @show dist last_value value estimated_gradient step
        last_value = value
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

function draw_points(lcmgl::LCMGLClient, points::AbstractArray)
    LCMGL.color(lcmgl, 0, 1, 0)
    point_size(lcmgl, 5)
    begin_mode(lcmgl, LCMGL.POINTS)
    for point in points
        vertex(lcmgl, convert(Vector, point)...)
    end
    end_mode(lcmgl)
end

function draw_points(points::AbstractArray)
    LCMGLClient("raycast") do lcmgl
        draw_points(lcmgl, points)
        switch_buffer(lcmgl)
    end
end

end
