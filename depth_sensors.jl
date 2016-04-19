module DepthSensors

using GeometryTypes
using AffineTransforms
using SpatialFields

export Kinect, raycast_depths, raycast_points

function generateKinectRays(rows, cols, vertical_fov=0.4682, horizontal_fov=0.5449)
    camera_cx = (cols + 1) / 2.
    camera_cy = (rows + 1) / 2.
    tan_vert_fov = tan(vertical_fov)
    tan_hor_fov = tan(horizontal_fov)

    rays = Array{Point{3, Float64}}(rows, cols)

    for v in 1:rows
        for u in 1:cols
            rays[v, u] = Point{3, Float64}(
            (u - camera_cx) * tan_vert_fov / camera_cx,
            (v - camera_cy) * tan_hor_fov / camera_cy,
                    1.0
                )
            rays[v, u] /= norm(rays[v, u])
        end
    end
    return rays
end

type DepthSensor
    rays::Array{Point{3, Float64}, 2}

end

Kinect(rows, cols, vertical_fov=0.4682, horizontal_fov=0.5449) = DepthSensor(generateKinectRays(rows, cols, vertical_fov, horizontal_fov))

function doRaycast(camera_origin, camera_view_ray, field::ScalarField)
    EPS = 1E-5
    SAFE_RATE = 0.1
    SAFE_ITER_LIMIT = 15
    dist = 0
    k = 0
    while (abs(evaluate(field, camera_origin + dist*camera_view_ray)) > EPS && k < SAFE_ITER_LIMIT)
        dist = dist + SAFE_RATE*evaluate(field, camera_origin + dist*camera_view_ray)
        k += 1
    end
    if abs(evaluate(field, camera_origin + dist*camera_view_ray)) > 1000*EPS
        return NaN
    else
        return dist
    end
end

function raycast_depths{N, T}(surface::ScalarField{N, T}, sensor::DepthSensor, sensor_origin::AffineTransform)
    distances = similar(sensor.rays, T)
    for row in 1:size(sensor.rays, 1)
        for col in 1:size(sensor.rays, 2)
            camera_view_ray = sensor_origin.scalefwd * convert(Vector, sensor.rays[row, col])
            camera_view_ray /= norm(camera_view_ray)
            distances[row, col] = doRaycast(sensor_origin.offset, camera_view_ray, surface)
        end
    end
    distances
end

function raycast_points{N, T}(surface::ScalarField{N, T}, sensor::DepthSensor, sensor_origin::AffineTransform)
    distances = raycast_depths(surface, sensor, sensor_origin)
    points = Point{N, T}[]
    for row in 1:size(distances, 1)
        for col in 1:size(distances, 2)
            if !isnan(distances[row, col])
                push!(points, sensor_origin * convert(Vector, distances[row, col] * sensor.rays[row, col] / norm(sensor.rays[row, col])))
            end
        end
    end
    points
end

end
