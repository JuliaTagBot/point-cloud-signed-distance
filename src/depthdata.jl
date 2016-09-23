module DepthData

using LCMGL
import StaticArrays: SVector, @SVector
import ColorTypes: RGB

immutable DepthSensorReturn
    position::SVector{3, Float64}
    color::RGB{Float64}
end

immutable PointCloud
    camera_origin::SVector{3, Float64}
    points::Vector{DepthSensorReturn}
end

Base.show(io::IO, point_cloud::PointCloud) = write(io, "PointCloud with origin: $(point_cloud.camera_origin) containing $(length(point_cloud.points)) points")

function read_point_cloud(file::IOStream)
    origin_line = readline(file)
    components = split(origin_line, ',')
    camera_origin = @SVector [parse(Float64, components[i]) for i = 1:3]
    data = readdlm(file, ',')
    points = Vector{DepthSensorReturn}(size(data, 1))
    for i in 1:size(data, 1)
        points[i] = DepthSensorReturn(SVector(data[i, 1], data[i, 2], data[i, 3]),
            RGB{Float64}(data[i, 4], data[i, 5], data[i, 6]))
    end
    PointCloud(camera_origin, points)
end

function render_lcmgl(point_cloud::PointCloud)
    LCMGLClient("point_cloud") do lcmgl
        render_lcmgl(point_cloud, lcmgl)
    end
end

function render_lcmgl(point_cloud::PointCloud, lcmgl::LCMGLClient)
    for point in point_cloud.points[1:22500]
        LCMGL.color(lcmgl, point.color.r, point.color.g, point.color.b)
        begin_mode(lcmgl, LCMGL.POINTS)
        vertex(lcmgl, point.position...)
        end_mode(lcmgl)
    end
    switch_buffer(lcmgl)
end

end  # module
