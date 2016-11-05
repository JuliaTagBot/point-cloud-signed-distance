module Gjk2

import GeometryTypes
const gt = GeometryTypes
import CoordinateTransformations: Transformation, transform_deriv
import StaticArrays: SVector
import Base: dot, zero, +

abstract Annotated{P}

include("tags.jl")
include("acceleratedmesh.jl")

immutable Difference{PA, PB}
    a::PA
    b::PB
end

dot(n::Number, d::Difference) = Difference(dot(n, d.a), dot(n, d.b))
dot(n::Number, t::Tagged) = n * value(t)
+(d1::Difference, d2::Difference) = Difference(d1.a + d2.a, d1.b + d2.b)


zero{PA, PB}(d::Difference{PA, PB}) = Difference(zero(d.a), zero(d.b))
# zero{PA, PB}(::Type{Difference{PA, PB}}) = Difference(zero(PA), zero(PB))

type CollisionCache{GeomA, GeomB, D1 <: Difference, D2 <: Difference}
    bodyA::GeomA
    bodyB::GeomB
    simplex_points::Vector{D1}
    closest_point::D2
end

function CollisionCache(geomA, geomB)
    simplex_points = [Difference(any_inside(geomA), any_inside(geomB))]
    closest_point = Difference(value(simplex_points[1].a), value(simplex_points[1].b))
    CollisionCache(geomA, geomB, simplex_points, closest_point)
end

function argminmax(f::Function, iter)
    state = start(iter)
    min_arg, state = next(iter, state)
    max_arg = min_arg
    min_val = f(min_arg)
    max_val = min_val
    while !done(iter, state)
        arg, state = next(iter, state)
        val = f(arg)
        if val > max_val
            max_arg = arg
            max_val = val
        elseif val < min_val
            min_arg = arg
            min_val = val
        end
    end
    min_arg, max_arg
end

function support_vector_max(geometry, direction, initial_guess::Tagged)
    best_pt, score = gt.support_vector_max(geometry, direction)
    Tagged(best_pt)
end

function support_vector_max{N, T}(mesh::AcceleratedMesh{N, T}, direction,
                                  initial_guess::Tagged)
    verts = gt.vertices(mesh.mesh)
    best = Tagged(convert(gt.Vec{N, T}, verts[initial_guess.tag]), initial_guess.tag)
    score = dot(direction, best.point)
    while true
        candidates = mesh.neighbors[best.tag]
        neighbor_index, best_neighbor_score = gt.argmax(n -> dot(direction, verts[n]), candidates)
        if best_neighbor_score > score || (best_neighbor_score == score && neighbor_index > best.tag)
            score = best_neighbor_score
            best = Tagged(convert(gt.Vec{N, T}, verts[neighbor_index]), neighbor_index)
        else
            break
        end
    end
    best
end

function support_vector_max{N, T}(mesh::gt.HomogenousMesh{gt.Point{N, T}}, direction, initial_guess::Tagged)
    best_arg, best_value = gt.argmax(x-> x⋅direction, gt.vertices(mesh))
    best_vec = convert(gt.Vec{N, T}, best_arg)
    Tagged(best_vec)
end

@inline function projection_weights(p::gt.Vec, q::gt.Vec)
    [1.0]
end
@inline function projection_weights{T}(pt::T, s::gt.Simplex{1})
    [1.0]
end

function projection_weights{T}(pt::T, s::gt.Simplex, best_sqd=eltype(T)(Inf))
    w = convert(Vector, gt.weights(pt, s))

    @inbounds for i in 1:length(w)
        if w[i] < 0
            w_face = projection_weights(pt, gt.simplex_face(s, i))
            w[i] = 0
            w[1:(i-1)] = w_face[1:(i-1)]
            w[(i+1):end] = w_face[i:end]
            return w
        end
    end
    w
end

function projection_weights(pt, fs::gt.FlexibleSimplex)
    gt.with_immutable(fs) do s
        projection_weights(pt, s)
    end
end

function gjk!(cache::CollisionCache, poseA::Transformation, poseB::Transformation)
    const max_iter = 100
    const atol = 1e-6
    const N = 3
    const origin = zero(gt.Vec{3, Float64})
    const rotAinv = transform_deriv(inv(poseA), origin)
    const rotBinv = transform_deriv(inv(poseB), origin)
    in_interior::Bool = false
    best_point = poseA(cache.closest_point.a) - poseB(cache.closest_point.b)
    simplex = gt.FlexibleSimplex([poseA(value(p.a)) - poseB(value(p.b)) for p in cache.simplex_points])
    weights = projection_weights(origin, simplex)

    for k in 1:max_iter
        # @show simplex weights
        direction = -best_point
        direction_in_A = rotAinv * direction
        direction_in_B = rotBinv * direction

        best_vertex_index, worst_vertex_index =
        argminmax(1:length(cache.simplex_points)) do i
            d = cache.simplex_points[i]
            dot(value(d.a), direction_in_A) + dot(value(d.b), -direction_in_B)
        end
        starting_vertex = cache.simplex_points[best_vertex_index]

        improved_vertex = Difference(
            support_vector_max(cache.bodyA, direction_in_A, starting_vertex.a),
            support_vector_max(cache.bodyB, -direction_in_B, starting_vertex.b))
        improved_point = poseA(value(improved_vertex.a)) - poseB(value(improved_vertex.b))
        # @show direction best_point improved_point
        score = dot(improved_point, direction)
        if score <= dot(best_point, direction) + atol
            break
        else
            if length(cache.simplex_points) <= N
                push!(cache.simplex_points, improved_vertex)
                push!(simplex, improved_point)
            else
                cache.simplex_points[worst_vertex_index] = improved_vertex
                simplex._[worst_vertex_index] = improved_point
            end
            weights = projection_weights(origin, simplex)
            length(weights) > N && all(weights .>= 0) && break
            for i in length(weights):-1:1
                if weights[i] <= atol
                    deleteat!(weights, i)
                    deleteat!(simplex, i)
                    deleteat!(cache.simplex_points, i)
                end
            end
            best_point = sum(broadcast(*, weights, simplex._))
        end
    end
    cache.closest_point = dot(weights, cache.simplex_points)
    cache.closest_point, norm(best_point)
end

end
