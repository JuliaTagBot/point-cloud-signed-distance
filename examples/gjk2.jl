module Gjk2

import GeometryTypes
const gt = GeometryTypes
import CoordinateTransformations: Transformation, transform_deriv
import StaticArrays: SVector, MVector, SMatrix
import Base: dot, zero, +, @pure

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

@pure dimension{N, T}(::gt.AbstractGeometry{N, T}) = Val{N}
@pure scalartype{N, T}(::gt.AbstractGeometry{N, T}) = T

@pure dimension{N, T, S}(::gt.AbstractSimplex{S, gt.Vec{N, T}}) = Val{N}
@pure scalartype{N, T, S}(::gt.AbstractSimplex{S, gt.Vec{N, T}}) = T

@pure dimension{N, T}(::gt.HomogenousMesh{gt.Point{N, T}}) = Val{N}
@pure scalartype{N, T}(::gt.HomogenousMesh{gt.Point{N, T}}) = T

@pure dimension{N, T}(::gt.Vec{N, T}) = Val{N}
@pure scalartype{N, T}(::gt.Vec{N, T}) = T

@inline dimension(m::gt.MinkowskiDifference) = dimension(m.c1)
@inline scalartype(m::gt.MinkowskiDifference) = promote_type(scalartype(m.c1), scalartype(m.c2))

@inline dimension(mesh::AcceleratedMesh) = dimension(mesh.mesh)
@inline scalartype(mesh::AcceleratedMesh) = scalartype(mesh.mesh)


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

function pinvli(mat::AbstractArray)
    @assert size(mat, 1) >= size(mat, 2)
    inv(mat' * mat) * mat'
end

typealias Simplex{M, N, T} Union{MVector{M, SVector{N, T}}, SVector{M, SVector{N, T}}}
typealias SimplexType{M, N, T} Union{Type{MVector{M, SVector{N, T}}}, Type{SVector{M, SVector{N, T}}}}

@generated function edgespan{N, M, T}(simplex::Simplex{M, N, T})
    edgespan_impl(simplex)
end

function edgespan_impl{N, M, T}(simplex::SimplexType{M, N, T})
    Expr(:call, :(SMatrix{$N, $(M - 1)}),
        [:(simplex[$i][$j] - simplex[1][$j]) for i in 2:M for j in 1:N]...)
end


@generated function weights{N, T1, T2}(pt::SVector{N, T1}, simplex::Simplex{1, N, T2})
    T = promote_type(T1, T2)
    :(SVector{1, $T}(one($T)))
end

@generated function weights{N, M, T1, T2}(pt::SVector{N, T1}, simplex::Simplex{M, N, T2})
    weights_impl(pt, simplex)
end

function weights_impl{N, M, T1, T2}(pt::Type{SVector{N, T1}}, simplex::SimplexType{M, N, T2})
    expr = quote
        span = edgespan(simplex)
        w = pinvli(span) * (pt - simplex[1])
    end
    push!(expr.args, Expr(:call, :(SVector{$M, $(promote_type(T1, T2))}), :(1 - sum(w)), [:(w[$i]) for i in 1:(M - 1)]...))
    expr
end

# @inline function projection_weights(p::gt.Vec, q::gt.Vec)
#     [1.0], false
# end
# @inline function projection_weights{T}(pt::T, s::gt.Simplex{1})
#     [1.0], false
# end
#

@generated function simplex_face{N, M, T}(simplex::Simplex{M, N, T}, i::Integer)
    simplex_face_impl(simplex, i)
end

function simplex_face_impl{N, M, T}(simplex::SimplexType{M, N, T}, i)
    Expr(:call, :(SVector{$(M - 1), SVector{$N, $T}}),
        [:(i > $j ? simplex[$j] : simplex[$(j+1)]) for j in 1:(M - 1)]...)
end

@generated function projection_weights{N, M, T1, T2}(pt::SVector{N, T1}, simplex::Simplex{M, N, T2})
    projection_weights_impl(pt, simplex)
end

function projection_weights_impl{N, M, T1, T2}(pt::Type{SVector{N, T1}}, simplex::SimplexType{M, N, T2})
    T = promote_type(T1, T2)
    expr = quote
        w = weights(pt, simplex)
    end
    for i in 1:M
        push!(expr.args, quote
            if w[$i] < 0
                face = simplex_face(simplex, $i)
                w_face = projection_weights(pt, face)
                return $(Expr(:call, :(SVector{$M, $T}), [:(w_face[$j]) for j in 1:(i-1)]..., :(zero($T)), [:(w_face[$(j-1)]) for j in (i+1):M]...))
            end
        end)
    end
    push!(expr.args, quote
        return w
    end)
    expr
end

# function projection_weights{T}(pt::T, s::gt.Simplex, best_sqd=eltype(T)(Inf))
#     w = convert(Vector, gt.weights(pt, s))
#
#     @inbounds for i in 1:length(w)
#         if w[i] < 0
#             w_face, _ = projection_weights(pt, gt.simplex_face(s, i))
#             w[i] = 0
#             w[1:(i-1)] = w_face[1:(i-1)]
#             w[(i+1):end] = w_face[i:end]
#             return w, false
#         end
#     end
#     w, length(s) > length(pt)
# end
#
# function projection_weights(pt, fs::gt.FlexibleSimplex)
#     gt.with_immutable(fs) do s
#         projection_weights(pt, s)
#     end
# end

function gjk!(cache::CollisionCache, poseA::Transformation, poseB::Transformation)
    gjk!(dimension(cache.bodyA), cache, poseA, poseB)
end

function gjk!{N}(::Type{Val{N}}, cache::CollisionCache, poseA::Transformation, poseB::Transformation)
    const max_iter = 100
    const atol = 1e-6
    const origin = zero(gt.Vec{N, Float64})
    const rotAinv = transform_deriv(inv(poseA), origin)
    const rotBinv = transform_deriv(inv(poseB), origin)
    simplex = gt.FlexibleSimplex([poseA(value(p.a)) - poseB(value(p.b)) for p in cache.simplex_points])
    best_point = simplex._[1]
    in_interior::Bool = false

    for k in 1:max_iter
        weights, in_interior = projection_weights(origin, simplex)
        if in_interior
            break
        end
        for i in length(weights):-1:1
            if weights[i] <= atol
                deleteat!(weights, i)
                deleteat!(simplex, i)
                deleteat!(cache.simplex_points, i)
            end
        end
        best_point = sum(broadcast(*, weights, simplex._))
        cache.closest_point = dot(weights, cache.simplex_points)

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
        end
    end
    simplex, norm(best_point), in_interior
end

function signed_distance!(cache::CollisionCache, poseA::Transformation, poseB::Transformation)
    signed_distance!(dimension(cache.bodyA), cache, poseA, poseB)
end

function signed_distance!{N}(dim::Type{Val{N}}, cache::CollisionCache, poseA::Transformation, poseB::Transformation)
    simplex, separation, in_collision = gjk!(dim, cache, poseA, poseB)

    if in_collision
        gt.with_immutable(simplex) do s
            const origin = zero(gt.Vec{N, Float64})
            _, penetration_distance = gt.argmax(1:length(s)) do i
                face = gt.simplex_face(s, i)
                weights, _ = projection_weights(origin, face)
                distance_to_face = norm(sum(face[i] * weights[i] for i in 1:length(face)))
                -distance_to_face
            end
            return penetration_distance
        end
    else
        return separation
    end
end

end
