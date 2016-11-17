module Gjk2

import GeometryTypes
const gt = GeometryTypes
import CoordinateTransformations: Transformation, transform_deriv
import StaticArrays: SVector, MVector, SMatrix, insert
import Base: dot, zero, +, *, @pure, convert

@generated function convert{SV <: SVector, N, T}(::Type{SV}, v::gt.Vec{N, T})
    Expr(:call, :(SVector{$N, $T}),
        Expr(:tuple, [:(v[$i]) for i in 1:N]...))
end

@generated function convert{SV <: SVector, N, T}(::Type{SV}, v::gt.Point{N, T})
    Expr(:call, :(SVector{$N, $T}),
        Expr(:tuple, [:(v[$i]) for i in 1:N]...))
end

abstract Annotated{P}

include("tags.jl")
include("acceleratedmesh.jl")

immutable Difference{PA, PB}
    a::PA
    b::PB
end

dot(n::Number, d::Difference) = Difference(dot(n, d.a), dot(n, d.b))
*(n::Number, d::Difference) = Difference(*(n, d.a), *(n, d.b))
dot(n::Number, t::Tagged) = n * value(t)
+(d1::Difference, d2::Difference) = Difference(d1.a + d2.a, d1.b + d2.b)


zero{PA, PB}(d::Difference{PA, PB}) = Difference(zero(d.a), zero(d.b))
# zero{PA, PB}(::Type{Difference{PA, PB}}) = Difference(zero(PA), zero(PB))

@pure dimension(t::Any) = dimension(typeof(t))
@pure scalartype(t::Any) = scalartype(typeof(t))

@pure dimension{N, T}(::Type{gt.AbstractGeometry{N, T}}) = Val{N}
@pure scalartype{N, T}(::Type{gt.AbstractGeometry{N, T}}) = T

@pure dimension{N, T, S}(::Type{gt.AbstractSimplex{S, gt.Vec{N, T}}}) = Val{N}
@pure scalartype{N, T, S}(::Type{gt.AbstractSimplex{S, gt.Vec{N, T}}}) = T

@pure dimension{N, T}(::Type{gt.HomogenousMesh{gt.Point{N, T}}}) = Val{N}
@pure scalartype{N, T}(::Type{gt.HomogenousMesh{gt.Point{N, T}}}) = T

@pure dimension{N, T}(::Type{gt.Vec{N, T}}) = Val{N}
@pure scalartype{N, T}(::Type{gt.Vec{N, T}}) = T

@pure dimension{N, T, M}(::Type{AcceleratedMesh{N, T, M}}) = Val{N}
@pure scalartype{N, T, M}(::Type{AcceleratedMesh{N, T, M}}) = T

@pure dimension{M, N, T}(::Type{gt.Simplex{M, gt.Vec{N, T}}}) = Val{N}
@pure scalartype{M, N, T}(::Type{gt.Simplex{M, gt.Vec{N, T}}}) = T

type CollisionCache{GeomA, GeomB, M, D1 <: Difference, D2 <: Difference}
    bodyA::GeomA
    bodyB::GeomB
    simplex_points::MVector{M, D1}
    closest_point::D2
end

function CollisionCache(geomA, geomB)
    N = dimension(geomA)
    @assert dimension(geomA) == dimension(geomB)
    CollisionCache(N, geomA, geomB)
end

function edgespan(points::AbstractVector)
    span = [p - points[1] for p in points[2:end]]
end

function CollisionCache{N}(::Type{Val{N}}, geomA, geomB)
    simplex_points = [Difference(any_inside(geomA), any_inside(geomB))]
    closest_point = Difference(value(simplex_points[1].a), value(simplex_points[1].b))

    # Search for a starting simplex in the Minkowski difference by sampling
    # random directions until we find a set of points with linearly independent
    # edgespan.
    max_iter = 100
    for i in 1:max_iter
        direction = 2 * (rand(N) .- 0.5)
        candidate = Difference(support_vector_max(geomA, direction, simplex_points[1].a),
                               support_vector_max(geomB, -direction, simplex_points[1].b))
        mat = hcat(edgespan(map(p -> value(p.a) - value(p.b), [simplex_points..., candidate]))...)
        if det(mat' * mat) > 1e-3
            push!(simplex_points, candidate)
            if length(simplex_points) > N
                simplex = MVector{N+1}(simplex_points)
                return CollisionCache(geomA, geomB, simplex, closest_point)
            end
        end
    end

    error("Could not find a sensible initial simplex. Both geometries might have zero volume.")
end

dimension{G1, G2, M, D1, D2}(::Type{CollisionCache{G1, G2, M, D1, D2}}) = dimension(G1)

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
    Tagged(convert(SVector, best_pt))
end

function support_vector_max{N, T}(pt::gt.Vec{N, T}, direction, initial_guess::Tagged)
    Tagged(convert(SVector, pt))
end

function support_vector_max{N, T}(mesh::AcceleratedMesh{N, T}, direction,
                                  initial_guess::Tagged)
    verts = gt.vertices(mesh.mesh)
    best = Tagged(convert(SVector, verts[initial_guess.tag]), initial_guess.tag)
    score = dot(direction, best.point)
    while true
        candidates = mesh.neighbors[best.tag]
        neighbor_index, best_neighbor_score = gt.argmax(n -> dot(direction, convert(SVector, verts[n])), candidates)
        if best_neighbor_score > score || (best_neighbor_score == score && neighbor_index > best.tag)
            score = best_neighbor_score
            best = Tagged(convert(SVector, verts[neighbor_index]), neighbor_index)
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

dimension{M, N, T}(::Simplex{M, N, T}) = Val{N}
scalartype{M, N, T}(::Simplex{M, N, T}) = T

dimension{N, T}(::SVector{N, T}) = Val{N}
scalartype{N, T}(::SVector{N, T}) = T

any_inside{M, N, T}(simplex::Simplex{M, N, T}) = Tagged(simplex[1])
function support_vector_max{M, N, T}(simplex::Simplex{M, N, T}, direction, initial_guess::Tagged)
    best_pt, score = gt.argmax(p -> dot(p, direction), simplex)
    Tagged(best_pt)
end

any_inside(pt::SVector) = Tagged(pt)
support_vector_max(pt::SVector, direction, initial_guess::Tagged) = Tagged(pt)


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

@generated function simplex_face{N, M, T}(simplex::Simplex{M, N, T}, i::Integer)
    simplex_face_impl(simplex, i)
end

function simplex_face_impl{N, M, T}(simplex::SimplexType{M, N, T}, i)
    Expr(:call, :(SVector{$(M - 1), SVector{$N, $T}}),
        Expr(:tuple, [:(i > $j ? simplex[$j] : simplex[$(j+1)]) for j in 1:(M - 1)]...))
end

include("johnson_distance.jl")

# function projection_weights{N, M, T1, T2}(pt::SVector{N, T1}, simplex::Simplex{M, N, T2})
#     w = weights(pt, simplex)
#
#     for i in 1:length(w)
#         if w[i] < 0
#             face = simplex_face(simplex, i)
#             @show i face weights(pt, face)
#             w_face, _ = projection_weights(pt, face)
#         end
#     end
#
#     (w, 0)
#     #
#     #
#     # w_min, i = findmin(w)
#     # if w_min < 0
#     #     face = simplex_face(simplex, i)
#     #     println((i, face))
#     #     w_face, _ = projection_weights(pt, face)
#     #     return (insert(w_face, i, 0), i)
#     # else
#     #     return (w, 0)
#     # end
# end

# function projection_weights_impl{N, M, T1, T2}(pt::Type{SVector{N, T1}}, simplex::SimplexType{M, N, T2})
#     T = promote_type(T1, T2)
#     expr = quote
#     end
#     for i in 1:M
#         push!(expr.args, quote
#             if w[$i] < 0
#                 face = simplex_face(simplex, $i)
#                 println(($i, face))
#                 w_face, _ = projection_weights(pt, face)
#                 return (insert())
#                 return ($(Expr(:call, :(SVector{$M, $T}), [:(w_face[$j]) for j in 1:(i-1)]..., :(zero($T)), [:(w_face[$(j-1)]) for j in (i+1):M]...)), $i)
#             end
#         end)
#     end
#     push!(expr.args, quote
#         return (w, 0)
#     end)
#     expr
# end

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

@generated function transform_simplex(cache::CollisionCache, poseA, poseB)
    transform_simplex_impl(cache, poseA, poseB)
end

@pure value{N}(::Type{Val{N}}) = N

function transform_simplex_impl(cache, poseA, poseB)
    Expr(:call, :(MVector),
        [:(poseA(value(cache.simplex_points[$i].a)) -
           poseB(value(cache.simplex_points[$i].b))) for i in 1:value(dimension(cache))+1]...)
end

function gjk!{N}(::Type{Val{N}}, cache::CollisionCache, poseA::Transformation, poseB::Transformation)
    const max_iter = 100
    const atol = 1e-6
    const origin = zeros(SVector{N, Float64})
    const rotAinv = transform_deriv(inv(poseA), origin)
    const rotBinv = transform_deriv(inv(poseB), origin)
    simplex = transform_simplex(cache, poseA, poseB)
    in_interior = false
    best_point = simplex[1]

    for k in 1:max_iter
        weights, index_to_replace = projection_weights(origin, simplex)
        @show simplex weights
        in_interior = index_to_replace == 0
        # index_to_replace = 0
        # for i in 1:length(weights)
        #     if weights[i] <= atol
        #         index_to_replace = i
        #         in_interior = false
        #         break
        #     end
        # end
        if in_interior
            break
        end
        #
        # for i in length(weights):-1:1
        #     if weights[i] <= atol
        #         deleteat!(weights, i)
        #         deleteat!(simplex, i)
        #         deleteat!(cache.simplex_points, i)
        #     end
        # end
        # if in_interior
        #     return simplex, norm(best_point), in_interior
        # end
        best_point = dot(weights, simplex)
        @show best_point
        # best_point = sum(broadcast(*, weights, simplex._))
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
        @show dot(best_point, direction)
        @show improved_point score
        if score <= dot(best_point, direction) + atol
            break
        else
            cache.simplex_points[index_to_replace] = improved_vertex
            simplex[index_to_replace] = improved_point
            # if length(cache.simplex_points) <= N
            #     push!(cache.simplex_points, improved_vertex)
            #     push!(simplex, improved_point)
            # else
            #     cache.simplex_points[worst_vertex_index] = improved_vertex
            #     simplex._[worst_vertex_index] = improved_point
            # end
        end
    end
    return simplex, best_point, in_interior
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
