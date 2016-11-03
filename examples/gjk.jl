module Gjk

import Base: @pure, -, .*, +, dot
import GeometryTypes
import StaticArrays: SVector, @SVector, MVector, @MVector
const gt = GeometryTypes

abstract Annotated{P}

immutable Tagged{P, T} <: Annotated{P}
    point::P
    tag::T
end

Tagged{P}(point::P) = Tagged(point, nothing)

value(t::Tagged) = t.point

immutable Difference{P, P1, P2} <: Annotated{P}
    point::P
    p1::P1
    p2::P2
end
Difference(p1, p2) = Difference(p1 - p2, p1, p2)
# TaggedDifference{P, P1, P2}(p1::Tagged{P, Tag1}, p2::Tagged{P, Tag2}) =
#     TaggedDifference{P, Tag1, Tag2}(p1.point - p2.point, p1, p2)
# TaggedDifference{P}(p1::P, p2::P) = TaggedDifference(Tagged(p1), Tagged(p2))

# -(t1::Tagged, t2::Tagged) = TaggedDifference(t1, t2)
-(t1::Tagged, t2::Tagged) = value(t1) - value(t2)
+(t1::Tagged, t2::Tagged) = value(t1) + value(t2)
-(p::gt.Vec, t::Tagged) = p - value(t)
.*(t::Tagged, n::Number) = .*(value(t), n)
-(t::Tagged) = -value(t)

.*(d::Difference, n::Number) = Difference(.*(d.p1, n), .*(d.p2, n))
+(d1::Difference, d2::Difference) = Difference(d1.p1 + d2.p1, d1.p2 + d2.p2)
-(d::Difference) = Difference(-d.point, -d.p1, -d.p2)
dot(v, d::Difference) = dot(v, value(d))
dot(d::Difference, v) = dot(value(d), v)
dot(d1::Difference, d2::Difference) = dot(value(d1), value(d2))

Base.Tuple(t::Tagged) = Tuple(value(t))
Base.Tuple(d::Difference) = Tuple(value(d))

value(d::Difference) = d.point

type AcceleratedMesh{N, T, MeshType <: gt.AbstractMesh}
    mesh::MeshType
    neighbors::Vector{Set{Int}}
end

function plane_fit(data)
    centroid = mean(data, 2)
    U, s, V = svd(data .- centroid)
    i = indmin(s)
    normal = U[:,i]
    offset = dot(normal, centroid)
    normal, offset
end


function AcceleratedMesh{N, T}(mesh::gt.AbstractMesh{gt.Point{N, T}})
    neighbors = Set{Int}[Set{Int}() for vertex in gt.vertices(mesh)]
    for face in gt.faces(mesh)
        for i in 1:length(face)
            for j in i+1:length(face)
                if face[i] != face[j]
                    push!(neighbors[gt.onebased(face, i)], gt.onebased(face, j))
                    push!(neighbors[gt.onebased(face, j)], gt.onebased(face, i))
                end
            end
        end
    end

    # The enhanced GJK algorithm is susceptible to becoming stuck in local minima
    # if all of the neighbors of a given vertex are coplanar. It also benefits from
    # having some distant neighbors for each node, to avoid having to always take the
    # long way around the mesh to get to the other side.
    # To try to fix this, we will compute a fitting plane for all of the existing
    # neighbors for each vertex. We will then add neighbors corresponding to the
    # vertices at the maximum distance on each side of that plane.
    verts = gt.vertices(mesh)
    for i in eachindex(neighbors)
        @assert length(neighbors[i]) >= 2
        normal, offset = plane_fit(reinterpret(T,
            [verts[n] for n in neighbors[i]], (N, length(neighbors[i]))))
        push!(neighbors[i], indmin(map(v -> dot(convert(gt.Point{N, T}, normal), v), verts)))
        push!(neighbors[i], indmax(map(v -> dot(convert(gt.Point{N, T}, normal), v), verts)))
    end
    AcceleratedMesh{N, T, typeof(mesh)}(mesh, neighbors)
end

any_inside{N, T}(mesh::AcceleratedMesh{N, T}) = Tagged(gt.Vec{N, T}(gt.vertices(mesh.mesh)[1]), 1)
function any_inside(geometry)
    point = gt.any_inside(geometry)
    Tagged(point)
end

function any_inside{N, T}(mesh::gt.AbstractMesh{gt.Point{N, T}})
    point = convert(gt.Vec{N, T}, gt.vertices(mesh)[1])
    Tagged(point)
end

function any_inside(m::gt.MinkowskiDifference)
    t1 = any_inside(m.c1)
    t2 = any_inside(m.c2)
    Difference(t1, t2)
end

function support_vector_max(geometry, direction, initial_guess::Tagged)
    best_pt, score = gt.support_vector_max(geometry, direction)
    Tagged(best_pt), score
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
    best, score
end

function support_vector_max{N, T}(mesh::gt.HomogenousMesh{gt.Point{N, T}}, direction, initial_guess::Tagged)
    best_arg, best_value = gt.argmax(x-> x⋅direction, gt.vertices(mesh))
    best_vec = convert(gt.Vec{N, T}, best_arg)
    Tagged(best_vec), best_value::T
end

function support_vector_max(m::gt.MinkowskiDifference, direction, initial_guess::Annotated)
    t1, score1 = support_vector_max(m.c1, direction, initial_guess.p1)
    t2, score2 = support_vector_max(m.c2, -direction, initial_guess.p2)
    Difference(t1, t2), score1 + score2
end

@pure dimensionof{N, T}(::gt.AbstractGeometry{N, T}) = Val{N}
@pure scalartype{N, T}(::gt.AbstractGeometry{N, T}) = T

@pure dimensionof{N, T, S}(::gt.AbstractSimplex{S, gt.Vec{N, T}}) = Val{N}
@pure scalartype{N, T, S}(::gt.AbstractSimplex{S, gt.Vec{N, T}}) = T

@pure dimensionof{N, T}(::gt.HomogenousMesh{gt.Point{N, T}}) = Val{N}
@pure scalartype{N, T}(::gt.HomogenousMesh{gt.Point{N, T}}) = T

@pure dimensionof{N, T}(::gt.Vec{N, T}) = Val{N}
@pure scalartype{N, T}(::gt.Vec{N, T}) = T

@inline dimensionof(m::gt.MinkowskiDifference) = dimensionof(m.c1)
@inline scalartype(m::gt.MinkowskiDifference) = promote_type(scalartype(m.c1), scalartype(m.c2))

@inline dimensionof(mesh::AcceleratedMesh) = dimensionof(mesh.mesh)
@inline scalartype(mesh::AcceleratedMesh) = scalartype(mesh.mesh)

@generated simplexlength{N}(::Type{Val{N}}) = quote
    Val{$(N + 1)}
end

function gjk(hull, max_iter=100, atol=1e-6)
    gjk(hull, dimensionof(hull), scalartype(hull), max_iter, atol)
end

function gjk{N, T}(hull, ::Type{Val{N}}, ::Type{T}, max_iter, atol)
    tagged = any_inside(hull)
    simplex = gt.FlexibleSimplex([tagged])
    # simplex = Simplex(tagged, simplexlength(dimensionof(hull)))
    # @code_warntype gjk(hull, simplex, tagged.point, max_iter, atol)
    gjk(hull, simplex, value(tagged), max_iter, atol)
end

@inline function proj_sqdist(p::gt.Vec, q::gt.Vec, best_sqd=eltype(p)(Inf))
    q, min(best_sqd, gt.sqnorm(p-q))
end
@inline function proj_sqdist{T}(pt::T, s::gt.Simplex{1}, best_sqd=eltype(T)(Inf))
    proj_sqdist(pt, gt.translation(s), best_sqd)
end
function proj_sqdist(pt, d::Difference, best_sqd)
    p1, d1 = proj_sqdist(pt, d.p1, best_sqd)
    p2, d2 = proj_sqdist(pt, d.p2, best_sqd)
    Difference(p1, p2), d1 - d2
end
proj_sqdist(pt, t::Tagged, best_sqd) = proj_sqdist(pt, value(t), best_sqd)

sqdist(pt, s, best=Inf) = proj_sqdist(pt, s, best)[2]

function proj_sqdist{T}(pt::T, s::gt.Simplex, best_sqd=eltype(T)(Inf))
    w = gt.weights(pt, map(value, s))
    best_proj = sum(s .* w)
    # best_proj = gt.vertexmat(s) * w
    # at this point best_proj lies in the subspace spanned by s,
    # but not necessarily inside s
    sqd = sqdist(pt, best_proj)
    if sqd >= best_sqd  # pt is far away even from the subspace spanned by s
        return best_proj, best_sqd
    elseif any(w .< 0)  # pt is closest to point inside a face of s
        @inbounds for i in 1:length(w)
            if w[i] < 0
                proj, sqd = proj_sqdist(pt, gt.simplex_face(s,i), best_sqd)
                if sqd < best_sqd
                    best_sqd = sqd
                    best_proj = proj
                end
            end
        end
        return best_proj, best_sqd
    else # proj lies in the interiour of s
        return best_proj, sqd
    end
end

function proj_sqdist(pt, fs::gt.FlexibleSimplex)
    gt.with_immutable(fs) do s
        proj_sqdist(pt, s)
    end
end

function gjk{N, T}(hull, simplex::gt.FlexibleSimplex, pt_best::Union{gt.Vec{N, T}, Difference{gt.Vec{N, T}}}, max_iter=100, atol=1e-6)
    zero_point = zero(gt.Vec{N, T})
    for k in 1:max_iter
        direction = -pt_best
        # Do a linear search over just the simplex to find a good starting point,
        # then do a neighbor-wise search over the mesh to try to find an even
        # better point.
        starting_vertex, _ = gt.argmax(t -> dot(value(t), direction), simplex._)
        improved_vertex, score = support_vector_max(hull, direction, starting_vertex)
        # If we found something no better than the existing simplex, then return
        if score <= dot(pt_best, direction) + atol
            break
        else
            if length(simplex) <= N
                push!(simplex, improved_vertex)
            else
                worst_index, _ = gt.argmax(i -> -dot(value(simplex._[i]), direction), 1:length(simplex))
                simplex._[worst_index] = improved_vertex
            end
            pt_best, sqd = proj_sqdist(zero_point, simplex)
            sqd == 0 && break
        end
    end
    simplex, pt_best
end

end
