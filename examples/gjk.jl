module Gjk

import Base: @pure
import GeometryTypes
import StaticArrays: SVector, @SVector, MVector, @MVector
const gt = GeometryTypes

immutable Tagged{P, T}
    point::P
    tag::T
end

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
    Tagged(point, point)
end

function any_inside{N, T}(mesh::gt.AbstractMesh{gt.Point{N, T}})
    point = convert(gt.Vec{N, T}, gt.vertices(mesh)[1])
    Tagged(point, point)
end

function any_inside(m::gt.MinkowskiDifference)
    t1 = any_inside(m.c1)
    t2 = any_inside(m.c2)
    Tagged(t1.point - t2.point, (t1.tag, t2.tag))
end

function support_vector_max(geometry, direction, initial_guess)
    best_pt, score = gt.support_vector_max(geometry, direction)
    Tagged(best_pt, best_pt), score
end

function support_vector_max{N, T}(mesh::AcceleratedMesh{N, T}, direction,
                                  initial_guess::Integer)
    verts = gt.vertices(mesh.mesh)
    best = Tagged(convert(gt.Vec{N, T}, verts[initial_guess]), initial_guess)
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

function support_vector_max{N, T}(mesh::gt.HomogenousMesh{gt.Point{N, T}}, direction, initial_guess)
    best_arg, best_value = gt.argmax(x-> x⋅direction, gt.vertices(mesh))
    best_vec = convert(gt.Vec{N, T}, best_arg)
    Tagged(best_vec, best_vec), best_value::T
end

function support_vector_max(m::gt.MinkowskiDifference, direction, initial_guess)
    t1, score1 = support_vector_max(m.c1, direction, initial_guess[1])
    t2, score2 = support_vector_max(m.c2, -direction, initial_guess[2])
    Tagged(t1.point - t2.point, (t1.tag, t2.tag)), score1 + score2
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

# typealias Simplex{M, N, T, Tag} MVector{M, Tagged{gt.Vec{N, T}, Tag}}

function gjk(hull, max_iter=100, atol=1e-6)
    gjk(hull, dimensionof(hull), scalartype(hull), max_iter, atol)
end

# function Simplex{M, N, T, TagType}(tagged::Tagged{gt.Vec{N, T}, TagType}, simplexlen::Type{Val{M}})
#     Simplex{M, N, T, TagType}(tagged for i in 1:M)
# end

function gjk{N, T}(hull, ::Type{Val{N}}, ::Type{T}, max_iter, atol)
    tagged = any_inside(hull)
    simplex = gt.FlexibleSimplex([tagged])
    # simplex = Simplex(tagged, simplexlength(dimensionof(hull)))
    # @code_warntype gjk(hull, simplex, tagged.point, max_iter, atol)
    gjk(hull, simplex, tagged.point, max_iter, atol)
end

function proj_sqdist{N, T, S, Tag}(v::gt.Vec{N, T}, simplex::gt.Simplex{S, Gjk.Tagged{gt.Vec{N, T}, Tag}})
    untagged_simplex = gt.Simplex{S, gt.Vec{N, T}}(ntuple(i -> simplex._[i].point, S))
    gt.proj_sqdist(v, untagged_simplex)
end

function proj_sqdist(pt, fs::gt.FlexibleSimplex)
    gt.with_immutable(fs) do s
        proj_sqdist(pt, s)
    end
end

function gjk{N, T, Tag}(hull, simplex::gt.FlexibleSimplex{Tagged{gt.Vec{N, T}, Tag}}, pt_best, max_iter=100, atol=1e-6)
    zero_point = zero(gt.Vec{N, T})
    for k in 1:max_iter
        direction = -pt_best
        # Do a linear search over just the simplex to find a good starting point,
        # then do a neighbor-wise search over the mesh to try to find an even
        # better point.
        starting_vertex, _ = gt.argmax(t -> dot(t.point, direction), simplex._)
        improved_vertex, score = support_vector_max(hull, direction, starting_vertex.tag)
        # If we found something no better than the existing simplex, then return
        if score <= dot(pt_best, direction) + atol
            break
        else
            @show length(simplex) N
            if length(simplex) <= N
                push!(simplex, improved_vertex)
            else
                error("here")
                worst_index = indmin(p -> dot(p.point, direction), simplex)
                simplex._[worst_index] = improved_vertex
            end
            pt_best, sqd = proj_sqdist(zero_point, simplex)
            sqd == 0 && break
        end
    end
    simplex, pt_best
end

end
