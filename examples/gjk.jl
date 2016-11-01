module Gjk

import Base: @pure
import GeometryTypes
import StaticArrays: SVector, @SVector, MVector, @MVector
const gt = GeometryTypes

immutable Tagged{P}
    point::P
    tag::Int
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
any_inside(geometry) = Tagged(gt.any_inside(geometry), 0)
any_inside{N, T}(mesh::gt.AbstractMesh{gt.Point{N, T}}) = Tagged(convert(gt.Vec{N, T}, gt.vertices(mesh)[1]), 0)

function support_vector_max(geometry, direction, initial_guess::Tagged)
    best_pt, score = gt.support_vector_max(geometry, direction)
    Tagged(best_pt, 0), score
end

function support_vector_max{N, T}(mesh::AcceleratedMesh{N, T}, direction,
                                  initial_guess::Tagged{gt.Vec{N, T}})
    best = initial_guess
    score = dot(direction, best.point)
    verts = gt.vertices(mesh.mesh)
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
    Tagged(convert(gt.Vec{N, T}, best_arg), 0), best_value
end

@pure dimensionof{N, T}(::gt.AbstractGeometry{N, T}) = Val{N}()
@pure scalartype{N, T}(::gt.AbstractGeometry{N, T}) = T

@pure dimensionof{N, T, S}(::gt.AbstractSimplex{S, gt.Vec{N, T}}) = Val{N}()
@pure scalartype{N, T, S}(::gt.AbstractSimplex{S, gt.Vec{N, T}}) = T

@pure dimensionof{N, T}(::gt.HomogenousMesh{gt.Point{N, T}}) = Val{N}()
@pure scalartype{N, T}(::gt.HomogenousMesh{gt.Point{N, T}}) = T

@inline dimensionof(mesh::AcceleratedMesh) = dimensionof(mesh.mesh)
@inline scalartype(mesh::AcceleratedMesh) = scalartype(mesh.mesh)

@pure @generated simplexlength{N}(::Val{N}) = quote
    Val{$(N + 1)}()
end

typealias Simplex{M, N, T} MVector{M, Tagged{gt.Vec{N, T}}}

function gjk(hull, max_iter=100, atol=1e-6)
    gjk(hull, dimensionof(hull), scalartype(hull), simplexlength(dimensionof(hull)), max_iter, atol)
end

function gjk{N, T, SimplexLength}(hull, ::Val{N}, ::Type{T}, ::Val{SimplexLength}, max_iter, atol)
    tagged = any_inside(hull)
    simplex = Simplex{SimplexLength, N, T}(tagged for i in 1:SimplexLength)
    gjk(hull, simplex, tagged.point, max_iter, atol)
end

function gjk{N, T, S}(hull, simplex::Simplex{S, N, T}, pt_best, max_iter=100, atol=1e-6)
    zero_point = zero(gt.Vec{N, T})
    simplex_points = 1
    for k in 1:max_iter
        direction = -pt_best
        # Do a linear search over just the simplex to find a good starting point,
        # then do a neighbor-wise search over the mesh to try to find an even
        # better point.
        starting_vertex, _ = gt.argmax(t -> dot(t.point, direction), simplex)
#         @code_warntype support_vector_max(hull, direction, starting_vertex)
        improved_vertex::Tagged{gt.Vec{N, T}}, score::T = support_vector_max(hull, direction, starting_vertex)
        # If we found something no better than the existing simplex, then return
        if score <= dot(pt_best, direction) + atol
            break
        else
            if simplex_points <= N
                simplex[simplex_points + 1] = improved_vertex
                simplex_points += 1
            else
                worst_index, _ = gt.argmax(i -> -dot(simplex[i].point, direction), 1:simplex_points)
                simplex[worst_index] = improved_vertex
            end
            pt_best, sqd::T = gt.proj_sqdist(zero_point, gt.Simplex((convert(gt.Vec{N, T}, t.point) for t in view(simplex, 1:simplex_points))...))
            sqd == 0 && return simplex
        end

    end
    simplex, pt_best
end

# function neighbors{T}(mesh::gt.HomogenousMesh{gt.Point{3, T}}, vertex_index::Integer)
#     neighboring_faces = filter(face -> any(gt.onebased(face, i) == vertex_index for i in 1:length(face)), gt.faces(mesh))
#     Tagged{gt.Point{3, T}}[Tagged(gt.vertices(mesh)[face][i], gt.onebased(face, i)) for face in neighboring_faces for i in 1:length(face) if gt.onebased(face, i) != vertex_index]
# end

end
