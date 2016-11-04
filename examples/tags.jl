immutable Tagged{P, T} <: Annotated{P}
    point::P
    tag::T
end

Tagged{P}(point::P) = Tagged(point, nothing)

value(t::Tagged) = t.point

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
