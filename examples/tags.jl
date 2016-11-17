immutable Tagged{P, T} <: Annotated{P}
    point::P
    tag::T
end

Tagged{P}(point::P) = Tagged(point, nothing)

*(n::Number, t::Tagged) = n * value(t)

value(t::Tagged) = t.point

function any_inside(geometry)
    any_inside(dimension(geometry), geometry)
end

function any_inside{N}(::Type{Val{N}}, geometry)
    point = gt.any_inside(geometry)
    Tagged(convert(SVector{N}, point...))
end

function any_inside{N, T}(mesh::gt.AbstractMesh{gt.Point{N, T}})
    point = convert(SVector{N, T}, first(gt.vertices(mesh))...)
    Tagged(point)
end

function any_inside(m::gt.MinkowskiDifference)
    t1 = any_inside(m.c1)
    t2 = any_inside(m.c2)
    Difference(t1, t2)
end
