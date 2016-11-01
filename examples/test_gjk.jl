include("gjk.jl")
using DrakeVisualizer
mesh = DrakeVisualizer.contour_mesh(x -> sum((x - [2, -1.5, 0]).^2) - 1, [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])
acc = Gjk.AcceleratedMesh(mesh)
simplex = Gjk.gjk(acc)

Profile.clear_malloc_data()
simplex = Gjk.gjk(acc)
