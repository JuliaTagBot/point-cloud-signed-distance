using Base.Test
using Flash
using IJulia
using StaticArrays

@testset "Models" begin
    beanbag = Flash.Models.beanbag()
    squishable = Flash.Models.squishable()
    arm = Flash.Models.two_link_arm()
    deformable_arm = Flash.Models.two_link_arm(true)
end

@testset "Skins" begin
    model = Flash.Models.beanbag()
    state = Flash.ManipulatorState(model)
    skin = Flash.skin(state)
    @test isapprox(skin(SVector(0.0, 0, 0)), -1.0)
end

@testset "Notebooks" begin
    jupyter = IJulia.jupyter

    for notebook in ["../examples/manipulator.ipynb", "../examples/squishable.ipynb", "../examples/deformable_manipulator.ipynb", "../examples/irb140.ipynb"]
        run(`$jupyter nbconvert --to notebook --execute $notebook --output $notebook`)
    end
end
