using IJulia
jupyter = IJulia.jupyter

for notebook in ["../examples/manipulator.ipynb", "../examples/squishable.ipynb", "../examples/deformable_manipulator.ipynb"]
    run(`$jupyter nbconvert --to notebook --execute $notebook --output $notebook`)
end
