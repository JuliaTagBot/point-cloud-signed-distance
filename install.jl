type DevRequirement
    url::ASCIIString
    sha::ASCIIString
    name::ASCIIString
end

DevRequirement(url::AbstractString, sha::AbstractString) = DevRequirement(url, sha, extract_pkg_name(url))

function checkout(req::DevRequirement)
    cd(Pkg.dir(req.name)) do
        run(`git remote set-url origin $(req.url)`)
        run(`git fetch origin`)
        run(`git checkout $(req.sha)`)
    end
end

function clone_or_checkout(req::DevRequirement)
    if !isdir(Pkg.dir(req.name))
        Pkg.clone(req.url)
    end
    checkout(req)
end


function extract_pkg_name(url::AbstractString)
    m = match(r"(?:^|[/\\])(\w+?)(?:\.jl)?(?:\.git)?$", url)
    if m == nothing
        throw(PkgError("can't determine package name from URL: $url"))
    else
        return m.captures[1]
    end
end

function parse_dev_requirements(dev_file)
    reqs = Vector{DevRequirement}()
    for line in split(open(readall, dev_file), '\n')
        line = replace(line, r"#.*", "")
        line = strip(line)
        m = match(r"([^\s]+)\s+([^\s]+)", line)
        if m != nothing
            push!(reqs, DevRequirement(m.captures[1], m.captures[2]))
        end
    end
    reqs
end

function install()
    user_reqfile = "REQUIRE"
    if !isdir(Pkg.dir())
        mkpath(Pkg.dir())
        Pkg.init()
    end
    cp(user_reqfile, joinpath(Pkg.dir(), "REQUIRE"); remove_destination=true)
    dev_reqs = parse_dev_requirements("REQUIRE.dev")
    map(clone_or_checkout, dev_reqs)
    Pkg.resolve()
end

install()
