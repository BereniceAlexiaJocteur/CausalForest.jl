using Documenter
using NewPackage

makedocs(
    sitename = "NewPackage.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/bjack205/NewPackage.jl.git",
    devbranch = "main"
)
