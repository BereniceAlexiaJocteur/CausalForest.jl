using Documenter
using CausalForest

makedocs(
    sitename = "CausalForest.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/BereniceAlexiaJocteur/CausalForest.jl.git",
    devbranch = "main"
)
