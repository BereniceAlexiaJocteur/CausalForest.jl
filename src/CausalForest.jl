__precompile__()

module CausalForest

import Base: length, show, convert, promote_rule, zero
using DecisionTree
using DelimitedFiles
using Random
using Statistics

export TreeOOB, EnsembleOOB, apply_tree_oob, build_forest_oob, apply_forest_oob, load_data

#####Includes#####

include("util.jl")
include("load_data.jl")
include("oob_random_forest.jl")
include("classification/tree.jl")
include("classification/main.jl")
include("regression/tree.jl")
include("regression/main.jl")

end #module
