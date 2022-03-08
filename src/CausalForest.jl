__precompile__()

module CausalForest

import Base: length, show, convert, promote_rule, zero
using DecisionTree
using DelimitedFiles
using Random
using Statistics
import StatsBase

export TreeOOB, EnsembleOOB, apply_tree_oob, build_forest_oob, apply_forest_oob, load_data,
    build_tree, build_forest, apply_forest

#####Includes#####

include("util.jl")
include("load_data.jl")
include("oob_random_forest.jl")
include("classification/tree.jl")
include("classification/main.jl")
include("regression/tree.jl")
include("regression/main.jl")

########## Types ##########

struct LeafCausalH # honest
    inds_build   :: Vector{Int} # indices in the leaf during construction
    inds_pred    :: Vector{Int} # indices to predict these are the same if no honesty
end

struct LeafCausalNH # nothonest
    inds_build   :: Vector{Int} # indices in the leaf
end

struct NodeCausalH{S}
    featid  :: Int
    featval :: S
    left    :: Union{LeafCausalH, NodeCausalH{S}}
    right   :: Union{LeafCausalH, NodeCausalH{S}}
end

struct NodeCausalNH{S}
    featid  :: Int
    featval :: S
    left    :: Union{LeafCausalNH, NodeCausalNH{S}}
    right   :: Union{LeafCausalNH, NodeCausalNH{S}}
end

LeafOrNodeCausalH{S} = Union{LeafCausalH, NodeCausalH{S}}

const LeafOrNodeCausalNH{S} = Union{LeafCausalNH, NodeCausalNH{S}}

struct TreeCausalH{S} # honest causal tree
    tree       :: LeafOrNodeCausalH{S}
    inds_build :: Vector{Int}
    inds_pred  :: Vector{Int}
    oob        :: Vector{Int}
end

struct TreeCausalNH{S} # honest causal tree
    tree :: LeafOrNodeCausalNH{S}
    inds :: Vector{Int}
    oob  :: Vector{Int}
end

struct EnsembleCausal{S}
    trees     :: Union{Vector{TreeCausalH{S}}, Vector{TreeCausalNH{S}}}
    centering :: Bool
    bootstrap :: Bool
    honest    :: Bool
    X         :: AbstractMatrix{S}
    Y         :: AbstractVector{Float64}
    T         :: AbstractVector{Int}
    model_Y   :: Union{Nothing, EnsembleOOB{S, Float64}}
    model_T   :: Union{Nothing, EnsembleOOB{S, Int}}
    Y_center  :: Union{Nothing, AbstractVector{Float64}}
    T_center  :: Union{Nothing, AbstractVector{Int}}
end


is_leaf(l::LeafCausalH) = true
is_leaf(l::LeafCausalNH) = true
is_leaf(n::NodeCausalH) = false
is_leaf(n::NodeCausalNH) = false


length(forest::EnsembleCausal) = length(forest.trees)

#####Includes#####

include("causal/tree.jl")
include("causal/main.jl")


end #module
