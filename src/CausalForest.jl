__precompile__()

module CausalForest

import Base: length, show, convert, promote_rule, zero
using DecisionTree
using DelimitedFiles
using Random
using Statistics
using NonNegLeastSquares
using LinearAlgebra
using NearestNeighbors
using Optim
import StatsBase
using XGBoost
import LinearAlgebra

export TreeOOB, EnsembleOOB, apply_tree_oob, build_forest_oob, apply_forest_oob, load_data,
    build_tree, build_forest, apply_forest, shaff, knn_sensi, importance, knn_causal_sensi,
    build_forest_1, apply_forest_1, build_forest_2, apply_forest_2, apply_forest_2bis,
    build_forest_opti, build_forest_1_opti, build_forest_1_opti_corr, build_forest_3_opti,
    apply_forest_3, build_forest_3_opti_xgboost, apply_forest_3_xgboost, build_forest_3_opti_ols, apply_forest_3_ols,
    get_occurence_frequencies, sample_U, apply_forest_oob # pour test

#####Includes#####

include("util.jl")
include("load_data.jl")
include("oob_random_forest.jl")
include("classification/tree.jl")
include("classification/main.jl")
include("regression/tree.jl")
include("regression/main.jl")

########## Types ##########

mutable struct LeafCausalH # honest
    inds_build   :: Vector{Int} # indices in the leaf during construction
    inds_pred    :: Vector{Int} # indices to predict these are the same if no honesty
    label        :: Union{Nothing, Float64} # mean causal effect in leaf for Breiman like approach
end

mutable struct LeafCausalNH # nothonest
    inds_build   :: Vector{Int} # indices in the leaf
    label        :: Union{Nothing, Float64} # mean causal effect in leaf
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

LeafOrNodeCausalNH{S} = Union{LeafCausalNH, NodeCausalNH{S}}

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
    model_Y   :: Union{Nothing, EnsembleOOB{S, Float64}, Ensemble{S, Float64}, XGBoost.Booster}
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
include("causal_new_1/tree.jl")
include("causal_new_1/main.jl")
include("causal_new_2/tree.jl")
include("causal_new_2/main.jl")
include("causal_opti/tree.jl")
include("causal_opti/main.jl")
include("causal_new_1_opti/tree.jl")
include("causal_new_1_opti/main.jl")
include("causal_new_1_opti_corr/tree.jl")
include("causal_new_1_opti_corr/main.jl")
include("causal_new_3_opti/tree.jl")
include("causal_new_3_opti/main.jl")
include("causal_new_3_opti_xgboost/tree.jl")
include("causal_new_3_opti_xgboost/main.jl")
include("causal_new_3_opti_ols/tree.jl")
include("causal_new_3_opti_ols/main.jl")
include("sensitivity/sampling.jl")
include("sensitivity/prf.jl")
include("sensitivity/shaff.jl")
include("sensitivity/knn.jl")
include("sensitivity/importance.jl")
include("sensitivity/knnforest.jl")

end #module
