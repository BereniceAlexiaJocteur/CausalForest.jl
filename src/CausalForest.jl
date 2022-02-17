__precompile__()

module CausalForest

import Base: length, show, convert, promote_rule, zero
using DecisionTree
using DelimitedFiles
using Random
using Statistics

export TreeOOB, EnsembleOOB, apply_tree_oob, build_forest_oob, apply_forest_oob, load_data,
    build_tree

########## Types ##########

# TODO actualiser ses types pour causal forest uniquement + pas overwrite DecisionTree + ajouter à export

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
end

struct TreeCausalNH{S} # honest causal tree
    tree :: LeafOrNodeCausalNH{S}
    inds :: Vector{Int}
end

struct EnsembleCausal{S}
    trees     :: Union{Vector{LeafOrNodeCausalH{S}}, Vector{LeafOrNodeCausalNH{S}}}
    bootstrap :: Bool
    honest    :: Bool
    X         :: AbstractMatrix{S}
    Y         :: AbstractVector{Float64}
    T         :: AbstractVector{Int}
end


is_leaf(l::LeafCausalH) = true
is_leaf(l::LeafCausalNH) = true
is_leaf(n::NodeCausalH) = false
is_leaf(n::NodeCausalNH) = false

#TODO
#convert(::Type{Node{S, T}}, lf::Leaf{T}) where {S, T} = Node(0, zero(S), lf, Leaf(zero(T), [zero(T)]))
#promote_rule(::Type{Node{S, T}}, ::Type{Leaf{T}}) where {S, T} = Node{S, T}
#promote_rule(::Type{Leaf{T}}, ::Type{Node{S, T}}) where {S, T} = Node{S, T}


#####Includes#####

include("util.jl")
include("load_data.jl")
include("oob_random_forest.jl")
include("classification/tree.jl")
include("classification/main.jl")
include("regression/tree.jl")
include("regression/main.jl")
include("causal/tree.jl")
include("causal/main.jl")


end #module
