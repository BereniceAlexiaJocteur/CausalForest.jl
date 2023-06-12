import DecisionTree
import MLBase

function split_inds(inds::Vector{Int}, percentage::Float64)

    if !(0.0 < percentage < 1.0)
        throw("percentage must be in the range (0,1)")
    end

    N = length(inds)
    splitindex = round(Int, percentage*N)
    copy_inds = copy(inds)
    shuffle!(copy_inds)
    return copy_inds[splitindex+1:N], copy_inds[1:splitindex]
end

function _convertNH(node::treecausation.NodeMeta{S}, indX::Vector{Int}) where {S}
    if node.is_leaf
        return LeafCausalNH(indX[node.region], 0)
    else
        left = _convertNH(node.l, indX)
        right = _convertNH(node.r, indX)
        return NodeCausalNH{S}(node.feature, node.threshold, left, right)
    end
end

function _convertH(node::treecausation.NodeMeta{S}, indX::Vector{Int}) where {S}
    if node.is_leaf
        return LeafCausalH(indX[node.region], Vector{Int}(), 0)
    else
        left = _convertH(node.l, indX)
        right = _convertH(node.r, indX)
        return NodeCausalH{S}(node.feature, node.threshold, left, right)
    end
end

function _fill!(
        node     :: LeafOrNodeCausalH{S},
        ind      :: Int,
        X        :: Vector{S}) where {S}

    if is_leaf(node)
        append!(node.inds_pred, ind)
    else
        if X[node.featid] < node.featval
            _fill!(node.left, ind, X)
        else
            _fill!(node.right, ind, X)
        end
    end
end

function fill_treeH(
        node     :: treecausation.NodeMeta{S},
        indspred :: AbstractVector{Int},
        indX     :: Vector{Int},
        X        :: AbstractMatrix{S},
        Y        :: Vector{Float64},
        T        :: Vector{Int}) where {S}

    nodeH = _convertH(node, indX)
    n_samples = length(indspred)
    for i in 1:n_samples
        ind = indspred[i]
        X_obs = X[ind, :]
        _fill!(nodeH, ind, X_obs)
    end
    return nodeH
end

function fill_treeNH(
        node     :: treecausation.NodeMeta{S},
        indX     :: Vector{Int},
        X        :: AbstractMatrix{S},
        Y        :: Vector{Float64},
        T        :: Vector{Int}) where {S}

    nodeNH = _convertNH(node, indX)
    return nodeNH
end


function build_tree(
        honest             :: Bool,
        indsbuild          :: AbstractVector{Int},
        indspred           :: Union{Nothing, AbstractVector{Int}},
        labels             :: AbstractVector{T},
        treatment          :: AbstractVector{Int},
        features           :: AbstractMatrix{S},
        const_mtry         :: Bool,
        m_pois              = -1,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if max_depth == -1
        max_depth = typemax(Int)
    end

    if m_pois == -1 && !const_mtry
        p = size(features, 2)
        m_pois = floor(Int, min(sqrt(p)+20, p))
    end

    if m_pois == -1 && const_mtry
        p = size(features, 2)
        m_pois = floor(Int, sqrt(p))
    end

    rng = mk_rng(rng)::Random.AbstractRNG

    t = treecausation.fit(
        X                   = features,
        Y                   = labels,
        W                   = treatment,
        indX                = indsbuild,
        const_mtry          = const_mtry,
        m_pois              = Int(m_pois),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        rng                 = rng)

    if honest
        return TreeCausalH{S}(fill_treeH(t.root, indspred, t.inds, features, labels, treatment), indsbuild, indspred, setdiff(collect(1:length(labels)), union(indsbuild, indspred)))
    else
        return TreeCausalNH{S}(fill_treeNH(t.root, t.inds, features, labels, treatment), indsbuild, setdiff(collect(1:length(labels)), indsbuild))
    end
end

"""
Build a causal forest.

- if `bootstrap=True` we sample for each tree via bootstrap else we use subsampling
- if `honest=True` we use 2 samples one too build splits and the other one to fill leaves
    otherwise we use the whole sample for the two steps
- if `const_mtry=True` we use a constant mtry otherwise we use a random mtry following
    `min(max(Poisson(m_pois),1),number_of_features)`
- if `m_pois=-1` we set mtry to `sqrt(number_of_features)` else mtry is m_pois
- if `optimisatio=true` we use cross validation to tune regression random forest of gamma
"""
function build_forest(
    bootstrap          :: Bool,
    honest             :: Bool,
    labels             :: AbstractVector{T},
    treatment          :: AbstractVector{Int},
    features           :: AbstractMatrix{S},
    const_mtry         :: Bool,
    m_pois              = -1,
    n_trees            ::Int = 10,
    n_trees_centering  ::Int = 100,
    optimisation       ::Bool = true;
    partial_sampling    = 0.7,
    honest_proportion   = 0.5,
    max_depth           = -1,
    min_samples_leaf   ::Int = 5,
    min_samples_split  ::Int = 10,
    rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)&&!bootstrap
        throw("partial_sampling must be in the range (0,1]")
    end
    if !(0.0 < honest_proportion <= 1.0)&&honest
        throw("honest_proportion must be in the range (0,1]")
    end

    n_tot_samples = length(labels)

    if bootstrap
        n_samples = length(labels)
    else
        n_samples = floor(Int, partial_sampling * length(labels))
    end

    if honest
        forest = Vector{TreeCausalH{S}}(undef, n_trees)
    else
        forest = Vector{TreeCausalNH{S}}(undef, n_trees)
    end

    if sum(treatment) > 0.55 * n_tot_samples
        model_T = -1
        T_center = 1 .- treatment
    else
        model_T = 1
        T_center = treatment
    end

    if optimisation #TODO verifier crossvalidation tuning

        np = size(features, 2)
        min_split = [5, 10, 25]
        min_leaf = [2, 5, 10]
        mtry = [round(Int, sqrt(np)), round(Int, np/3), np]
        G = Iterators.product(min_leaf, min_split, mtry)
        sc = []
        p = []

        for g in G
            model = build_forest_oob(labels[T_center.==0], features[T_center.==0, :], g[3], n_trees_centering, 0.7, -1, g[1], g[2])
            perf  = StatsBase.rmsd(labels[T_center.==0], apply_forest_oob(model))

            push!(sc, perf)
            push!(p, (g[1], g[2], g[3]))
        end
        ind_opt = argmin(sc)
        model_Y = DecisionTree.build_forest(labels[T_center.==0], features[T_center.==0,:], p[ind_opt][3], n_trees_centering, 0.7, -1, p[ind_opt][1], p[ind_opt][2])
    else
        model_Y = DecisionTree.build_forest(labels[T_center.==0], features[T_center.==0,:], -1, n_trees_centering)
    end
    #Y_center = labels - DecisionTree.apply_forest(model_Y, features) TODO
    Y_center = labels
    Y_vec = Y_center
    T_vec = T_center


    if rng isa Random.AbstractRNG
        if bootstrap
            Threads.@threads for i in 1:n_trees
                if honest
                    inds = collect(1:n_tot_samples)
                    inds1, inds2 = split_inds(inds, honest_proportion)
                    indsbuild = rand(rng, inds1, length(inds1))
                    indspred = rand(rng, inds2, length(inds2))
                else
                    indsbuild = rand(rng, 1:n_tot_samples, n_samples)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    Y_vec,
                    T_vec,
                    features,
                    const_mtry,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                    rng = rng)
            end
        else
            Threads.@threads for i in 1:n_trees
                if honest
                    inds = StatsBase.sample(rng, 1:n_tot_samples, n_samples; replace=false)
                    indsbuild, indspred = split_inds(inds, honest_proportion)
                else
                    indsbuild = StatsBase.sample(rng, 1:n_tot_samples, n_samples; replace=false)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    Y_vec,
                    T_vec,
                    features,
                    const_mtry,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                    rng = rng)
            end
        end
    elseif rng isa Integer # each thread gets its own seeded rng
        if bootstrap
            Threads.@threads for i in 1:n_trees
                Random.seed!(rng + i)
                if honest
                    inds = collect(1:n_tot_samples)
                    inds1, inds2 = split_inds(inds, honest_proportion)
                    indsbuild = rand(inds1, length(inds1))
                    indspred = rand(inds2, length(inds2))
                else
                    indsbuild = rand(1:n_tot_samples, n_samples)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    Y_vec,
                    T_vec,
                    features,
                    const_mtry,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split)
            end
        else
            Threads.@threads for i in 1:n_trees
                Random.seed!(rng + i)
                if honest
                    inds = StatsBase.sample(1:n_tot_samples, n_samples; replace=false)
                    indsbuild, indspred = split_inds(inds, honest_proportion)
                else
                    indsbuild = StatsBase.sample(1:n_tot_samples, n_samples; replace=false)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    Y_vec,
                    T_vec,
                    features,
                    const_mtry,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split)
            end
        end
    else
        throw("rng must of be type Integer or Random.AbstractRNG")
    end

    return EnsembleCausal{S}(forest, bootstrap, honest, features, labels, treatment, model_Y, model_T, Y_center, T_center)
end

apply_treeH(leaf :: LeafCausalH, x :: AbstractVector{S}) where {S} = leaf.inds_pred
apply_treeNH(leaf :: LeafCausalNH, x :: AbstractVector{S}) where {S} = leaf.inds_build

function apply_treeH(tree :: NodeCausalH{S}, x :: AbstractVector{S}) where {S}
    if tree.featid == 0
        return apply_treeH(tree.left, x)
    elseif x[tree.featid] < tree.featval
        return apply_treeH(tree.left, x)
    else
        return apply_treeH(tree.right, x)
    end
end

function apply_treeNH(tree :: NodeCausalNH{S}, x :: AbstractVector{S}) where {S}
    if tree.featid == 0
        return apply_treeNH(tree.left, x)
    elseif x[tree.featid] < tree.featval
        return apply_treeNH(tree.left, x)
    else
        return apply_treeNH(tree.right, x)
    end
end

function neighbours(
    forest :: EnsembleCausal{S},
    b      :: Int,
    x      :: AbstractVector{S}
    ) where {S}
    if forest.honest
        return apply_treeH(forest.trees[b].tree, x)
    else
        return apply_treeNH(forest.trees[b].tree, x)
    end
end

function apply_forest(
    forest :: EnsembleCausal{S},
    x      :: AbstractVector{S}
    ) where {S}

    n_samples = length(forest.Y)
    alpha_pos = zeros(n_samples)
    alpha_neg = zeros(n_samples)
    n_trees = length(forest)
    for b in 1:n_trees
        N = neighbours(forest, b, x)
        l_pos = 0
        l_neg = 0
        for e in N
            if forest.T[e] == 1
                l_pos += 1
            else
                l_neg += 1
            end
        end
        for e in N
            if forest.T[e] == 1
                alpha_pos[e] += 1/l_pos
            else
                alpha_neg[e] += 1/l_neg
            end
        end
    end
    alpha_pos = alpha_pos/n_trees
    alpha_neg = alpha_neg/n_trees
    pos = 0
    neg = 0
    for i in 1:n_samples
        if forest.T[i] == 1
            pos += forest.Y_center[i]*alpha_pos[i]
        else
            neg += forest.Y_center[i]*alpha_neg[i]
        end
    end
    return pos-neg
end

"""
Get the causal effect for each row in x given a causal forest
"""
function apply_forest(
    forest :: EnsembleCausal{S},
    x      :: AbstractMatrix{S}
    ) where {S}

    N = size(x, 1)
    predictions = Array{Float64}(undef, N)
    for i in 1:N
        predictions[i] = apply_forest(forest, x[i, :])
    end
    return predictions
end
