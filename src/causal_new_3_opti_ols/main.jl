function _convertNH(node::treecausation_3_opti_ols.NodeMeta{S}, indX::Vector{Int}) where {S}
    if node.is_leaf
        return LeafCausalNH(indX[node.region], 0)
    else
        left = _convertNH(node.l, indX)
        right = _convertNH(node.r, indX)
        return NodeCausalNH{S}(node.feature, node.threshold, left, right)
    end
end

function _convertH(node::treecausation_3_opti_ols.NodeMeta{S}, indX::Vector{Int}) where {S}
    if node.is_leaf
        return LeafCausalH(indX[node.region], Vector{Int}(), 0)
    else
        left = _convertH(node.l, indX)
        right = _convertH(node.r, indX)
        return NodeCausalH{S}(node.feature, node.threshold, left, right)
    end
end

function fill_treeH(
        node     :: treecausation_3_opti_ols.NodeMeta{S},
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
        node     :: treecausation_3_opti_ols.NodeMeta{S},
        indX     :: Vector{Int},
        X        :: AbstractMatrix{S},
        Y        :: Vector{Float64},
        T        :: Vector{Int}) where {S}

    nodeNH = _convertNH(node, indX)
    return nodeNH
end


function build_tree_3_opti_ols(
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

    t = treecausation_3_opti_ols.fit(
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

- if `centering=True` Y and W are centered else they stay unchanged
- if `bootstrap=True` we sample for each tree via bootstrap else we use subsampling
- if `honest=True` we use 2 samples one too build splits and the other one to fill leaves
    otherwise we use the whole sample for the two steps
- if `const_mtry=True` we use a constant mtry otherwise we use a random mtry following
    `min(max(Poisson(m_pois),1),number_of_features)`
"""
function build_forest_3_opti_ols(
    centering          :: Bool,
    bootstrap          :: Bool,
    honest             :: Bool,
    labels             :: AbstractVector{T},
    treatment          :: AbstractVector{Int},
    features           :: AbstractMatrix{S},
    const_mtry         :: Bool,
    m_pois              = -1,
    n_trees             = 10,
    partial_sampling    = 0.7,
    honest_proportion   = 0.5,
    max_depth           = -1,
    min_samples_leaf    = 5,
    min_samples_split   = 10,
    n_trees_centering   = 100;
    rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)&&!bootstrap
        throw("partial_sampling must be in the range (0,1]")
    end
    if centering
        throw("no centering for this splitting criterion")
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

    model_Y = nothing
    z = hcat(ones(length(labels[treatment.==0,:]),1), features[treatment.==0,:])
    coeff = (z'*z)\z'*labels[treatment.==0,:]
    z1 = hcat(ones(length(labels),1), features)
    Y_center = vec(labels - z1*coeff)
    println(size(labels))
    println(size(Y_center))
    println(size(z1*coeff))
    Y_vec = Y_center
    T_vec = treatment


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
                forest[i] = build_tree_3_opti_ols(
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
                forest[i] = build_tree_3_opti_ols(
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
                forest[i] = build_tree_3_opti_ols(
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
                forest[i] = build_tree_3_opti_ols(
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

    return EnsembleCausal{S}(forest, centering, bootstrap, honest, features, labels, treatment, model_Y, nothing, Y_center, nothing)
end

function apply_forest_3_ols(
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
function apply_forest_3_ols(
    forest :: EnsembleCausal{S},
    x      :: AbstractMatrix{S}
    ) where {S}

    N = size(x, 1)
    predictions = Array{Float64}(undef, N)
    for i in 1:N
        predictions[i] = apply_forest_3_ols(forest, x[i, :])
    end
    return predictions
end
