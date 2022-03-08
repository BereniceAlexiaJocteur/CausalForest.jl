function split_inds(inds::Vector{Int}, percentage::Float64)

    if !(0.0 < percentage < 1.0)
        throw("percentage must be in the range (0,1)")
    end

    N = length(inds)
    splitindex = round(Int, percentage*N)
    shuffle!(inds)
    return inds[splitindex+1:N], inds[1:splitindex]
end

function _convertNH(node::treecausation.NodeMeta{S}, indX::Vector{Int}) where {S}
    if node.is_leaf
        return LeafCausalNH(indX[node.region])
    else
        left = _convertNH(node.l, indX)
        right = _convertNH(node.r, indX)
        return NodeCausalNH{S}(node.feature, node.threshold, left, right)
    end
end

function _convertH(node::treecausation.NodeMeta{S}, indX::Vector{Int}) where {S}
    if node.is_leaf
        return LeafCausalH(indX[node.region], Vector{Int}())
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

function fill_tree(
        node     :: treecausation.NodeMeta{S},
        indspred :: AbstractVector{Int},
        indX     :: Vector{Int},
        X        :: AbstractMatrix{S}) where {S}

    nodeH = _convertH(node, indX)
    n_samples = length(indspred)
    for i in 1:n_samples
        ind = indspred[i]
        X_obs = X[ind, :]
        _fill!(nodeH, ind, X_obs)
    end
    return nodeH
end


function build_tree(
        honest             :: Bool,
        indsbuild          :: AbstractVector{Int},
        indspred           :: Union{Nothing, AbstractVector{Int}},
        labels             :: AbstractVector{T},
        treatment          :: AbstractVector{Int},
        features           :: AbstractMatrix{S},
        m_pois              = -1,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if max_depth == -1
        max_depth = typemax(Int)
    end

    if m_pois == -1
        p = size(features, 2)
        m_pois = min(sqrt(p)+20, p)
    end

    rng = mk_rng(rng)::Random.AbstractRNG

    t = treecausation.fit(
        X                   = features,
        Y                   = labels,
        W                   = treatment,
        indX                = indsbuild,
        m_pois              = Int(m_pois),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        rng                 = rng)

    if honest
        return TreeCausalH{S}(fill_tree(t.root, indspred, t.inds, features), indsbuild, indspred, setdiff(collect(1:length(labels)), union(indsbuild, indspred)))
    else
        return TreeCausalNH{S}(_convertNH(t.root, t.inds), indsbuild, setdiff(collect(1:length(labels)), indsbuild))
    end
end


function build_forest(
    centering          :: Bool,
    bootstrap          :: Bool,
    honest             :: Bool,
    labels             :: AbstractVector{T},
    treatment          :: AbstractVector{Int},
    features           :: AbstractMatrix{S},
    m_pois              = -1,
    n_trees             = 10,
    partial_sampling    = 0.7,
    honest_proportion   = 0.5,
    max_depth           = -1,
    min_samples_leaf    = 5,
    min_samples_split   = 2,
    n_trees_centering   = 100;
    rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)&&!bootstrap
        throw("partial_sampling must be in the range (0,1]")
    end

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

    if centering
        model_Y = build_forest_oob(labels, features, -1, n_trees_centering)
        model_T = build_forest_oob(treatment, features, -1, n_trees_centering)
        Y_center = labels - apply_forest_oob(model_Y)
        T_center = treatment - apply_forest_oob(model_T)
        Y_vec = Y_center
        T_vec = T_center
    else
        model_Y = nothing
        model_T = nothing
        Y_center = nothing
        T_center = nothing
        Y_vec = labels
        T_vec = treatment
    end

    if rng isa Random.AbstractRNG
        if bootstrap
            Threads.@threads for i in 1:n_trees
                if honest
                    inds = collect(1:n_samples)
                    inds1, inds2 = split_inds(inds, honest_proportion)
                    indsbuild = rand(rng, inds1, length(inds1))
                    indspred = rand(rng, inds2, length(inds2))
                else
                    indsbuild = rand(rng, 1:n_samples, n_samples)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    Y_vec,
                    T_vec,
                    features,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                    rng = rng)
            end
        else
            Threads.@threads for i in 1:n_trees
                if honest
                    inds = StatsBase.sample(rng, 1:n_samples, n_samples; replace=false)
                    indsbuild, indspred = split_inds(inds, honest_proportion)
                else
                    indsbuild = StatsBase.sample(rng, 1:n_samples, n_samples; replace=false)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    Y_vec,
                    T_vec,
                    features,
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
                    inds = collect(1:n_samples)
                    inds1, inds2 = split_inds(inds, honest_proportion)
                    indsbuild = rand(inds1, length(inds1))
                    indspred = rand(inds2, length(inds2))
                else
                    indsbuild = rand(1:n_samples, n_samples)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    Y_vec,
                    T_vec,
                    features,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split)
            end
        else
            Threads.@threads for i in 1:n_trees
                Random.seed!(rng + i)
                if honest
                    inds = StatsBase.sample(1:n_samples, n_samples; replace=false)
                    indsbuild, indspred = split_inds(inds, honest_proportion)
                else
                    indsbuild = StatsBase.sample(1:n_samples, n_samples; replace=false)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    Y_vec,
                    T_vec,
                    features,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split)
            end
        end
    else
        throw("rng must of be type Integer or Random.AbstractRNG")
    end

    return EnsembleCausal{S}(forest, centering, bootstrap, honest, features, labels, treatment, model_Y, model_T, Y_center, T_center)
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

    centering = forest.centering
    n_samples = length(forest.Y)
    alpha = zeros(n_samples)
    n_trees = length(forest)
    for b in 1:n_trees
        N = neighbours(forest, b, x)
        l = length(N)
        for e in N
            alpha[e] += 1/l
        end
    end
    alpha = alpha/n_trees
    T_alpha = 0
    Y_alpha = 0
    if centering
        Y_vec =  forest.Y_center
        T_vec =  forest.T_center
    else
        Y_vec = forest.Y
        T_vec = forest.T
    end
    for i in 1:n_samples
        T_alpha += alpha[i]*T_vec[i]
        Y_alpha += alpha[i]*Y_vec[i]
    end
    num = 0
    denom = 0
    for i in 1:n_samples
        diffT = T_vec[i]-T_alpha
        num += alpha[i]*diffT*(Y_vec[i]-Y_alpha)
        denom += alpha[i]*diffT^2
    end
    return num/denom
end

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
