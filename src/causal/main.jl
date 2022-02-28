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
        node     :: LeafOrNodeCausalH{S}, # TODO c'est plus node causal
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
        min_samples_split   = 2,
        min_purity_increase = 0.0;
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
        min_purity_increase = Float64(min_purity_increase),
        rng                 = rng)

    if honest
        return TreeCausalH{S}(fill_tree(t.root, indspred, t.inds, features), indsbuild, indspred)
    else
        return TreeCausalNH{S}(_convertNH(t.root, t.inds), indsbuild)
    end
end


function build_forest(
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
    min_purity_increase = 0;
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

    if rng isa Random.AbstractRNG
        if bootstrap
            Threads.@threads for i in 1:n_trees
                #inds = rand(rng, 1:t_samples, n_samples)  # TODO utiliser basestats plutot
                if honest
                    inds = collect(1:n_samples)
                    inds1, inds2 = split_inds(inds, honest_proportion)
                    #indsbuild = StatsBase.sample(rng, inds1, length(inds1); replace=true)
                    indsbuild = rand(rng, inds1, length(inds1))
                    #indspred = StatsBase.sample(rng, inds2, length(inds2); replace=true)
                    indspred = rand(rng, inds2, length(inds2))
                else
                    #indsbuild = StatsBase.sample(rng, 1:n_samples, n_samples; replace=true)
                    indsbuild = rand(rng, 1:n_samples, n_samples)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    labels,
                    treatment,
                    features,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                    min_purity_increase,
                    rng = rng)
            end
        else
            Threads.@threads for i in 1:n_trees
                #inds = rand(rng, 1:t_samples, n_samples)  # TODO utiliser basestats plutot
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
                    labels,
                    treatment,
                    features,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                    min_purity_increase,
                    rng = rng)
            end
        end
    elseif rng isa Integer # each thread gets its own seeded rng
        if bootstrap
            Threads.@threads for i in 1:n_trees
                Random.seed!(rng + i)
                #inds = rand(1:t_samples, n_samples) #TODO
                if honest
                    inds = collect(1:n_samples)
                    inds1, inds2 = split_inds(inds, honest_proportion)
                    #indsbuild = StatsBase.sample(rng, inds1, length(inds1); replace=true)
                    indsbuild = rand(rng, inds1, length(inds1))
                    #indspred = StatsBase.sample(rng, inds2, length(inds2); replace=true)
                    indspred = rand(rng, inds2, length(inds2))
                else
                    #indsbuild = StatsBase.sample(rng, 1:n_samples, n_samples; replace=true)
                    indsbuild = rand(rng, 1:n_samples, n_samples)
                    indspred = nothing
                end
                forest[i] = build_tree(
                    honest,
                    indsbuild,
                    indspred,
                    labels,
                    treatment,
                    features,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                    min_purity_increase)
            end
        else
            Threads.@threads for i in 1:n_trees
                Random.seed!(rng + i)
                #inds = rand(1:t_samples, n_samples) #TODO
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
                    labels,
                    treatment,
                    features,
                    m_pois,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                    min_purity_increase)
            end
        end
    else
        throw("rng must of be type Integer or Random.AbstractRNG")
    end

    return EnsembleCausal{S}(forest, bootstrap, honest, features, labels, treatment)
end

# TODO

function apply_forest()

end
