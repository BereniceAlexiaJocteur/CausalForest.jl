function _convert(node::treeregressor.NodeMeta{S}, labels::Array{T}) where {S, T <: Float64}
    if node.is_leaf
        return Leaf{T}(node.label, labels[node.region])
    else
        left = _convert(node.l, labels)
        right = _convert(node.r, labels)
        return Node{S, T}(node.feature, node.threshold, left, right)
    end
end

function _convertOOB(
        node   :: treeregressor.NodeMeta{S},
        labels :: Array{T},
        inds   :: AbstractVector{Int},
        oob    :: AbstractMatrix{S}) where {S, T}

    tree = _convert(node, labels)
    return TreeOOB{S, T}(tree, inds, oob)
end

function build_tree(
        inds               :: AbstractVector{Int},
        labels             :: AbstractVector{T},
        features           :: AbstractMatrix{S},
        n_subfeatures       = 0,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if max_depth == -1
        max_depth = typemax(Int)
    end
    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end

    rng = mk_rng(rng)::Random.AbstractRNG
    t = treeregressor.fit(
        X                   = features[inds, :],
        Y                   = labels[inds],
        W                   = nothing,
        max_features        = Int(n_subfeatures),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = rng)

    return _convertOOB(t.root, labels[t.labels], setdiff(collect(1:length(labels)), inds),
        features[setdiff(collect(1:length(labels)), inds), :])
end

function build_forest_oob(
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        n_subfeatures       = -1,
        n_trees             = 10,
        partial_sampling    = 0.7,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)
        throw("partial_sampling must be in the range (0,1]")
    end

    if n_subfeatures == -1
        n_features = size(features, 2)
        n_subfeatures = round(Int, sqrt(n_features))
    end

    t_samples = length(labels)
    n_samples = floor(Int, partial_sampling * t_samples)

    forest = Vector{TreeOOB{S, T}}(undef, n_trees)

    if rng isa Random.AbstractRNG
        Threads.@threads for i in 1:n_trees
            inds = rand(rng, 1:t_samples, n_samples)
            forest[i] = build_tree(
                inds,
                labels,
                features,
                n_subfeatures,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
                rng = rng)
        end
    elseif rng isa Integer # each thread gets its own seeded rng
        Threads.@threads for i in 1:n_trees
            Random.seed!(rng + i)
            inds = rand(1:t_samples, n_samples)
            forest[i] = build_tree(
                inds,
                labels,
                features,
                n_subfeatures,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase)
        end
    else
        throw("rng must of be type Integer or Random.AbstractRNG")
    end

    return EnsembleOOB{S, T}(length(labels), features, forest)
end
