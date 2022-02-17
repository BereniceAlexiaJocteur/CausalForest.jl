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
    t,indX = treecausation.fit(
        X                   = features[indsbuild, :],
        Y                   = labels[indsbuild],
        W                   = treatment[indsbuild],
        indX                = indsbuild,
        m_pois              = Int(m_pois),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = rng)

    #return (t, indX)
    if honest
        return fill_tree(t, indspred, indX, features)
    else
        return _convertNH(t, indX)
    end
end
