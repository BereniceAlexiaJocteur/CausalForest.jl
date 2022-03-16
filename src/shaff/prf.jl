function _empty!(
    tree   :: LeafOrNodeCausalH{S}
    ) where {S}
    if is_leaf(tree)
        tree.inds_build = Int[]
        tree.inds_pred = Int[]
    else
        _empty!(tree.left)
        _empty!(tree.right)
    end
end


function _empty!(
    tree   :: LeafOrNodeCausalNH{S}
    ) where {S}
    if is_leaf(tree)
        tree.inds_build = Int[]
    else
        _empty!(tree.left)
        _empty!(tree.right)
    end
end

function _fill_proj!(
    tree       :: LeafOrNodeCausalH{S},
    inds_build :: Vector{Int},
    inds_pred  :: Vector{Int},
    X          :: AbstractMatrix{S},
    set_u      :: Set{Int}
    ) where {S}

    if is_leaf(tree)
        tree.inds_build = inds_build
        tree.inds_pred = inds_pred
    elseif tree.featid in set_u
        _fill_proj!(tree.left, inds_build, inds_pred, X, set_u)
        _fill_proj!(tree.right, inds_build, inds_pred, X, set_u)
    else
        n_build = length(inds_build)
        n_pred = length(inds_pred)
        build_left = Int[]
        build_right = Int[]
        pred_left = Int[]
        pred_right = Int[]
        for i in 1:n_build
            if X[inds_build[i], tree.featid] < tree.featval
                push!(build_left, inds_build[i])
            else
                push!(build_right, inds_build[i])
            end
        end
        for i in 1:n_pred
            if X[inds_pred[i], tree.featid] < tree.featval
                push!(pred_left, inds_pred[i])
            else
                push!(pred_right, inds_pred[i])
            end
        end
        _fill_proj!(tree.left, build_left, pred_left, X, set_u)
        _fill_proj!(tree.right, build_right, pred_right, X, set_u)
    end
end

function _fill_proj!(
    tree       :: LeafOrNodeCausalNH{S},
    inds_build :: Vector{Int},
    X          :: AbstractMatrix{S},
    set_u      :: Set{Int}
    ) where {S}

    if is_leaf(tree)
        tree.inds_build = inds_build
    elseif tree.featid in set_u
        _fill_proj!(tree.left, inds_build, X, set_u)
        _fill_proj!(tree.right, inds_build, X, set_u)
    else
        n_build = length(inds_build)
        build_left = Int[]
        build_right = Int[]
        for i in 1:n_build
            if X[inds_build[i], tree.featid] < tree.featval
                push!(build_left, inds_build[i])
            else
                push!(build_right, inds_build[i])
            end
        end
        _fill_proj!(tree.left, build_left, X, set_u)
        _fill_proj!(tree.right, build_right, X, set_u)
    end
end

function build_projected_tree(
    tree   :: TreeCausalH{S},
    set_u  :: Set{Int},
    X      :: AbstractMatrix{S}
    ) where {S}

    inds_build = tree.inds_build
    inds_pred = tree.inds_pred
    root = deecopy(tree.tree) # TODO verif si pas deepcopy plutot si juste copy il faut la definir a la main  Base.copy(s::Nodeetc)

    _empty!(root)
    _fill_proj!(root, inds_build, inds_pred, X, set_u)

    return TreeCausalH{S}(root, inds_build, inds_pred, tree.oob)

end


function build_projected_tree(
    tree   :: TreeCausalNH{S},
    set_u  :: Set{Int},
    X      :: AbstractMatrix{S}
    ) where {S}

    inds = tree.inds
    root = deepcopy(tree.tree) # TODO verif si pas deepcopy plutot

    _empty!(root)
    _fill_proj!(root, inds, X, set_u)

    return TreeCausalNH{S}(root, inds, tree.oob)

end

function apply_forest_oob(
    forest :: EnsembleCausal{S}
    ) where {S}

    centering = forest.centering
    n_samples = length(forest.Y)
    n_trees = length(forest)
    predictions = Array{Float64}(undef, n_samples)
    if centering
        Y_vec =  forest.Y_center
        T_vec =  forest.T_center
    else
        Y_vec = forest.Y
        T_vec = forest.T
    end

    for j in 1:n_samples

        alpha = zeros(n_samples)
        nb_oob_samples = 0

        for b in 1:n_trees
            if j in forest.trees[b].oob
                nb_oob_samples += 1
                N = neighbours(forest, b, forest.X[j, :]) #TODO attention x avant mais mort  --> forest.X[j,:] ??
                l = length(N)
                for e in N
                    alpha[e] += 1/l
                end
            end
        end
        alpha = alpha/nb_oob_samples
        T_alpha = 0
        Y_alpha = 0
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
        predictions[j] = num/denom
    end
    return predictions
end

function prf_causal_effect( # causal effect on oob
    forest   :: EnsembleCausal{S},
    set_u    :: Set{Int}
    ) where {S}

    n_trees = length(forest.trees)

    if forest.trees isa Vector{TreeCausalH{S}}
        trees = Vector{TreeCausalH{S}}(undef, n_trees)
    else
        trees = Vector{TreeCausalNH{S}}(undef, n_trees)
    end

    for i in 1:n_trees
        trees[i] = build_projected_tree(forest.trees[i], set_u, forest.X)
    end

    prf = EnsembleCausal{S}(trees, forest.centering, forest.bootstrap, forest.honest,
        forest.X, forest.Y, forest.T, forest.model_Y, forest.model_T, forest.Y_center,
        forest.T_center)

    return apply_forest_oob(prf)
end
