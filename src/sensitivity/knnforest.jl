function get_last_node(
    node         :: LeafOrNodeCausalH{S},
    last_node    :: LeafOrNodeCausalH{S},
    set_features :: Set{Int},
    x            :: AbstractVector{S}
    ) where {S}

    if is_leaf(node)
        return last_node
    elseif  !(node.featid in set_features)
        last_node = node
    end

    if x[node.featid]<node.featval
        return get_last_node(node.left, last_node, set_features, x)
    else
        return get_last_node(node.right, last_node, set_features, x)
    end
end

function get_last_node(
    node         :: LeafOrNodeCausalNH{S},
    last_node    :: LeafOrNodeCausalNH{S},
    set_features :: Set{Int},
    x            :: AbstractVector{S}
    ) where {S}

    if is_leaf(node)
        return last_node
    elseif  !(node.featid in set_features)
        last_node = node
    end

    if x[node.featid]<node.featval
        return get_last_node(node.left, last_node, set_features, x)
    else
        return get_last_node(node.right, last_node, set_features, x)
    end
end

function get_sub_tree(
    node :: LeafOrNodeCausalH{S}
    ) where {S}

    if is_leaf(node)
        return node.inds_pred
    else
        return vcat(get_sub_tree(node.left), get_sub_tree(node.right))
    end
end

function get_sub_tree(
    node :: LeafOrNodeCausalNH{S}
    ) where {S}

    if is_leaf(node)
        return node.inds_build
    else
        return vcat(get_sub_tree(node.left), get_sub_tree(node.right))
    end
end

function get_knn_causal(
    forest       :: EnsembleCausal{S},
    set_features :: Set{Int},
    x            :: AbstractVector{S},
    k            :: Int
    ) where {S}

    n_samples = length(forest.Y)
    alpha = zeros(n_samples)
    n_trees = length(forest)
    for b in 1:n_trees
        last_node = get_last_node(forest.trees[b].tree, forest.trees[b].tree, set_features, x)
        N = get_sub_tree(last_node)
        for e in N
            alpha[e] += 1
        end
    end

    return partialsortperm(alpha, 1:k, rev=true)
end

function cost_knn_causal(
    forest              :: EnsembleCausal{S},
    set_u               :: Set{Int},
    base_causal_effect  :: Vector{Float64},
    k                   :: Int
    ) where {S}

    n = length(forest.Y)
    som = 0
    projected_causal_effect = [mean(base_causal_effect[get_knn_causal(forest, set_u, forest.X[i,:], k)]) for i in 1:n]

    for i in 1:n
        som += (base_causal_effect[i] - projected_causal_effect[i])^2
    end

    som /= n * var(base_causal_effect)
    return 1 - som

end

function estimate_Shapley_knn_causal(
    forest    :: EnsembleCausal{S},
    dict_sets :: Dict{Set{Int}, Int},
    K         :: Int,
    proba_set :: Dict{Set{Int}, Float64},
    k         :: Int
    ) where {S}

    n_imp = size(forest.X, 2)

    w_vect = cat([sqrt((weight(U, n_imp)*dict_sets[U])/(K*proba_set[U])) for U in keys(dict_sets)],
                 [sqrt((weight(U, n_imp)*dict_sets[U])/(K*proba_set[U])) for U in keys(dict_sets)],
                 10000, 10000; dims = 1)
    w_matrix = diagm(w_vect)

    z_vect = cat([to_binary_vector(U, n_imp) for U in keys(dict_sets)],
                 [to_binary_vector_comp(U, n_imp) for U in keys(dict_sets)],
                 [[0 for i=1:n_imp]], [[1 for i=1:n_imp]]; dims = 1)
    z_matrix = vec(z_vect)

    cause = apply_forest_oob(forest)

    cost_matrix = cat([transpose(cost_knn_causal(forest, set_u, cause, k)) for set_u in keys(dict_sets)],
                      [transpose(cost_knn_causal(forest, setdiff(Set(1:n_imp), set_u), cause, k)) for set_u in keys(dict_sets)],
                      0, 1; dims = 1)

    A_matrix = w_matrix * z_matrix
    A_matrix = reduce(hcat, A_matrix)'

    b_matrix = w_matrix * cost_matrix

    beta = nonneg_lsq(A_matrix, b_matrix)

    return beta

end


function knn_causal_sensi(forest :: EnsembleCausal{S}, K :: Int, k = -1) where {S}
    freq = get_occurence_frequencies(forest)
    samp = sample_U(freq, K)
    dict_samp = StatsBase.countmap(samp)
    if k == -1
        k = round(Int, sqrt(length(forest.Y)))
    end
    return estimate_Shapley_knn_causal(forest, dict_samp, K, freq, k)
end
