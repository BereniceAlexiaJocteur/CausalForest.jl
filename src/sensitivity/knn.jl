function knn_causal_effect(
    forest     :: EnsembleCausal{S},
    set_u      :: Set{Int},
    k          :: Int
    ) where {S}

    X = forest.X[:, sort(collect(set_u))]
    X = transpose(X) # transpose data beacaus knn considere columns as one data
    brutetree = BruteTree(X)  # not a tree just linear search
    indxs , _ = knn(brutetree, X, k)
    return indxs
end

function cost_knn(
    forest              :: EnsembleCausal{S},
    set_u               :: Set{Int},
    base_causal_effect  :: Vector{Float64},
    k                   :: Int
    ) where {S}

    n = length(forest.Y)
    som = 0
    knn_mat = knn_causal_effect(forest, set_u, k)
    projected_causal_effect = [mean(base_causal_effect[knn_mat[i]]) for i in 1:n]

    for i in 1:n
        som += (base_causal_effect[i] - projected_causal_effect[i])^2
    end

    som /= n * var(base_causal_effect)
    return 1 - som

end

function estimate_Shapley_knn(
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

    cost_matrix = cat([transpose(cost_knn(forest, set_u, cause, k)) for set_u in keys(dict_sets)],
                      [transpose(cost_knn(forest, setdiff(Set(1:n_imp), set_u), cause, k)) for set_u in keys(dict_sets)],
                      0, 1; dims = 1)

    A_matrix = w_matrix * z_matrix
    A_matrix = reduce(hcat, A_matrix)'

    b_matrix = w_matrix * cost_matrix

    beta = nonneg_lsq(A_matrix, b_matrix)

    return beta

end


function knn_sensi(forest :: EnsembleCausal{S}, K :: Int, k = -1) where {S}
    freq = get_occurence_frequencies(forest)
    samp = sample_U(freq, K)
    dict_samp = StatsBase.countmap(samp)
    if k == -1
        k = round(Int, sqrt(length(forest.Y)))
    end
    return estimate_Shapley_knn(forest, dict_samp, K, freq, k)
end
