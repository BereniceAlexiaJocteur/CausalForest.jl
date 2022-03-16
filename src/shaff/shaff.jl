# pour le nnls voir optim.jl + ajouter les gros poids pour full et empty comme dans R !!

function cost_function(
    forest              :: EnsembleCausal{S},
    set_u               :: Set{Int},
    base_causal_effect  :: Vector{Float64}
    ) where {S}

    n = length(forest.Y)
    som = 0
    projected_causal_effect = prf_causal_effect(forest, set_u)

    for i in 1:n
        som += (base_causal_effect[i] - projected_causal_effect[i])^2
    end

    som /= n * var(base_causal_effect)
    return 1 - som

end

function weight(set_u :: Set{Int}, n_imp :: Int)
    card_u = length(set_u)
    combi = binomial(n_imp, card_u)
    return (n_imp - 1) / (combi * card_u * (n_imp - card_u))
end

function to_binary_vector(set_u :: Set{Int}, n_imp :: Int)
    res = [0 for i=1:n_imp]
    for j in set_u
        res[j] = 1
    end
    return res
end

function to_binary_vector_comp(set_u :: Set{Int}, n_imp :: Int)
    res = [1 for i=1:n_imp]
    for j in set_u
        res[j] = 0
    end
    return res
end

#TODO faire estimate Shapley en regardant bien le code R de shaff + attention besoin des probas de U !!

function estimate_Shapley(
    forest    :: EnsembleCausal{S},
    dict_sets :: Dict{Set{Int}, Int},
    K         :: Int,
    proba_set :: Dict{Set{Int}, Float64}
    ) where {S}

    n_imp = size(forest.X, 2)

    w_vect = cat([sqrt((weight(U, n_imp)*dict_sets[U])/(K*proba_set[U])) for U in keys(dict_sets)],
                 [sqrt((weight(U, n_imp)*dict_sets[U])/(K*proba_set[U])) for U in keys(dict_sets)],
                 dims = 1)
    w_matrix = diagm(w_vect)

    z_vect = cat([to_binary_vector(U, n_imp) for U in keys(dict_sets)],
                 [to_binary_vector_comp(U, n_imp) for U in keys(dict_sets)],
                 dims = 1)
    z_matrix = vec(z_vect)

    A_matrix = w_matrix * z_matrix
    A_matrix = reduce(hcat, A_matrix)'

    cause = apply_forest_oob(forest)

    cost_matrix = cat([transpose(cost_function(forest, set_u, cause)) for set_u in keys(dict_sets)],
                      [transpose(cost_function(forest, setdiff(Set(1:n_imp), set_u), cause)) for set_u in keys(dict_sets)], #TODO idem sur comp
                      dims = 1)

    b_matrix = w_matrix * cost_matrix #TODO modifier dim ??

    beta = nonneg_lsq(A_matrix, b_matrix) #TODO modifier l'algo pivot vs nnls

    return beta

end

function shaff(forest :: EnsembleCausal{S}, K :: Int) where {S}
    freq = get_occurence_frequencies(forest)
    samp = sample_U(freq, K)
    dict_samp = StatsBase.countmap(samp)
    return estimate_Shapley(forest, dict_samp, K, freq)
end
