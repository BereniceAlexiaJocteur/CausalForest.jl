"""
Get frequency of splitting on each covariate by depth in a causal forest
"""
function get_freq_by_depth(
    tree      :: Union{TreeCausalH{S}, TreeCausalNH{S}},
    depth_max :: Int,
    num_inp   :: Int
    ) where {S}

    curr_depth = 1
    curr_lvl = [tree.tree]
    next_lvl = []
    res = zeros(depth_max, num_inp)

    while curr_depth <= depth_max
        for i in curr_lvl
            if !is_leaf(i)
                res[curr_depth, i.featid] += 1
                if !is_leaf(i.left)
                    push!(next_lvl, i.left)
                end
                if !is_leaf(i.right)
                    push!(next_lvl, i.right)
                end
            end
        end
        curr_depth += 1
        curr_lvl = next_lvl
        next_lvl = []
    end

    return res

end

"""
Get frequency based importance for each covariate in causal forest
"""
function importance(
    forest    :: EnsembleCausal{S},
    depth_max = 4,
    coeff     = 2
    ) where {S}

    n_inputs = size(forest.X, 2)
    freq = zeros(depth_max, n_inputs)

    for tree in forest.trees
        freq .+= get_freq_by_depth(tree, depth_max, n_inputs)
    end

    som = sum(freq, dims=2)

    res = zeros(n_inputs)

    for j in 1:n_inputs
        num = 0
        denom = 0
        for k in 1:depth_max
            w = float(k)^(-coeff)
            num += freq[k, j]/som[k]*w
            denom += w
        end
        res[j] = num/denom
    end

    return res

end
