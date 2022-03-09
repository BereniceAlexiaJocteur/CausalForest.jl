function get_all_paths_in_tree!( #TODO le probleme est la on va faire comme dans pythan OKOK
    tree    :: Union{LeafOrNodeCausalH{S}, LeafOrNodeCausalNH{S}},
    result  = [],
    path    = []
    ) where {S}

    push!(path, tree.featid)
    test = true

    if !is_leaf(tree.left)
        get_all_paths_in_tree!(tree.left, result, path)
        test = false
    end
    if !is_leaf(tree.right)
        get_all_paths_in_tree!(tree.right, result, path)
        test = false
    end

    if test
        push!(result, copy(path)) # TODO copy de path plutot
    end

    pop!(path)
    return result
end

function get_occurence_frequencies(forest :: EnsembleCausal{S}) where {S}
    n_trees = length(forest.trees)
    dico = Dict{Set, Float64}()
    full_set = Set(1:size(forest.X, 2))
    empty_set = Set()

    for i in 1:n_trees
        tree = forest.trees[i].tree
        paths_list = get_all_paths_in_tree!(tree)
        for j in 1:length(paths_list)
            full_path = paths_list[j]
            for k in 1:length(full_path)
                set_u = Set(full_path[1:k])
                if set_u == full_set
                    break
                end
                if haskey(dico, set_u)
                    dico[set_u] += 1
                else
                    dico[set_u] = 1
                end
            end
        end
    end

    delete!(dico, full_set)
    delete!(dico, empty_set)

    nb_sets = sum(values(dico))

    for key in keys(dico)
        dico[key] = dico[key] / nb_sets
    end

    return dico

end

function sample_U(dico :: Dict{Set, Float64}, n :: Int)
    return StatsBase.sample(collect(keys(dico)), StatsBase.pweights(collect(values(dico))), n)
end
