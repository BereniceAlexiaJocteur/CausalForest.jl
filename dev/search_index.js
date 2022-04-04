var documenterSearchIndex = {"docs":
[{"location":"index.html","page":"Introduction","title":"Introduction","text":"Modules = [CausalForest]","category":"page"},{"location":"index.html#CausalForest.apply_forest-Union{Tuple{S}, Tuple{CausalForest.EnsembleCausal{S}, AbstractMatrix{S}}} where S","page":"Introduction","title":"CausalForest.apply_forest","text":"Get the causal effect for each row in x given a causal forest\n\n\n\n\n\n","category":"method"},{"location":"index.html#CausalForest.build_forest-Union{Tuple{T}, Tuple{S}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool, Any}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool, Any, Any}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool, Any, Any, Any}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool, Any, Any, Any, Any}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool, Any, Any, Any, Any, Any}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool, Any, Any, Any, Any, Any, Any}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool, Any, Any, Any, Any, Any, Any, Any}, Tuple{Bool, Bool, Bool, AbstractVector{T}, AbstractVector{Int64}, AbstractMatrix{S}, Bool, Any, Any, Any, Any, Any, Any, Any, Any}} where {S, T<:Float64}","page":"Introduction","title":"CausalForest.build_forest","text":"Build a causal forest.\n\nif centering=True Y and W are centered else they stay unchanged\nif bootstrap=True we sample for each tree via bootstrap else we use subsampling\nif honest=True we use 2 samples one too build splits and the other one to fill leaves   otherwise we use the whole sample for the two steps\n\n-if constmtry=True we use a constant m try otherwise we use a random mtry following     min(max(Poisson(mpois),1),numberoffeatures)\n\n\n\n\n\n","category":"method"}]
}
