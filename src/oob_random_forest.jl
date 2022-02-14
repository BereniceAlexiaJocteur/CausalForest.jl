const LeafOrNode{S, T} = Union{Leaf{T}, Node{S, T}}

struct TreeOOB{S, T}
    tree     :: LeafOrNode{S, T}
    inds     :: AbstractVector{Int}
    oobfeat  :: AbstractMatrix{S}
end

struct EnsembleOOB{S, T}
    nsamp    :: Int
    features :: AbstractMatrix{S}
    trees    :: Vector{TreeOOB{S, T}}
end

mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)
