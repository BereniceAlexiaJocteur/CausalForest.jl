# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of CausalForest.jl is released under the MIT license.

module treecausation_2
    include("../util.jl")

    import Random
    import PoissonRandom # to generate random mtry
    import StatsBase # to sample features
    import Optim
    export fit

    mutable struct NodeMeta{S}
        l           :: NodeMeta{S}  # right child
        r           :: NodeMeta{S}  # left child
        old_purity  :: Float64
        new_purity  :: Float64
        feature     :: Int          # feature used for splitting
        threshold   :: S            # threshold value
        is_leaf     :: Bool
        depth       :: Int
        region      :: UnitRange{Int} # a slice of the samples used to decide the split of the node
        features    :: Vector{Int}    # a list of features
        split_at    :: Int            # index of samples
        function NodeMeta{S}(
                features :: Vector{Int},
                region   :: UnitRange{Int},
                depth    :: Int,
                pur      :: Float64) where S
            node = new{S}()
            node.depth = depth
            node.old_purity = pur
            node.region = region
            node.features = features
            node.is_leaf = false
            node
        end
    end

    struct Tree{S}
        root   :: NodeMeta{S}
        inds   :: Vector{Int}
    end

    # find an optimal split that satisfy the given constraints
    # (max_depth, min_samples_split)
    function _split!(
            X                   :: AbstractMatrix{S}, # the feature array
            Y                   :: AbstractVector{Float64}, # the label array
            W                   :: AbstractVector{Int}, # the treatment array
            node                :: NodeMeta{S}, # the node to split
            const_mtry          :: Bool,
            m_pois              :: Int, # hyperparameter for poisson for random mtry
            max_depth           :: Int, # the maximum depth of the resultant tree
            min_samples_leaf    :: Int, # the minimum number of samples each leaf needs to have
            min_samples_split   :: Int, # the minimum number of samples in needed for a split
            indX                :: AbstractVector{Int}, # an array of sample indices,
                                                # we split using samples in indX[node.region]
            # the three arrays below are given for optimization purposes
            Xf                  :: AbstractVector{S},
            Yf                  :: AbstractVector{Float64},
            Wf                  :: AbstractVector{Int},
            rng                 :: Random.AbstractRNG) where {S}

        region = node.region
        n_samples = length(region)
        r_start = region.start - 1

        @inbounds @simd for i in 1:n_samples
            Yf[i] = Y[indX[i + r_start]]
            Wf[i] = W[indX[i + r_start]]
        end

        if (min_samples_leaf * 2 >  n_samples
         || min_samples_split    >  n_samples
         || max_depth            <= node.depth
         )
            node.is_leaf = true
            return
        end

        features = node.features
        n_features = length(features)
        best_purity = typemax(Float64)
        best_feature = -1
        threshold_lo = X[1]
        threshold_hi = X[1]

        # true if every feature is constant
        unsplittable = true
        total_features = size(X, 2)
        if const_mtry
            mtry = m_pois
        else
            mtry = min(max(PoissonRandom.pois_rand(m_pois), 1), total_features)
        end
        random_features = StatsBase.sample(1:total_features, mtry, replace=false)
        for feature in random_features

            @simd for i in 1:n_samples
                Xf[i] = X[indX[i + r_start], feature]
            end

            # sort Yf and indX by Xf
            util.q_bi_sort!(Xf, indX, 1, n_samples, r_start)
            @simd for i in 1:n_samples
                Yf[i] = Y[indX[i + r_start]]
                Wf[i] = W[indX[i + r_start]]
            end
            # lo and hi are the indices of
            # the least upper bound and the greatest lower bound
            # of the left and right nodes respectively

            hi = 0
            last_f = Xf[1]
            while hi < n_samples
                lo = hi + 1
                curr_f = Xf[lo]
                hi = (lo < n_samples && curr_f == Xf[lo+1]
                    ? searchsortedlast(Xf, curr_f, lo, n_samples, Base.Order.Forward)
                    : lo)

                if lo-1 >= min_samples_leaf && n_samples - (lo-1) >= min_samples_leaf
                    unsplittable = false

                    function f_l(par)
                        res = 0
                        @simd for i in (hi+1):n_samples
                            res += (Yf[i] - par[1]*Wf[i] - par[2])^2
                        end
                        return res/(n_samples-hi)
                    end

                    function g_l!(G, par)
                        G[1] = 0
                        G[2] = 0
                        @simd for i in (hi+1):n_samples
                            G[1] += 2*(Yf[i] - par[1]*Wf[i] - par[2])*Wf[i]
                            G[2] += 2*(Yf[i] - par[1]*Wf[i] - par[2])
                        end
                        G[1] = G[1]/(n_samples-hi)
                        G[2] = G[2]/(n_samples-hi)
                    end

                    function f_r(par)
                        res = 0
                        @simd for i in 1:hi
                            res += (Yf[i] - par[1]*Wf[i] - par[2])^2
                        end
                        return res/hi
                    end

                    function g_r!(G, par)
                        G[1] = 0
                        G[2] = 0
                        @simd for i in 1:hi
                            G[1] += 2*(Yf[i] - par[1]*Wf[i] - par[2])*Wf[i]
                            G[2] += 2*(Yf[i] - par[1]*Wf[i] - par[2])
                        end
                        G[1] = G[1]/hi
                        G[2] = G[2]/hi
                    end

                    opti_l = Optim.optimize(f_l, g_l!, zeros(2), Optim.GradientDescent())
                    opti_r = Optim.optimize(f_r, g_r!, zeros(2), Optim.GradientDescent())

                    minimum_l = Optim.minimum(opti_l)
                    minimum_r = Optim.minimum(opti_r)

                    purity = minimum_l + minimum_r
                    if purity < best_purity && !isapprox(purity, best_purity)
                        # will take average at the end, if possible
                        threshold_lo = last_f
                        threshold_hi = curr_f
                        best_purity  = purity
                        best_feature = feature
                        println(best_purity, best_feature)
                    end
                end

                last_f = curr_f
            end
            print("featureok")
        end
        print("splitok")

        # no splits honor min_samples_leaf
        node.new_purity = best_purity
        @inbounds if (unsplittable
                || best_feature == -1) # in case best pur = inf
            node.is_leaf = true
            return
        else

            @simd for i in 1:n_samples
                Xf[i] = X[indX[i + r_start], best_feature]
            end

            try
                node.threshold = (threshold_lo + threshold_hi) / 2.0
            catch
                node.threshold = threshold_hi
            end
            # split the samples into two parts: ones that are greater than
            # the threshold and ones that are less than or equal to the threshold
            #                                 ---------------------
            # (so we partition at threshold_lo instead of node.threshold)
            node.split_at = util.partition!(indX, Xf, threshold_lo, region)
            node.feature = best_feature
            node.features = features
        end

    end

    @inline function fork!(node :: NodeMeta{S}) where S
        ind = node.split_at
        region = node.region
        features = node.features
        # no need to copy because we will copy at the end
        node.l = NodeMeta{S}(features, region[    1:ind], node.depth + 1, node.new_purity)
        node.r = NodeMeta{S}(features, region[ind+1:end], node.depth + 1, node.new_purity)
    end

    function _fit(
            X                     :: AbstractMatrix{S},
            Y                     :: AbstractVector{Float64},
            W                     :: AbstractVector{Int},
            indX                  :: AbstractVector{Int},
            const_mtry            :: Bool,
            m_pois                :: Int,
            max_depth             :: Int,
            min_samples_leaf      :: Int,
            min_samples_split     :: Int,
            rng=Random.GLOBAL_RNG :: Random.AbstractRNG) where {S}

        n_samples, n_features = size(X)

        Yf  = Array{Float64}(undef, n_samples)
        Xf  = Array{S}(undef, n_samples)
        Wf  = Array{Int}(undef, n_samples)

        root = NodeMeta{S}(collect(1:n_features), 1:length(indX), 0, Inf)
        stack = NodeMeta{S}[root]

        @inbounds while length(stack) > 0
            print("ok")
            node = pop!(stack)
            _split!(
                X, Y, W,
                node,
                const_mtry,
                m_pois,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                indX,
                Xf, Yf, Wf,
                rng)
            if !node.is_leaf
                fork!(node)
                push!(stack, node.r)
                push!(stack, node.l)
            end
        end
        return (root, indX)
    end

    function fit(;
            X                     :: AbstractMatrix{S},
            Y                     :: AbstractVector{Float64},
            W                     :: AbstractVector{Int},
            indX                  :: AbstractVector{Int},
            const_mtry            :: Bool,
            m_pois                :: Int,
            max_depth             :: Int,
            min_samples_leaf      :: Int,
            min_samples_split     :: Int,
            rng=Random.GLOBAL_RNG :: Random.AbstractRNG) where {S}

        n_samples, n_features = size(X)

        util.check_input(
            X,
            Y,
            W,
            indX,
            const_mtry,
            m_pois,
            max_depth,
            min_samples_leaf,
            min_samples_split)

        root, indX = _fit(
            X,
            Y,
            W,
            indX,
            const_mtry,
            m_pois,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            rng)

        return Tree{S}(root, indX)

    end
end
