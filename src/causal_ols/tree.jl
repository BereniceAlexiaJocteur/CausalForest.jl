# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of CausalForest.jl is released under the MIT license.

module treecausation_ols
    include("../util.jl")

    import Random
    import PoissonRandom # to generate random mtry
    import StatsBase # to sample features
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

        t_w_sum = zero(Int) # sum of W in node
        t_y_sum = zero(Float64) # sum of Y in node
        t_w_ssq = zero(Int) # sum of W squares in node
        t_wy_sum = zero(Float64) # sum of WY in node
        @inbounds @simd for i in 1:n_samples
            t_w_sum += Wf[i]
            t_y_sum += Yf[i]
            t_w_ssq += Wf[i]*Wf[i]
            t_wy_sum += Wf[i]*Yf[i]
        end


        if (min_samples_leaf * 2 >  n_samples
         || min_samples_split    >  n_samples
         || max_depth            <= node.depth
         #TODO|| min_samples_split    >  t_w_sum
         #TODO|| min_samples_split    >  n_samples - t_w_sum
         )
            node.is_leaf = true
            return
        end

        features = node.features
        n_features = length(features)
        best_purity = typemin(Float64)
        best_feature = -1
        threshold_lo = X[1]
        threshold_hi = X[1]
        #min_child_size = (t_w_ssq - t_w_sum*t_w_sum/n_samples)*0.05

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

            r_w_sum = t_w_sum
            l_w_sum = zero(Int)
            r_y_sum = t_y_sum
            l_y_sum = zero(Float64)
            r_w_ssq = t_w_ssq
            l_w_ssq = zero(Int)
            r_wy_sum = t_wy_sum
            l_wy_sum = zero(Float64)

            @simd for i in 1:n_samples
                Xf[i] = X[indX[i + r_start], feature]
            end

            # sort Yf and indX by Xf
            util.q_bi_sort!(Xf, indX, 1, n_samples, r_start)
            @simd for i in 1:n_samples
                Yf[i] = Y[indX[i + r_start]]
                Wf[i] = W[indX[i + r_start]]
            end
            nl, nr = 0, n_samples
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

                if l_w_sum >= min_samples_leaf && lo-1-l_w_sum >= min_samples_leaf && r_w_sum >= min_samples_leaf && n_samples - (lo-1)-r_w_sum >= min_samples_leaf
                    unsplittable = false
                    difference = l_wy_sum/l_w_sum + (r_y_sum-r_wy_sum)/(nr-r_w_sum) - (l_y_sum-l_wy_sum)/(nl-l_w_sum) - r_wy_sum/r_w_sum
                    purity = (nl*nr)/(n_samples*n_samples)*difference*difference
                    if purity > best_purity && !isapprox(purity, best_purity)
                        # will take average at the end, if possible
                        threshold_lo = last_f
                        threshold_hi = curr_f
                        best_purity  = purity
                        best_feature = feature
                    end
                end

                # update values
                # that would require the smaller number of iterations
                if (hi << 1) < n_samples + lo # i.e., hi - lo < n_samples - hi
                    @simd for i in lo:hi
                        nr   -= 1
                        r_w_sum -= Wf[i]
                        r_y_sum -= Yf[i]
                        r_w_ssq -= Wf[i]*Wf[i]
                        r_wy_sum -= Wf[i]*Yf[i]
                    end
                else
                    nr = r_w_sum = r_w_ssq = 0
                    r_y_sum = r_wy_sum = zero(Float64)
                    @simd for i in (hi+1):n_samples
                        nr   += 1
                        r_w_sum += Wf[i]
                        r_y_sum += Yf[i]
                        r_w_ssq += Wf[i]*Wf[i]
                        r_wy_sum += Wf[i]*Yf[i]
                    end
                end
                nl   = n_samples - nr
                l_w_sum = t_w_sum - r_w_sum
                l_y_sum = t_y_sum - r_y_sum
                l_w_ssq = t_w_ssq - r_w_ssq
                l_wy_sum = t_wy_sum - r_wy_sum

                last_f = curr_f
            end

        end

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
