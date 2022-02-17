# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of CausalForest.jl is released under the MIT license.

# written by Poom Chiarawongse <eight1911@gmail.com>

module treecausation
    include("../util.jl")

    import Random
    import PoissonRandom # to generate random mtry
    import StatsBase # to sample features
    export fit

    mutable struct NodeMeta{S}
        l           :: NodeMeta{S}  # right child
        r           :: NodeMeta{S}  # left child
        #label       :: Float64      # most likely label
        feature     :: Int          # feature used for splitting
        threshold   :: S            # threshold value
        is_leaf     :: Bool
        depth       :: Int
        region      :: UnitRange{Int} # a slice of the samples used to decide the split of the node # TODO ca doit etre les indices d'origine et pas les indices après centrage
        features    :: Vector{Int}    # a list of features
        # TODO ajouter n_features pour pas recalculer à chaque fois ??
        split_at    :: Int            # index of samples
        function NodeMeta{S}(
                features :: Vector{Int},
                region   :: UnitRange{Int},
                depth    :: Int) where S
            node = new{S}()
            node.depth = depth
            node.region = region
            node.features = features
            node.is_leaf = false
            node
        end
    end

    struct Tree{S}
        root   :: NodeMeta{S}
        labels :: Vector{Int}
        # TODO à modifier surement
    end

    # find an optimal split that satisfy the given constraints
    # (max_depth, min_samples_split, min_purity_increase)
    function _split!(
            X                   :: AbstractMatrix{S}, # the feature array
            Y                   :: AbstractVector{Float64}, # the label array
            W                   :: AbstractVector{Int}, # the treatment array
            node                :: NodeMeta{S}, # the node to split
            m_pois              :: Int, # hyperparameter for poisson for random mtry
            max_depth           :: Int, # the maximum depth of the resultant tree
            min_samples_leaf    :: Int, # the minimum number of samples each leaf needs to have
            min_samples_split   :: Int, # the minimum number of samples in needed for a split
            min_purity_increase :: Float64, # minimum purity needed for a split TODO necessary ?
            indX                :: AbstractVector{Int}, # an array of sample indices,
                                                # we split using samples in indX[node.region]
            # the two arrays below are given for optimization purposes
            Xf                  :: AbstractVector{S},
            Yf                  :: AbstractVector{Float64},
            Wf                  :: AbstractVector{Int},
            rng                 :: Random.AbstractRNG) where {S}

        region = node.region
        n_samples = length(region)
        # TODO suppr r_start = region.start - 1
        r_start = region.start - 1

        @inbounds @simd for i in 1:n_samples
            Yf[i] = Y[indX[i + r_start]]
            Wf[i] = W[indX[i + r_start]]
            #Yf[i] = Y[indX[i]]
            #Wf[i] = W[indX[i]]
        end

        t_w_sum = zero(Int) # sum of W in node
        t_y_sum = zero(Float64) # sum of Y in node
        t_w_ssq = zero(Int) # sum of W squares in node
        t_wy_sum = zero(Float64) # sum of WY
        @inbounds @simd for i in 1:n_samples
            t_w_sum += Wf[i]
            t_y_sum += Yf[i]
            t_w_ssq += Wf[i]*Wf[i]
            t_wy_sum += Wf[i]*Yf[i]
        end

        # node.label =  tsum / wsum TODO not necessary on ne veut pas résumer info dans noeud (moyenne des Y ou majorité...)
        if (min_samples_leaf * 2 >  n_samples
         || min_samples_split    >  n_samples
         || max_depth            <= node.depth
         # TODO|| tsum * node.label    > -1e-7 * wsum + tssq # equivalent to old_purity > -1e-7    + considere critère traitement
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

        # TODO useless now indf = 1
        # the number of new constants found during this split
        # TODO on regarde plus ca     n_constant = 0
        # true if every feature is constant
        unsplittable = true
        # the number of non constant features we will see if
        # only sample n_features used features
        # is a hypergeometric random variable
        total_features = size(X, 2)

        # this is the total number of features that we expect to not
        # be one of the known constant features. since we know exactly
        # what the non constant features are, we can sample at 'non_constants_used'
        # non constant features instead of going through every feature randomly.
        #TODO on ne regarde plus ca non_constants_used = util.hypergeometric(n_features, total_features-n_features, max_features, rng)
        #@inbounds while (unsplittable || indf <= non_constants_used) && indf <= n_features
        #@inbounds while (unsplittable || indf <= non_constants_used) && indf <= n_features # TODO modifier cond du while
            #feature = let   # TODO verifier ce bloc poisson pour mtry ???? ou c'est au dessus ??
                #indr = rand(rng, indf:n_features)
                #features[indf], features[indr] = features[indr], features[indf]
                #features[indf]
            #end
        mtry = min(max(PoissonRandom.pois_rand(m_pois), 1), total_features)
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

            #@simd for i in 1:n_samples
            for i in 1:n_samples
                Xf[i] = X[indX[i + r_start], feature]
                #Xf[i] = X[indX[i], feature]
            end

            # sort Yf and indX by Xf
            util.q_bi_sort!(Xf, indX, 1, n_samples, r_start)
            #util.q_bi_sort!(Xf, indX, 1, n_samples, 0) # TODO verif
            # @simd for i in 1:n_samples
            for i in 1:n_samples
                Yf[i] = Y[indX[i + r_start]]
                Wf[i] = W[indX[i + r_start]]
                #Yf[i] = Y[indX[i]]
                #Wf[i] = W[indX[i]]
            end
            nl, nr = 0, n_samples
            # lo and hi are the indices of
            # the least upper bound and the greatest lower bound
            # of the left and right nodes respectively
            hi = 0
            last_f = Xf[1] # TODO encore utile ??
            # TODO osef now is_constant = true
            while hi < n_samples
                lo = hi + 1
                curr_f = Xf[lo]
                hi = (lo < n_samples && curr_f == Xf[lo+1]
                    ? searchsortedlast(Xf, curr_f, lo, n_samples, Base.Order.Forward)
                    : lo)

                # TODO osef now ? (lo != 1) && (is_constant = false)
                # honor min_samples_leaf
                # TODO ajouter condition nombre de trt
                if lo-1 >= min_samples_leaf && n_samples - (lo-1) >= min_samples_leaf # TODO ajouter condition sur nb treat
                    unsplittable = false
                    difference = (l_w_ssq/nl - (l_w_sum/nl)^2)/(l_wy_sum/nl - (l_w_sum/nl))-(r_w_ssq/nr - (r_w_sum/nr)^2)/(r_wy_sum/nr - (r_w_sum/nr))
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

            # keep track of constant features to be used later.
            # TODO on considere plus cela if is_constant
                #n_constant += 1
                #features[indf], features[n_constant] = features[n_constant], features[indf]
            #end

            # TODO useless now indf += 1
        end

        # no splits honor min_samples_leaf
        @inbounds if (unsplittable
                # TODO on s'en occupe plus ?? || best_purity - tsum * node.label < min_purity_increase * wsum)
                )
            node.is_leaf = true
            return
        else
            # new_purity - old_purity < stop.min_purity_increase
            @simd for i in 1:n_samples
                Xf[i] = X[indX[i + r_start], best_feature] # TODO
                #Xf[i] = X[indX[i], best_feature]
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
            #node.split_at = util.partitionv2!(indX, Xf, threshold_lo, region) # TODO corriger cette fonction -> elle est peut etre plus necessaire ?
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
        node.l = NodeMeta{S}(features, region[    1:ind], node.depth + 1)
        node.r = NodeMeta{S}(features, region[ind+1:end], node.depth + 1)
    end

    function _fit(
            X                     :: AbstractMatrix{S},
            Y                     :: AbstractVector{Float64},
            W                     :: AbstractVector{Int},
            indX                  :: AbstractVector{Int},
            m_pois                :: Int,
            max_depth             :: Int,
            min_samples_leaf      :: Int,
            min_samples_split     :: Int,
            min_purity_increase   :: Float64,
            rng=Random.GLOBAL_RNG :: Random.AbstractRNG) where {S}

        n_samples, n_features = size(X)

        Yf  = Array{Float64}(undef, n_samples)
        Xf  = Array{S}(undef, n_samples)
        Wf  = Array{Int}(undef, n_samples)

        #indX = collect(1:n_samples) # TODO modifier car on veut les indices avant centrages -> mettre indX en parametre de la function
        root = NodeMeta{S}(collect(1:n_features), 1:n_samples, 0) # TODO on modif region
        #root = NodeMeta{S}(collect(1:n_features), indX, 0)
        stack = NodeMeta{S}[root]

        @inbounds while length(stack) > 0
            node = pop!(stack)
            _split!(
                X, Y, W,
                node,
                m_pois,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
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
            m_pois                :: Int,
            max_depth             :: Int,
            min_samples_leaf      :: Int,
            min_samples_split     :: Int,
            min_purity_increase   :: Float64,
            rng=Random.GLOBAL_RNG :: Random.AbstractRNG) where {S}

        n_samples, n_features = size(X)

        #util.check_input( # TODO modif cette fonction et son nom
        #    X,
        #    Y,
        #    W,
        #    indX,
        #    m_pois,
        #    max_depth,
        #    min_samples_leaf,
        #    min_samples_split,
        #    min_purity_increase)

        root, indX = _fit(
            X,
            Y,
            W,
            indX,
            m_pois,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            rng)

        #return Tree{S}(root, indX) # TODO on retourne plus indX et faut modifier la struct tree
        return (root, indX)

    end
end
