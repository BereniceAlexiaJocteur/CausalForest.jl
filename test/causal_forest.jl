using CausalForest
using Test

@testset "causal forest non centered test" begin

    indices,X,T,Y = load_data("causal")
    Xtest = rand(Float64, (1000, size(X, 2)))
    Xtest[:, 1] =  LinRange(0, 1, 1000)
    true_effect = Xtest[:, 1].>0.5

    cf1 = build_forest(false, false, false, Y, T, X, -1, 100)
    pred1 = apply_forest(cf1, Xtest)
    comparison1 = isapprox.(true_effect, pred1, atol=0.3)
    @test (sum(comparison1)/1000) > 0.7

    cf2 = build_forest(false, false, true, Y, T, X, -1, 100)
    pred2 = apply_forest(cf2, Xtest)
    comparison2 = isapprox.(true_effect, pred2, atol=0.3)
    @test (sum(comparison2)/1000) > 0.7

    cf3 = build_forest(false, true, false, Y, T, X, -1, 100)
    pred3 = apply_forest(cf3, Xtest)
    comparison3 = isapprox.(true_effect, pred3, atol=0.3)
    @test (sum(comparison3)/1000) > 0.7

    cf4 = build_forest(false, true, true, Y, T, X, -1, 100)
    pred4 = apply_forest(cf4, Xtest)
    comparison4 = isapprox.(true_effect, pred4, atol=0.3)
    @test (sum(comparison4)/1000) > 0.7

end

@testset "causal forest centered test" begin

    indices,X,T,Y = load_data("causal")
    Xtest = rand(Float64, (1000, size(X, 2)))
    Xtest[:, 1] =  LinRange(0, 1, 1000)
    true_effect = Xtest[:, 1].>0.5

    cf1 = build_forest(true, false, false, Y, T, X, -1, 100)
    pred1 = apply_forest(cf1, Xtest)
    comparison1 = isapprox.(true_effect, pred1, atol=0.4)
    @test (sum(comparison1)/1000) > 0.6

    cf2 = build_forest(true, false, true, Y, T, X, -1, 100)
    pred2 = apply_forest(cf2, Xtest)
    comparison2 = isapprox.(true_effect, pred2, atol=0.3)
    @test (sum(comparison2)/1000) > 0.7

    cf3 = build_forest(true, true, false, Y, T, X, -1, 100)
    pred3 = apply_forest(cf3, Xtest)
    comparison3 = isapprox.(true_effect, pred3, atol=0.3)
    @test (sum(comparison3)/1000) > 0.7

    cf4 = build_forest(true, true, true, Y, T, X, -1, 100)
    pred4 = apply_forest(cf4, Xtest)
    comparison4 = isapprox.(true_effect, pred4, atol=0.3)
    @test (sum(comparison4)/1000) > 0.7

end
