using CausalForest
using Test

@testset "oob regression random forest test" begin

    n, m = 10^3, 5
    features = randn(n, m)
    weights = rand(-2:2, m)
    labels = features * weights
    model = build_forest_oob(labels, features, -1, 100)
    predictions = apply_forest_oob(model)
    comparison = isapprox.(labels, predictions, atol=2.5)
    @test (sum(comparison)/n) > 0.8

end
