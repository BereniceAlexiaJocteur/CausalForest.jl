using CausalForest
using Test

@testset "oob classification random forest test" begin

    features, labels = load_data("iris")
    features = float.(features)
    labels   = string.(labels)
    model = build_forest_oob(labels, features, 2, 10, 0.5, 6)
    predictions = apply_forest_oob(model)
    comparison = predictions .== labels
    len = length(labels)
    @test (sum(comparison)/len) > 0.9

end
