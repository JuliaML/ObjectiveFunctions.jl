using ObjectiveFunctions
using Base.Test

let nin=2, nout=3
    t = Affine{Float64}(nin, nout)
    l = L2DistLoss()
    λ = 1e-3
    p = L2Penalty(λ)
    obj = RegularizedObjective(t, l, p)
    @show obj typeof(obj)

    @test output_value(t) === input_value(obj.loss)
    @test input_length(obj.loss) == nout
    @test output_length(obj.loss) == 1

    input = rand(nin)
    target = rand(nout)

    # test the forward pass
    transform!(obj, target, input)
    @test input_value(obj) == input
    @test output_value(t) ≈ value(t.w) * input + value(t.b)
    total_loss = sum(value(l, target[i], output_value(t)[i]) for i=1:nout)
    total_penalty = 0.5λ * sum(θᵢ^2 for θᵢ in params(t))
    @show total_loss total_penalty
    @test output_value(obj)[1] ≈ total_loss + total_penalty

    # TODO: test the backward pass
end
