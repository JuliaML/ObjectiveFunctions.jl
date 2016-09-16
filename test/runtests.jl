using ObjectiveFunctions
using Base.Test

@testset "RegObj" begin
    let nin=2, nout=3
        t = Affine(nin, nout)
        l = L2DistLoss()
        λ = 1e-3
        p = L2Penalty(λ)
        obj = objective(t, l, p)
        @show obj typeof(obj)

        @test output_value(t) === input_value(obj.loss)
        @test input_length(obj.loss) == nout
        @test output_length(obj.loss) == 1

        input = rand(nin)
        target = rand(nout)
        w, b = t.params.views

        # test the forward pass
        transform!(obj, target, input)
        @test input_value(obj) == input
        @test output_value(t) ≈ w * input + b
        total_loss = sum(value(l, target[i], output_value(t)[i]) for i=1:nout)
        total_penalty = 0.5λ * sum(θᵢ^2 for θᵢ in params(t))
        @show total_loss total_penalty
        @test output_value(obj)[1] ≈ total_loss + total_penalty

        # test the backward pass
        grad!(obj)
        x = input_value(obj.loss)
        ∇ = input_grad(obj.loss)
        @show ∇
        for i=1:nout
            @test ∇[i] == deriv(obj.loss.loss, target[i], x[i])
        end
        ∇input = input_grad(obj)
        @show ∇input
        for i=1:nin
            # TODO a real test
            @test ∇input[i] != 0
        end
    end
end
