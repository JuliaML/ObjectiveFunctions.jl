module ObjectiveFunctions

using Reexport
@reexport using LearnBase
@reexport using Losses
@reexport using Transformations
@reexport using Penalties

import LearnBase: transform!, grad!

# this is a transformation which stores the calculated loss, as well as the gradients of the inputs
# it can be linked to a transformation to allow easier forward/backward calcs
immutable LossTransform{L<:Loss,T} <: Transformation
    loss::L
    nin::Int
    input::Node{:input,T}
    target::Node{:target,T}
    output::Node{:output,T}

    function LossTransform(loss::Loss, nin::Int)
        input = Node(:input, zeros(T, nin))
        target = Node(:target, zeros(T, nin))
        output = Node(:output, zeros(T, 1))
        output_grad(output)[1] = one(T)  # ∂L/∂L == 1
        new(loss, nin, input, target, output)
    end
end

function transform!(lt::LossTransform, target::AbstractVector)
    lt.output.val[1] = value(lt.loss, target, input_value(input))
    # TODO
end


# TODO:
# transform!(loss::LossTransform, args...) = ???
# grad!() = ???

# ------------------------------------------------------------------------

type RegularizedObjective{T<:Transformation, L<:LossTransform, P<:Penalty} <: Transformation
    transformation::T
    loss::L
    penalty::P
end



# TODO:
#   - build LossTransform during construction
#   - link transformation to loss
#   - transform! should update the transformation, then compute and return the loss (without penalty?)
#   - grad! should compute a loss derivative, then call grad!(trans, dl, penalty)





end # module
