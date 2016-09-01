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

    function LossTransform(loss::Loss, nin::Int)
        input = Node(:input, zeros(T, nin))
        new(loss, nin, input)
    end
end

# TODO:
# transform!(loss::LossTransform, args...) = ???
# grad!() = ???

# ------------------------------------------------------------------------

type RegularizedObjective{T<:Transformation, L<:LossTransform, P<:Penalty}
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
