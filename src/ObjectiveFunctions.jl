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

# this assumes that input_value(lt) is pre-populated... copy the target in
function transform!(lt::LossTransform, target::AbstractVector)
    copy!(value(lt.target), target)
    transform!(lt)
end

function transform!(lt::LossTransform)
    lt.output.val[1] = sumvalue(lt.loss, value(lt.target), input_value(lt))
    lt
end

# TODO: is this right??  what is Losses.deriv?
function grad!(lt::LossTransform)
    deriv!(grad(lt.input), lt.loss, value(lt.target), value(lt.input))
end


# ------------------------------------------------------------------------

type RegularizedObjective{T<:Transformation, L<:LossTransform, P<:Penalty} <: Transformation
    transformation::T
    loss::L
    penalty::P
end

function transform!(obj::RegularizedObjective, target::AbstractVector, input::AbstractVector)
end

# we don't need data args because they were given to transform... check this!
function grad!(obj::RegularizedObjective)
end

# TODO:
#   - build LossTransform during construction
#   - link transformation to loss
#   - transform! should update the transformation, then compute and return the loss (without penalty?)
#   - grad! should compute a loss derivative, then call grad!(trans, dl, penalty)





end # module
