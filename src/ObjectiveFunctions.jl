__precompile__(true)

module ObjectiveFunctions

using Reexport
@reexport using LearnBase
@reexport using Losses
@reexport using Transformations
@reexport using Penalties

import LearnBase: transform!, grad, grad!
import Transformations: input_node, output_node, params

export
    LossTransform,
    RegularizedObjective

# ------------------------------------------------------------------------

# this is a transformation which stores the calculated loss, as well as the gradients of the inputs
# it can be linked to a transformation to allow easier forward/backward calcs
immutable LossTransform{T,L<:Loss} <: Transformation
    loss::L
    nin::Int
    input::Node{:input,T,1}
    target::Node{:target,T,1}
    output::Node{:output,T,1}

    function LossTransform(loss::Loss, nin::Int)
        input = Node(:input, zeros(T, nin))
        target = Node(:target, zeros(T, nin))
        output = Node(:output, zeros(T, 1))
        grad(output)[1] = one(T)  # ∂L/∂L == 1
        new(loss, nin, input, target, output)
    end
end

# this assumes that input_value(lt) is pre-populated... copy the target in, then pass to no-arg version
function transform!(lt::LossTransform, target::AbstractVector)
    copy!(value(lt.target), target)
    transform!(lt)
end

# input and target are pre-populated... compute the output value as: ∑ loss(targetᵢ, inputᵢ)
function transform!(lt::LossTransform)
    lt.output.val[1] = sumvalue(lt.loss, value(lt.target), input_value(lt))
    lt
end

# TODO: is this right??  what is Losses.deriv?
function grad!(lt::LossTransform)
    # update the input gradient using the values of target and input
    deriv!(grad(lt.input), lt.loss, value(lt.target), value(lt.input))
end


# ------------------------------------------------------------------------

# TODO: should we support penalties on outputs as well as parameters?

# convenience type for Empirical Risk Minimization and similar
type RegularizedObjective{T<:Transformation, L<:LossTransform, P<:Penalty} <: Minimizable
    transformation::T
    loss::L
    penalty::P
end

function RegularizedObjective(transformation::Transformation, loss::Loss, penalty::Penalty = NoPenalty())
    # create a LossTransform and link it to the transformation output
    lt = LossTransform{Float64,typeof(loss)}(loss, output_length(transformation))
    link_nodes!(output_node(transformation), input_node(lt))

    # return the constructed object
    RegularizedObjective(transformation, lt, penalty)
end

input_node(obj::RegularizedObjective) = input_node(obj.transformation)
output_node(obj::RegularizedObjective) = output_node(obj.loss)

params(obj::RegularizedObjective) = params(obj.transformation)
grad(obj::RegularizedObjective) = grad(obj.transformation)

function transform!(obj::RegularizedObjective, target::AbstractVector, input::AbstractVector)
    # forward pass through the transformation... assuming output has been linked to loss already
    transform!(obj.transformation, input)

    # loss input should be populated already... forward pass with target
    transform!(obj.loss, target)

    # add the penalty
    total_penalty = value(obj.penalty, params(obj.transformation))
    output_value(obj.loss)[1] += total_penalty
end

# we don't need data args because they were given or computed in transform (check this fact!)
function grad!(obj::RegularizedObjective)
    grad!(obj.loss)
    grad!(obj.transformation)

    θ = params(obj.transformation)
    ∇ = grad(obj.transformation)
    for (i,j) in zip(eachindex(θ), eachindex(∇))
        ∇[j] += deriv(obj.penalty, θ[i])
    end
    ∇
end

# ------------------------------------------------------------------------

end # module
