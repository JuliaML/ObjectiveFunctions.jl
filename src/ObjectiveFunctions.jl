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
    AbstractLossTransform,
    LossTransform,
    CrossEntropy,
    RegularizedObjective,
    objective,
    totalcost

abstract AbstractLossTransform{T} <: Transformation

# this assumes that input_value(lt) is pre-populated... copy the target in, then pass to no-arg version
function transform!(lt::AbstractLossTransform, target::AbstractVector)
    copy!(value(lt.target), target)
    transform!(lt)
end

# ------------------------------------------------------------------------

# this is a transformation which stores the calculated loss, as well as the gradients of the inputs
# it can be linked to a transformation to allow easier forward/backward calcs
immutable LossTransform{T,L<:Loss} <: AbstractLossTransform{T}
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

# NOTE: special handling for the Softmax/CrossEntropyLoss layer.
#   Instead of a proper activation function layer, we do both the
#   softmax activation and cross entropy loss together.

immutable CrossEntropy{T} <: AbstractLossTransform{T}
    n::Int
    input::Node{:input,T,1}
    target::Node{:target,T,1}
    output::Node{:output,T,1}

    function CrossEntropy(n::Int)
        input = Node(:input, zeros(T, n))
        target = Node(:target, zeros(T, n))
        output = Node(:output, zeros(T, 1))
        grad(output)[:] = one(T)  # ∂L/∂L == 1
        new(n, input, target, output)
    end
end
CrossEntropy(n::Int) = CrossEntropy{Float64}(n)

function transform!{T}(ce::CrossEntropy{T})
    ce.output.val[1] = -sum(ce.target.val[i] * log(ce.input.val[i]) for i=1:ce.n)
end

# ** NOTE **: This is really solving for the gradient with respect to the softmax activation inputs,
#             but we diverge from correctness to allow for the simplified gradient calc.
function grad!{T}(ce::CrossEntropy{T})
    for i=1:ce.n
        ce.input.∇[i] = ce.input.val[i] - ce.target.val[i]
    end
end

# ------------------------------------------------------------------------

# TODO: should we support penalties on outputs as well as parameters?

# convenience type for Empirical Risk Minimization and similar
immutable RegularizedObjective{T<:Transformation, L<:AbstractLossTransform, P<:Penalty} <: Minimizable
    transformation::T
    loss::L
    penalty::P
end

# first link the transformation to the loss, then construct
function objective(transformation::Transformation, lt::AbstractLossTransform, penalty::Penalty = NoPenalty())
    link_nodes!(output_node(transformation), input_node(lt))
    RegularizedObjective(transformation, lt, penalty)
end

# first create the LossTransform to wrap the Loss, then construct
function objective(transformation::Transformation, loss::Loss, penalty::Penalty = NoPenalty())
    lt = LossTransform{Float64,typeof(loss)}(loss, output_length(transformation))
    objective(transformation, lt, penalty)
end

# convenience when creating chains... auto-choose loss given final layer
function objective(chain::Chain, penalty::Penalty = NoPenalty())
    T = typeof(chain[end])
    loss = if T <: Affine || T <: Activation{:identity}
        L2DistLoss()
    elseif T <: Activation{:logistic}
        CrossentropyLoss()
    elseif T <: Activation{:softmax}
        CrossEntropy(output_length(chain[end]))
    else
        error("Can't pick a default loss for $T... choose it explicitly.")
    end
    objective(chain, loss, penalty)
end


input_node(obj::RegularizedObjective) = input_node(obj.transformation)
output_node(obj::RegularizedObjective) = output_node(obj.loss)

params(obj::RegularizedObjective) = params(obj.transformation)
grad(obj::RegularizedObjective) = grad(obj.transformation)
totalcost(obj::RegularizedObjective) = output_value(obj.loss)[1]

function transform!(obj::RegularizedObjective, target::AbstractArray, input::AbstractArray)
    # forward pass through the transformation... assuming output has been linked to loss already
    transform!(obj.transformation, input)

    # loss input should be populated already... forward pass with target
    transform!(obj.loss, target)

    # add the penalty
    total_penalty = value(obj.penalty, params(obj.transformation))
    output_value(obj.loss)[1] += total_penalty
end

# if our target is a single value, lets convert it to a length-1 vector
function transform!{T}(obj::RegularizedObjective, target::T, input::AbstractArray{T})
    transform!(obj, [target], input)
end

# we don't need data args because they were given or computed in transform (check this fact!)
function grad!(obj::RegularizedObjective)
    grad!(obj.loss)
    grad!(obj.transformation)

    θ = params(obj)
    ∇ = grad(obj)
    for (i,j) in zip(eachindex(θ), eachindex(∇))
        ∇[j] += deriv(obj.penalty, θ[i])
    end
    ∇
end

# ------------------------------------------------------------------------

end # module
