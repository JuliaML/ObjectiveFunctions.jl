__precompile__(true)

module ObjectiveFunctions

using Compat: @compat
using Reexport
@reexport using LearnBase
@reexport using LossFunctions
@reexport using Transformations
@reexport using PenaltyFunctions

import LearnBase: transform!, grad, grad!, params, update!
import Transformations: input_node, output_node, totalcost, InputNode, SumNode, OutputNode

export
    AbstractLossTransform,
    NoLoss,
    LossTransform,
    CrossEntropy,
    RegularizedObjective,
    objective

@compat abstract type AbstractLossTransform{T} <: Transformation end

immutable NoLoss <: AbstractLossTransform{Void} end

# this assumes that input_value(lt) is pre-populated... copy the target in, then pass to no-arg version
function transform!(lt::AbstractLossTransform, target::AbstractVector)
    copy!(value(lt.target), target)
    transform!(input_node(lt))
    transform!(lt)
end

# ------------------------------------------------------------------------

# this is a transformation which stores the calculated loss, as well as the gradients of the inputs
# it can be linked to a transformation to allow easier forward/backward calcs
immutable LossTransform{T,L<:Loss} <: AbstractLossTransform{T}
    loss::L
    nin::Int
    input::SumNode{T,1}
    target::SumNode{T,1}
    output::OutputNode{T,1}

    function (::Type{LossTransform{T, L}}){T, L <: Loss}(loss::Loss, nin::Int)
        input = InputNode(T, nin)
        target = InputNode(T, nin)
        output = OutputNode(T, 1)
        grad(output)[1] = one(T)  # ∂L/∂L == 1
        new{T, L}(loss, nin, input, target, output)
    end
end

LossTransform{L<:Loss}(loss::L, nin::Int) = LossTransform{Float64, L}(loss, nin)

# input and target are pre-populated... compute the output value as: ∑ loss(targetᵢ, inputᵢ)
function transform!(lt::LossTransform)
    lt.output.val[1] = value(lt.loss, value(lt.target), input_value(lt), AvgMode.Sum())
    lt
end

# TODO: is this right??  what is LossFunctions.deriv?
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
    input::SumNode{T,1}
    target::SumNode{T,1}
    output::OutputNode{T,1}

    function (::Type{CrossEntropy{T}}){T}(n::Int)
        input = InputNode(T, n)
        target = InputNode(T, n)
        output = OutputNode(T, 1)
        grad(output)[:] = one(T)  # ∂L/∂L == 1
        new{T}(n, input, target, output)
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
    # link_nodes!(output_node(transformation), input_node(lt))
    link_nodes!(transformation, lt)
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

function objective(t::Transformation, nl::NoLoss, penalty::Penalty = NoPenalty())
    RegularizedObjective(t,nl,penalty)
end


input_node(obj::RegularizedObjective) = input_node(obj.transformation)
output_node(obj::RegularizedObjective) = output_node(isa(obj.loss, NoLoss) ? obj.transformation : obj.loss)

params(obj::RegularizedObjective) = params(obj.transformation)
grad(obj::RegularizedObjective) = grad(obj.transformation)
totalcost(obj::RegularizedObjective) = norm(value(output_node(obj)))

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

# function apply_penalty(penalty::Penalty, θ, ∇)
#     for (i,j) in zip(eachindex(θ), eachindex(∇))
#         ∇[j] += deriv(penalty, θ[i])
#     end
#     ∇
# end

# we don't need data args because they were given or computed in transform (check this fact!)
function grad!(obj::RegularizedObjective)
    grad!(obj.loss)
    grad!(output_node(obj.transformation))
    grad!(obj.transformation)
    addgrad!(grad(obj), obj.penalty, params(obj))
end

# # handle the no-data case... probably just minimizing a function
# function update!(obj::RegularizedObjective, ::Void)
#     transform!(obj.transformation)
#
#     # output grad is ones, backwards pass
#     output_grad(obj.transformation)[:] = 1
#     grad!(obj.transformation)
#
#     apply_penalty(obj.penalty, params(obj), grad(obj))
# end

# ------------------------------------------------------------------------

end # module
