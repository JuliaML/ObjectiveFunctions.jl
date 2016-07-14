# Now each penalty just needs to define grad!(x,::Penalty) and prox!(x,::Penalty)
grad(x::AbstractArray, r::Penalty) = grad!(copy(x),r)
prox{T}(x::AbstractArray{T}, r::Penalty, ρ) = prox!(copy(x),r,ρ)
prox{T}(x::AbstractArray{T}, r::Penalty) = prox!(copy(x),r)

# Both grad(penalty, x) and grad(x, penalty) are valid?
grad(r::Penalty, x::AbstractArray) = grad(x,r)
prox(r::Penalty, x::AbstractArray, ρ) = prox(x,r,ρ)
prox(r::Penalty, x::AbstractArray) = prox(x,r)

# Null penalty type
immutable NoPenalty <: Penalty end
grad!{T}(x::AbstractArray{T},r::NoPenalty) = fill!(x,zero(T))
prox!(x::AbstractArray,r::NoPenalty) = x
prox!(x::AbstractArray,r::NoPenalty,ρ) = x
