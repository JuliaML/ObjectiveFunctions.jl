# Now each penalty just needs to define grad!(x,::Penalty) and prox!(x,::Penalty)
grad(x::AbstractArray, p::Penalty) = grad!(copy(x),r)
prox{T}(x::AbstractArray{T}, r::Penalty) = prox!(copy(x),r,ρ)
prox{T}(x::AbstractArray{T}, r::Penalty) = prox!(copy(x),r)

# Both grad(penalty, x) and grad(x, penalty) are valid?
grad(p::Penalty, x::AbstractArray) = grad(p,x)
prox(p::Penalty, x::AbstractArray,ρ) = prox(p,x,ρ)
prox(p::Penalty, x::AbstractArray) = prox(p,x)

# Null penalty type
immutable NoPenalty <: Penalty end
grad!{T}(x::AbstractArray{T},r::NoPenalty) = fill!(y,zero(T))
prox!(y::AbstractArray,r::NoPenalty) = y
