"""
    L2Penalty(λ)

Penalty on squared l2 norm, f(x) = (λ/2)⋅Σᵢ(xᵢ)²
"""
immutable L2Penalty{T<:Number} <: Penalty
    λdiv2::T # scale of penalty

    L2Penalty(λ::T) = new(convert(T,0.5*λ))
end

L2Penalty{T}() = L2Penalty(one(T))
value{T}(r::L2Penalty{T}, x0::AbstractArray{T}) = r.λdiv2*sumabs2(x0)

function prox!{T}(x0::AbstractArray{T}, r::L2Penalty{T}, ρ::T)
	scale!(x0, (one(T) / (2*r.λdiv2*ρ + one(T))))
end


"""
    L2Prox(λ,ρ)

Proximal operator on squared L2 norm penalty, prox[f,ρ](x) = argminₓ f(x) + 1/2ρ||x-x₀||²
with f(x) = (λ/2)⋅Σᵢ(xᵢ)².
"""
immutable L2Prox{T<:Number} <: Penalty
    penalty::L2Penalty
    γ::T # 1/(λ*ρ + 1)

    L2Prox(λ::T,ρ::T) = new(L2Penalty(λ),one(T)/(λ*ρ+one(T)))
end

L2Prox{T}() = L2Prox(one(T),one(T))
L2Prox{T}(λ::T) = L2Prox(λ::T,one(T))

value{T}(r::L2Prox{T}, x::AbstractArray{T}) = value(r.penalty,x)

function prox!{T}(r::L2Prox{T}, x0::AbstractArray{T}, ρ::T)
    scale!(x0, r.γ2)
end
