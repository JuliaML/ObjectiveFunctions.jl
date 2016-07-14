"""
    FirstDerivL2Penalty{T}(λ)

Penalty on the L2-norm of the first finite difference of a Vector{T}. This
encourages the vector to be smooth.
"""
immutable FirstDerivL2Penalty{T<:Number} <: Penalty
    DtD::SymTridiagonal{T}
    λ::T

    function FirstDerivL2Penalty(n::Integer,λ::T)
        # Gram matrix of forward diff operator
        DtD = SymTridiagonal{T}(
            T[1; fill(2,n-2); 1],    # diag
            fill(-one(T),n-1)        # upper diag
        )
        return new(DtD, λ)
    end
end

FirstDerivL2Penalty(n::Integer) = FirstDerivL2Penalty{Float64}(n,1.0)
FirstDerivL2Penalty{T<:Number}(n::Integer,λ::T) = FirstDerivL2Penalty{T}(n,λ)
FirstDerivL2Penalty{T<:Number}(x::AbstractVector{T}) = FirstDerivL2Penalty{T}(length(x),one(T))
FirstDerivL2Penalty{T<:Number}(x::AbstractVector{T},λ::T) = FirstDerivL2Penalty{T}(length(x),λ)

function value{T<:Number}(r::FirstDerivL2Penalty{T}, x::AbstractVector{T})
    z = zero(T)
    for i in 2:length(x)
        @inbounds z += (x[i]-x[i-1])^2
    end
    return convert(T,0.5)*z
end

function prox{T<:Number}(r::FirstDerivL2Penalty{T}, x0::AbstractVector{T}, ρ::T)
    γ = one(T)/(ρ*r.λ)
    return (r.DtD + UniformScaling(γ)) \ (x0.*γ)
end

function prox!{T<:Number}(x0::AbstractVector{T}, r::FirstDerivL2Penalty{T}, ρ::T)
    γ = one(T)/(ρ*r.λ)
    scale!(x0,γ)
    A_ldiv_B!(lufact(r.DtD+UniformScaling(γ)), x0)
end

