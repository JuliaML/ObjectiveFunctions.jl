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
        return FirstDerivL2Penalty(DtD, λ)
    end
end

FirstDerivL2Penalty{T}(n::Integer,λ::T) = FirstDerivL2Penalty{T}(n,λ)
FirstDerivL2Penalty{T}(a::AbstractVector{T},λ::T=one(T)) = FirstDerivL2Penalty{T}(length(x),λ)

function value{T<:Number}(r::FirstDerivL2Penalty{T}, x0::AbstractVector{T}, ρ::T)
    z = zero(T)
    @simd for i in 2:length(x0)
        @inbounds z += (x[i]-x[i-1])^2
    end
    return convert(T,0.5*z)
end

function prox{T<:Number}(r::FirstDerivL2Penalty{T}, x0::AbstractVector{T}, ρ::T)
    ρλ = ρ*r.λ
    return (r.DtD + UniformScaling(1./ρλ)) \ (x0./ρλ)
end

function prox!{T<:Number}(x0::AbstractVector{T}, r::FirstDerivL2Penalty{T}, ρ::T)
    ρλ = ρ*r.λ
    scale!(x0,one(T)/ρλ)
    A_ldiv_B!(lufact(r.DtD+UniformScaling(1./ρλ)), x0)
end

