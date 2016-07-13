"""
    FirstDerivPenalty{T,L2Penalty}(λ)

Penalty on the L2-norm of the first finite difference of a Vector{T}. This
encourages the vector to be smooth.
"""
immutable FirstDerivPenalty{T<:Number,L2Penalty} <: Penalty
    DtD::SymTridiagonal{T}
    λ::T

    function FirstDerivPenalty(n::Integer,λ::T)
    # Gram matrix of forward diff operator
    DtD = SymTridiagonal{T}(
        T[1; fill(2,n-2); 1],    # diag
        fill(-one(T),n-1)        # upper diag
    )
    return FirstDerivPenalty(DtD, λ)
end

function value{T<:Number}(r::FirstDerivPenalty{T,L2Penalty}, x0::AbstractVector{T}, ρ::T)
    z = zero(T)
    @simd for i in 2:length(x0)
        @inbounds z += (x[i]-x[i-1])^2
    end
    return convert(T,0.5*z)
end

function prox{T<:Number}(r::FirstDerivPenalty{T,L2Penalty}, x0::AbstractVector{T}, ρ::T)
    ρλ = ρ*r.λ
    return (r.DtD + UniformScaling(1./ρλ)) \ (x0./ρλ)
end

function prox!{T<:Number}(x0::AbstractVector{T}, r::FirstDerivPenalty{T,L2Penalty}, ρ::T)
    ρλ = ρ*r.λ
    scale!(x0,one(T)/ρλ)
    A_ldiv_B!(lufact(r.DtD+UniformScaling(1./ρλ)), x0)
end

