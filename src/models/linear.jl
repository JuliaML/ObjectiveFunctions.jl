"""
    LinearLeastSquares(A,B,λ)

Linear least squares model: f(x) = λ/2||A*x - b||²
"""
immutable LinearLeastSquares{T<:Number} <: Penalty
    A::AbstractMatrix{T}
    B::AbstractVecOrMat{T}
    λ::T
end

value(m::LinearLeastSquares{T},x::AbstractVecOrMat{T}) = convert(T,0.5)*m.λ*value(L2DistLoss(), m.B, m.A*x)

"""
Proximal operator for linear least squares:

argminₓ  λ/2||A*x - b||² + 1/2ρ||x - x₀||²
"""
immutable LinearLeastSquaresProx{T<:Number,M<:AbstractMatrix{T}} <: ProxOp
    penalty::LinearLeastSquares{T}
    AtA_ρλI::Base.LinAlg.LU{T,M} # LU factorization of AᵀA + (1/ρλ)*I
    AtB::Matrix{T} # Cached AᵀB
    γ::T # 1/ρ⋅λ

    function LinearLeastSquaresProx(p::LinearLeastSquaresProx{T}, ρ::T)
        A = p.A
        b = p.b
        n = size(A,2)

        # calculate AᵀB    
        AtB = At_mul_B(A,B)

        # LU decomp on AᵀA + (1/λ⋅ρ)*I
        γ::T = 2*cost.λhalf*ρ
        AtA_ρλI = lufact(At_mul_B(A,A) + UniformScaling(γ))

        return LeastSquaresProx(penalty,AtA_ρλI,AtB,γ)
    end

end

function prox!{T}(x, p::LinearLeastSquaresProx{T})
    # overwrite x
    scale!(x,p.γ)           # x *= 1/ρ⋅λ
    axpy!(one(T),p.AtB,x)   # x += AᵀB

    # Now, x = AᵀB + (1/λ⋅ρ)*x₀

    # Compute (AᵀA + (1/λ⋅ρ)*I) \ AᵀB + (1/λ⋅ρ)*x₀
    A_ldiv_B!(loss.AtA_ρI, x)

    # A_ldiv_B! overwrites solution in x
    return x
end


