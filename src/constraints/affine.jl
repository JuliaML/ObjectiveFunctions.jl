"""
Indicator function for an affine set { x | Ax = b }
"""
immutable AffineConstraint{T<:AbstractFloat} <: Penalty
    A::AbstractMatrix{T}
    b::AbstractVector{T}
    C::AbstractMatrix{T} # cached result for fast projections
    d::AbstractVector{T} # cached result for fast projections
    tol::AbstractFloat

    function AffineConstraint(A::AbstractMatrix{T},
                              b::AbstractVector{T},
                              tol::T)

        m,n = size(A)
        @_dimcheck length(b) == m
        
        ## Calculate projection
        pA = pinv(A)
        C = eye(n) - pA*A
        d = pA * b

        return new(A,b,C,d,tol)
    end
end
AffineConstraint{T}(A::AbstractMatrix{T},b::AbstractVector{T}) = AffineConstraint{T}(A,b,convert(T,DEFAULT_TOL))
prox{T}(c::AffineConstraint{T}, u::AbstractVector{T}) = (c.C * u) + c.d
value{T}(c::AffineConstraint{T}, u::AbstractVector{T}) = isapprox(c.A*u,c.b;rtol=c.tol) ? zero(T) : convert(T,Inf)
