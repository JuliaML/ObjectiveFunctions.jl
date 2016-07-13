"""
Indicator function for vectors of specified length in L2 Distance { x | |x|₂ = len }

Also works for matrices and higher order arrays { X | vecnorm(X) = 1}
"""
immutable L2NormEqConstraint{T<:AbstractFloat} <: Penalty
    len::T
    tol::T

    function L2NormEqConstraint(len::T,tol::T)
        len > 0 || error("length constraint must be positive.")
        new(len,tol)
    end
end
L2NormEqConstraint{T<:AbstractFloat}(len::T) = L2NormEqConstraint{T}(len,convert(T,DEFAULT_TOL))
L2NormEqConstraint() = L2NormEqConstraint(1.0)

function value{T<:AbstractFloat}(c::L2NormEqConstraint,x::AbstractArray{T})
    if abs(vecnorm(x)-c.len) <= c.tol
        return zero(T)
    else
        return convert(T,Inf)
    end
end

function prox!{T<:AbstractFloat}(x0::AbstractArray{T}, c::L2NormEqConstraint)
    scale!(x0,c.len/vecnorm(x0))
end

# TODO: Other norms?
