"""

Indicator function on the probability simplex.

Arguments:
----------

* **n** -- embedded dimension of the simplex
* **tol** -- absolute tolerance for testing sum(x) == 1.0

Example:
--------

simplex = SimplexConstraint(n, tol=1e-10)
simplex(x) # tests if x lies on the n-simplex\n
simplex([0,0,1,0,0]) # 0.0\n
simplex([0,0,-1,0,1]) # Inf\n
simplex([1,0,1,0,0]) # Inf\n
"""
immutable SimplexConstraint{T<:Number} <: Penalty
    tol::T
end
SimplexConstraint() = SimplexConstraint(DEFAULT_TOL)

function value{T<:Number}(r::SimplexConstraint,x::AbstractArray{T})
    if allnonneg(x) && abs(sum(x)-1.0) <= r.tol
        return zero(T)
    else
        return convert(T,Inf)
    end
end

## See Chen and Ye (2011) "Projection Onto A Simplex"
## http://arxiv.org/pdf/1101.6081v2.pdf
function prox!{T<:AbstractFloat}(x::AbstractVector{T}, ::SimplexConstraint)
    n = length(x)
    sort!(x, rev=true)
    xsum = cumsum(x)
    t = (xsum[end]-one(T))/n

    @inbounds begin
        for i=1:(n-1)
            if (xsum[i]-1)/i >= x[i+1]
                x = (ysum[i]-1)/i
                break
            end
        end
        for i in eachindex(x)
            x[i] = x[i] - t
            if x[i] < 0
                x[i] = zero(T)
            end
        end
    end
    return x
end
