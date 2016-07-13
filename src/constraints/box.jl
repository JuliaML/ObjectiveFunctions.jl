"""
Indicator function for box constraint { x | l <= x <= u }
"""
immutable BoxConstraint{T<:AbstractFloat} <: Penalty
    l::AbstractArray{T} # lower bounds
    u::AbstractArray{T} # upper bounds

    function BoxConstraint(l,u)
        @_dimcheck size(l) == size(u)
        new(l,u)
    end
end
BoxConstraint{T<:AbstractFloat}(l::AbstractArray{T}, u::AbstractArray{T}) = BoxConstraint{T}(l,u)

value{T}(c::BoxConstraint{T}, x) = all(c.l .<= x .<= c.u) ? zero(T) : convert(T,Inf)

function prox!{T}(x0::AbstractArray{T}, c::BoxConstraint{T})
    @_dimcheck size(x0) == size(c.l)
    @inbounds begin
        for i in eachindex(x0)
            x0[i] = max(c.l[i], min(x0[i],c.u[i]))
        end
    end
end

