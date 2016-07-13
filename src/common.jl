macro _not_implemented()
    quote
        throw(ArgumentError("Not implemented for the given type"))
    end
end

macro _dimcheck(condition)
    :(($(esc(condition))) || throw(DimensionMismatch("Dimensions of the parameters don't match")))
end

function allnonneg{T<:Real}(a::AbstractArray{T})
  @simd for i in eachindex(a)
    @inbounds a[i] < 0 && return false
  end
  return true
end

"""
    soft_iter_thres!(x,λ)

Performs soft iterative thresholding of Array x at level given by λ.
Each element in x is shrunk towards zero by λ (i.e. λ is added for
negative elements and subtracted for positive elements). If a sign
change occurs in either direction (i.e. -λ ≤ x[i] ≤ λ) then that
element is set to zero.
"""
function soft_iter_thres!{T}(x::AbstractArray{T},λ::T)
    @inbounds begin
        @simd for i in eachindex(x)
            if x[i] > 0
                x[i] = max(zero(T), x[i] - λ)
            else
                x[i] = min(zero(T), λ + x[i])
            end
        end
    end
end

"""
    soft_iter_thres!(dest,λ,x)

Performs soft iterative thresholding of Array x at level given by λ,
overwriting the result to Array dest.
"""
function soft_iter_thres!{T}(y::AbstractArray{T},λ::T,x::AbstractArray{T})
    @_dimcheck length(y) == length(x)
    @inbounds begin
        @simd for i in eachindex(x)
            if x[i] > 0
                y[i] = max(zero(T), x[i] - λ)
            else
                y[i] = min(zero(T), λ + x[i])
            end
        end
    end
end
