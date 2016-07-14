immutable L1Penalty{T<:Number} <: Penalty
    λ::T
end
L1Penalty() = L1Penalty(1.0)

value(r::L1Penalty, x::AbstractArray) = r.λ*sumabs(x)

function grad!{T}(
        x::AbstractArray{T},
        r::L1Penalty{T},
    )
    @simd for i in eachindex(x)
        @inbounds x[i] = p.λ*sign(x[i])
    end
    return dest
end

function prox!{T}(x::AbstractArray{T}, r::L1Penalty{T}, ρ::T)
    soft_iter_thres!(x,r.λ*ρ,x)
end

isdifferentiable(::L1Penalty) = false
isdifferentiable(::L1Penalty, at) = at != 0
istwicedifferentiable(::L1Penalty) = true
istwicedifferentiable(::L1Penalty, at) = true
islipschitzcont(::L1Penalty) = true
islipschitzcont_deriv(::L1Penalty) = true
isconvex(::L1Penalty) = true
isstronglyconvex(::L1Penalty) = false

