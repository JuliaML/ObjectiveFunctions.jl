immutable L1Penalty{T<:Number} <: Penalty
    λ::T
end
L1Penalty() = L1Penalty(1.0)

# TODO: consider calling value/grad for L1DistLoss?
value(p::L1Penalty, x::AbstractArray) = p.λ*sumabs(x)
function grad!{T}(
        dest::AbstractArray{T},
        p::L1Penalty{T},
        x::AbstractArray{T}
    )

    @_dimcheck length(dest) == length(x)
    @simd for i in eachindex(dest)
        @inbounds dest[i] = p.λ*sign(x[i])
    end
    return dest
end

# since grad! and prox! are elementwise for L1 penalty
grad!(x::AbstractArray,r::L1Penalty) = grad!(x,r,x)
prox!(x::AbstractArray,r::L1Penalty,ρ) = prox!(x,r,x,ρ)

isdifferentiable(::L1Penalty) = false
isdifferentiable(::L1Penalty, at) = at != 0
istwicedifferentiable(::L1Penalty) = true
istwicedifferentiable(::L1Penalty, at) = true
islipschitzcont(::L1Penalty) = true
islipschitzcont_deriv(::L1Penalty) = true
isconvex(::L1Penalty) = true
isstronglyconvex(::L1Penalty) = false

function prox!{T}(dest::AbstractArray{T}, r::L1Penalty{T}, x0::AbstractArray{T}, ρ)
    soft_iter_thres!(dest,r.λ*ρ,x0)
end
