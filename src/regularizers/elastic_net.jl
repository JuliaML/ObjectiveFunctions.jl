"""
Elastic net regularization, f(x) = λ₁⋅Σᵢ|xᵢ| + 0.5⋅λ₂⋅Σᵢ(xᵢ)² 
"""
immutable ElasticNetPenalty{T<:Number} <: Penalty
    γ1::T # scale of L1 penalty, λ₁
    γ2::T # half scale of L2 penalty, 0.5⋅λ₂
    ElasticNetPenalty(λ1::T,λ2::T) = new(λ1,convert(T,0.5)*λ2)
end
ElasticNetPenalty{T<:Number}(λ1::T,λ2::T) = ElasticNetPenalty{T}(λ1,λ2)
ElasticNetPenalty() = ElasticNetPenalty(1.0,1.0)

value{T}(r::ElasticNetPenalty{T}, x::AbstractArray{T}) = r.γ1*sumabs(x)+r.γ2*sumabs2(x)

function prox!{T}(r::ElasticNetPenalty{T}, x0::AbstractArray{T}, ρ::T)

    # precalculate re-used factors
    thres::T = r.γ1*ρ
    shrinkage::T = 1/(2*r.γ2*ρ + 1)
    
    for i in eachindex(x0)
        # first do soft-thresholding (for L1 norm)
        if x0[i] > 0
            @inbounds x0[i] = max(zero(T), x0[i] - thres)
        else
            @inbounds x0[i] = min(zero(T), thres + x0[i])
        end
        # then do iterative shrinkage (for L2 norm)
        @inbounds x0[i] *= shrinkage
    end
end

