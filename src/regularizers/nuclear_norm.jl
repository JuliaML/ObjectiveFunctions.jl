immutable NuclearNormPenalty{T<:AbstractFloat} <: Penalty
    λ::T
end

function value{T}(p::NuclearNormPenalty{T},A::AbstractMatrix{T})
    if size(A,1) > size(A,2)
        return trace(sqrtm(A'A))
    else
        return trace(sqrtm(A*A'))
    end
end

function prox!{T}(A::AbstractMatrix{T},r::NuclearNormPenalty{T},ρ::T)
    svdecomp = svdfact!(A)
    soft_iter_thres!(svdecomp.S,ρ*r.λ)
    copy!(A,full(svdecomp))
end

function prox{T}(A::AbstractMatrix{T},r::NuclearNormPenalty{T},ρ::T)
    svdecomp = svdfact(A)
    soft_iter_thres!(svdecomp.S,ρ*r.λ)
    full(svdecomp)
end

