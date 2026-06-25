function box_id(node::MondrianNode, x::AbstractVector{Float64})
    if isnothing(node.split)
        return node.id
    end
    
    if x[node.split.dim] <= node.split.threshold
        return box_id(node.left, x)
    else
        return box_id(node.right, x)
    end
end

box_id(mpartition::MondrianPartition, x::Vector{Float64}) = box_id(
    mpartition.root, x
)

function featurize(
    mpartition::MondrianPartition,
    X::Matrix{Float64}   # n_obs × dims
)
    n_obs = size(X, 1)
    
    @assert inside(X, mpartition.root.box)
    
    box_ids = [box_id(mpartition.root, view(X, i, :)) for i in 1:n_obs]
    
    active_boxes = unique(box_ids)
    box_to_col = Dict(bid => j for (j, bid) in enumerate(active_boxes))
    
    Φ = zeros(Float64, n_obs, length(active_boxes)) # one-hot rows
    for n in 1:n_obs
        Φ[n, box_to_col[box_ids[n]]] = 1.0
    end
    
    return Φ, box_to_col
end

struct MondrianFeaturization
	mpartitions::Vector{MondrianPartition}
	dims::Int
	box_to_col::Vector{Dict{Int, Int}}
end

function sample_mondrian_featurization(
    X::Matrix{Float64},  # n_obs × dims
    λ::Vector{Float64},
    M::Int; # number of Mondrians
    box::Union{Box, Nothing}=nothing
)
    @assert length(λ) == size(X, 2)

    if isnothing(box)
        box = Box(vec(minimum(X, dims=1)) .- eps(), vec(maximum(X, dims=1)) .+ eps())
    end
    
    mpartitions = [sample_mondrian_partition(box, λ) for _ in 1:M]
    
    Φs = Vector{Matrix{Float64}}(undef, M)
    box_to_col = Vector{...}(undef, M)
    for (i, mp) in enumerate(mpartitions)
        Φs[i], box_to_col[i] = featurize(mp, X)
    end
    Φ = hcat(Φs...)

    # last dim = other partitions not containing train data
    mf = MondrianFeaturization(mpartitions, size(Φ, 2) + 1, box_to_col)

    # last feature: other partitions without training data
    #  all of these are training data. 
    #  so they don't fall in bin outside training partitions
    Φ = hcat(Φ, zeros(size(Φ, 1)))
    
    return mf, Φ
end
