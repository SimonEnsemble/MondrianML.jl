# 🪓
struct Box
    lo::Vector{Float64}
    hi::Vector{Float64}
end

inside(x::AbstractArray{Float64}, box::Box) = all(x .> box.lo) && all(x .≤ box.hi)

function inside(X::Matrix{Float64}, box::Box)
	for n in 1:size(X, 1)
        x = view(X, n, :)
        if ! inside(x, box)
            return false
        end
    end
    return true
end

struct Split
	dim::Int
	threshold::Float64
end

struct MondrianNode
    id::Int
    τ::Float64
    box::Box
    split::Union{Split, Nothing}
    left ::Union{MondrianNode, Nothing}
    right::Union{MondrianNode, Nothing}
end

struct MondrianPartition
    root::MondrianNode
    dims::Int
    λ::Vector{Float64}
end

function sample_mondrian_partition(box::Box, λ::Union{Float64, Vector{Float64}})
	dims = length(box.hi)
	return MondrianPartition(
		grow_mondrian_tree(box, λ), dims, λ isa Float64 ? [λ for d=1:dims] : λ
	)
end

function grow_mondrian_tree(
    box::Box, λ::Union{Float64, Vector{Float64}};
    budget::Float64=(λ isa Float64 ? λ : sum(λ)),
    parent_τ::Float64=0.0,
    leaf_counter::Ref{Int}=Ref(0)
)
    ℓs = box.hi .- box.lo
    dims = length(ℓs)

    rates = λ .* ℓs
    cut_rate = sum(rates)

    t = randexp() / cut_rate
    τ = parent_τ + t

    if t > budget
        leaf_counter[] += 1
        return MondrianNode(leaf_counter[], τ, box, nothing, nothing, nothing)
    end

    dim = sample(1:dims, Weights(rates))
    threshold = box.lo[dim] + rand() * ℓs[dim]

    box_l = Box(copy(box.lo), copy(box.hi)) # <
    box_l.hi[dim] = threshold
    box_r = Box(copy(box.lo), copy(box.hi)) # >
    box_r.lo[dim] = threshold

    remaining_budget = budget - t

    left_node  = grow_mondrian_tree(
        box_l, λ; budget=remaining_budget, parent_τ=τ, leaf_counter=leaf_counter
    )
    right_node  = grow_mondrian_tree(
        box_r, λ; budget=remaining_budget, parent_τ=τ, leaf_counter=leaf_counter
    )

    return MondrianNode(-1, τ, box, Split(dim, threshold), left_node, right_node)
end

function count_leaves(node::MondrianNode)
    if isnothing(node.split)
        return 1
    else
        return count_leaves(node.left) + count_leaves(node.right)
    end
end

count_leaves(mpartition::MondrianPartition) = count_leaves(mpartition.root)

function Base.show(io::IO, mp::MondrianPartition)
    println(io, "MondrianPartition")
    println(io, "  # dims: $(mp.dims)")
    println(io, "  λ's: $(mp.λ)")
    println(io, "  parent box: $(mp.root.box)")
    println(io, "  # partitions: ", count_leaves(mp))
end
