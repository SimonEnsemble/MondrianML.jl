box_to_rect(box::Box) = Rect(
       box.lo[1], box.lo[2],
       box.hi[1] - box.lo[1],
       box.hi[2] - box.lo[2]
)

function viz(
    mpartition::MondrianPartition; 
    show_leaf_ids::Bool=true,
    color_active_boxes::Bool=true, 
    color_splits::Bool=false,
    X::Union{Nothing, Matrix{Float64}}=nothing
)
    @assert length(mpartition.root.box.hi) == 2

    cmap = ColorSchemes.seaborn_crest_gradient

    fig = Figure(size=(700, 600))
    ax  = Axis(
        fig[1, 1], xlabel="x₁", ylabel="x₂", title="Mondrian partition",
        aspect=DataAspect()
    )

    τ_max = sum(mpartition.λ)

    poly!(
        ax, box_to_rect(mpartition.root.box), color=("white", 0.0), 
        strokecolor=:black, strokewidth=2
    )

    # iterate thru all nodes
    stack = [mpartition.root]
    n_leaf_w_train = 0
    while ! isempty(stack)
        node = pop!(stack)

        # bounding box of this node
        rect = box_to_rect(node.box)
        
        if isnothing(node.split) # leaf
            if show_leaf_ids
                cx = (node.box.lo[1] + node.box.hi[1]) / 2
                cy = (node.box.lo[2] + node.box.hi[2]) / 2
                text!(
                    ax, cx, cy, text=string(node.id),
                    align=(:center, :center), fontsize=12
                )
            end
            
            if color_active_boxes && ! isnothing(X)
                if any(inside(x, node.box) for x in eachrow(X))
                    n_leaf_w_train += 1
                    box_color = ColorSchemes.devonS[n_leaf_w_train]
                    poly!(
                        ax, rect, color=(box_color, 0.4), 
                        strokecolor=:black, strokewidth=1
                    )
                end
            end
            continue
        end

        # split lines
        t = node.split.threshold
        if color_splits
            split_color = get(cmap, node.τ, (0.0, τ_max))
        else
            split_color = :black
        end
        
        if node.split.dim == 1
            lines!(
                ax, [t, t], [node.box.lo[2], node.box.hi[2]], 
                color=split_color, linewidth=2
            )
        else
            lines!(
                ax, [node.box.lo[1], node.box.hi[1]], [t, t],
                color=split_color, linewidth=2
            )
        end

        push!(stack, node.left, node.right)
    end

    if ! isnothing(X)
        scatter!(X[:, 1], X[:, 2], color=:black, marker=:+)
    end

    if color_splits
        Colorbar(
            fig[1, 2], colormap=cmap, limits=(0.0, τ_max), 
            label="τ (split time)"
        )
    end

    xlims!(ax, mpartition.root.box.lo[1], mpartition.root.box.hi[1])
    ylims!(ax, mpartition.root.box.lo[2], mpartition.root.box.hi[2])

    return fig
end
