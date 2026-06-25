### A Pluto.jl notebook ###
# v0.20.28

using Markdown
using InteractiveUtils

# ╔═╡ 45eb3179-bda4-472c-9e09-6e4e3f51ba6c
begin
    if isdefined(Main, :PlutoRunner)
        import Pkg; Pkg.activate("../")
    end
    using Revise
    using MondrianML, Test, PlutoUI, CairoMakie
    using MondrianML: Box
    TableOfContents()
end

# ╔═╡ eb56c5d1-dbec-4e20-9aa1-06aaae6d98ce
begin
	import AlgebraOfGraphics as AoG
	AoG.set_aog_theme!(fonts=[AoG.firasans("Light"), AoG.firasans("Light")])
	update_theme!(
		fontsize=20, 
		linewidth=4,
		markersize=14,
		titlefont=AoG.firasans("Light")
	)
end

# ╔═╡ 011e3eb6-620a-11f1-9087-adcee66c205f
md"# 🪓 Mondrian partitions"

# ╔═╡ 0b0fdbf3-45bf-4db7-b243-e297946dda75
box = Box([0.0, 0.0], [1.0, 1.2])

# ╔═╡ 3eb19759-d169-4cf9-8ffa-9fb2bf907d53
begin
	@test inside([0.2, 1.1], box)
	@test ! inside([-0.2, 0.8], box)
end

# ╔═╡ 337354f2-7e3f-4079-8ecb-f1d18deace04
mpartition = sample_mondrian_partition(box, [1.0, 0.5])

# ╔═╡ c603c3ce-6455-43cb-a3a5-6898e1da7934
boxes = get_boxes(mpartition)

# ╔═╡ 28606636-3285-4db1-82e4-a18b5186368c
X = rand(5, 2); X[:, 2] *= 1.2

# ╔═╡ 2c71de3e-86bf-4461-9e9b-4523f8bc0ad2
@assert inside(X, box)

# ╔═╡ 6835f537-07a2-4713-be35-a562ee3ac2b3
viz(mpartition, color_active_boxes=true, show_leaf_ids=true, X=X)

# ╔═╡ 00ce797f-46ca-4bcd-b171-55e3a5750d45
md"# 🏬 featurization"

# ╔═╡ 390a8ab5-517c-4979-9d33-ad7d783c6f74
begin
	local X = rand(1, 2)
	local mpartition = sample_mondrian_partition(box, [1.0, 0.5])
	local box_id = get_box_id(mpartition, X[1, :])
	println("box ID: ", box_id)
	viz(mpartition, color_active_boxes=true, show_leaf_ids=true, X=X)
end

# ╔═╡ 62b39d66-55b4-4ec8-9b98-9d6710fcbbb8
Φ₁, box_to_col = featurize(mpartition, X)

# ╔═╡ de53f552-85d9-45fe-803b-d3a76fbb3d0b
col_to_box = Dict(col => box for (box, col) in box_to_col)

# ╔═╡ bac7dec7-a402-4649-877b-a0de2ec417e6
begin
	local fig = viz(mpartition, show_leaf_ids=true)
	
	local ax = current_axis(fig)
	
	for n = 1:size(X, 1)
		scatter!(ax, X[n, 1], X[n, 2], label="$n")
	end
	Legend(fig[1, 2], ax)
	fig
end

# ╔═╡ 8aae7299-e713-40a8-b264-82889ef6fd6b
begin
	n_data = size(X, 1)
	@test size(Φ₁)[1] == n_data
	@test all(sum(Φ₁, dims=2) .≈ 1.0)
end

# ╔═╡ 94a989ee-807c-45be-a8aa-5b46d32fbf16
for n = 1:size(X, 1)
	hot_col = findfirst(Φ₁[n, :] .≈ 1.0)
	@test inside(X[n, :], boxes[col_to_box[hot_col]])
end

# ╔═╡ b0f4345b-a573-4a31-8992-f97810a1e408
md"# many-Mondrian partition featurization"

# ╔═╡ f20c60eb-e2b9-433d-bc7c-10f9cb00f25a
M = 25

# ╔═╡ a994e9d3-471f-47f1-bf08-34148d058760
mf, Φ = sample_mondrian_featurization(
	X, [0.5, 0.5], M
)

# ╔═╡ 273089d4-7dc4-4cfd-8e8c-37300ab3b264
Φ

# ╔═╡ 51b7ecb2-4e82-461d-9df1-ab64bd3833d9
@test all(sum(Φ, dims=2) .≈ M)

# ╔═╡ a2978bfd-0ca0-42e6-bf70-4e7abffd0b3d
begin
	data_in_partition_counts = vec(sum(Φ, dims=1))
	hist(
		data_in_partition_counts,
		bins=(-0.5:1:maximum(data_in_partition_counts)+0.5),
		axis=(;
			  xlabel="# data inside", 
			  ylabel="# leaves", limits=(-0.5, nothing, 0, nothing),
			  xticks=1:maximum(data_in_partition_counts)
		)
	)
end

# ╔═╡ f0c72197-f886-42bd-8409-af85d69aa5d4
md"# 🔨 Bayesian linear regression"

# ╔═╡ be4acc06-2744-4371-a3ac-acd6aa8f261a
@test featurize(X, mf) ≈ Φ

# ╔═╡ fe7fba36-1e11-468b-8816-71aee0b577eb
begin
    struct BayesianLinearRegression
        μ₀::Float64
        σ₀::Float64 # /M for function itself
        σ::Float64  # measurement noise
    end

    struct BayesianLinearRegressionFit
        μₙ::Vector{Float64}
        Λₙ::Cholesky{Float64, Matrix{Float64}}
    end

    function fit(
        model::BayesianLinearRegression,
        Φ::AbstractMatrix{Float64},   # n_obs × n_features
        y::Vector{Float64}
    )
        σ₀_inv = 1.0 / model.σ₀ ^ 2
        σ_inv  = 1.0 / model.σ ^ 2
        n_features = size(Φ, 2)

        Λₙ  = cholesky(Symmetric(σ₀_inv * I(n_features) + σ_inv * Φ' * Φ))
        rhs = σ₀_inv * fill(model.μ₀, n_features) + σ_inv * Φ' * y
        μₙ  = Λₙ \ rhs

        return BayesianLinearRegressionFit(μₙ, Λₙ)
    end

    function predict(
        fit::BayesianLinearRegressionFit,
        blr::BayesianLinearRegression,
        Φ::AbstractMatrix{Float64}
    )
        μ_pred = Φ * fit.μₙ
        V      = fit.Λₙ \ Φ'
        σ_pred = sqrt.(vec(sum(Φ .* V', dims=2)) .+ blr.σ^2)
        return μ_pred, σ_pred
    end
end

# ╔═╡ bc6d461d-53ab-45fa-af2c-5d8bf296984c
forrester(x) = (6 * x - 2)^2 * sin(12 * x - 4)   # x ∈ [0, 1]

# ╔═╡ 09bb10d8-53b4-406d-95ed-57d9ca368a2c
function test_1D(n::Int; M::Int=10, λ::Float64=1.0, seed::Int=1)
	xs = range(-0.2, 1.0, length=200)[2:end-1]

	# data
	Random.seed!(seed)
	X = rand(n, 1)
	y = [forrester(X[i, 1]) + randn() for i = 1:n]
	
	# model
	Random.seed!() # reset
	mf, Φ_train = generate_mondrian_featurization(
		X, [λ], M, box=Box([-0.2], [1.0])
	)
	println("# dims: ", mf.dims)
	blr = BayesianLinearRegression(0.0, 5.0 / M, 0.05)
	model = fit(blr, Φ_train, y)
	X_test = collect(reshape(xs, length(xs), 1))
	Φ_test = mondrian_featurize(X_test, mf)

	μ, σ = predict(model, blr, Φ_test)

	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
	
	# truth function
	lines!(xs, forrester.(xs), linestyle=:dot, label="true")

	# model 
	lines!(ax, xs, μ, label="posterior", color=Cycled(2))
    band!(ax, xs, μ .- σ, μ .+ σ, alpha=0.3, color=Cycled(2))

	# data
	scatter!(vec(X), y)

	ylims!(-10, 15)
	xlims!(0, 1)
	
	axislegend(position=:lt)
	fig
end

# ╔═╡ d5513ef7-e999-4d3e-b5a2-75db8c84dddf
test_1D(6, M=100, λ=2.0, seed=1)

# ╔═╡ Cell order:
# ╠═45eb3179-bda4-472c-9e09-6e4e3f51ba6c
# ╠═eb56c5d1-dbec-4e20-9aa1-06aaae6d98ce
# ╟─011e3eb6-620a-11f1-9087-adcee66c205f
# ╠═0b0fdbf3-45bf-4db7-b243-e297946dda75
# ╠═3eb19759-d169-4cf9-8ffa-9fb2bf907d53
# ╠═337354f2-7e3f-4079-8ecb-f1d18deace04
# ╠═c603c3ce-6455-43cb-a3a5-6898e1da7934
# ╠═28606636-3285-4db1-82e4-a18b5186368c
# ╠═2c71de3e-86bf-4461-9e9b-4523f8bc0ad2
# ╠═6835f537-07a2-4713-be35-a562ee3ac2b3
# ╟─00ce797f-46ca-4bcd-b171-55e3a5750d45
# ╠═390a8ab5-517c-4979-9d33-ad7d783c6f74
# ╠═62b39d66-55b4-4ec8-9b98-9d6710fcbbb8
# ╠═de53f552-85d9-45fe-803b-d3a76fbb3d0b
# ╠═bac7dec7-a402-4649-877b-a0de2ec417e6
# ╠═8aae7299-e713-40a8-b264-82889ef6fd6b
# ╠═94a989ee-807c-45be-a8aa-5b46d32fbf16
# ╟─b0f4345b-a573-4a31-8992-f97810a1e408
# ╠═f20c60eb-e2b9-433d-bc7c-10f9cb00f25a
# ╠═a994e9d3-471f-47f1-bf08-34148d058760
# ╠═273089d4-7dc4-4cfd-8e8c-37300ab3b264
# ╠═51b7ecb2-4e82-461d-9df1-ab64bd3833d9
# ╠═a2978bfd-0ca0-42e6-bf70-4e7abffd0b3d
# ╟─f0c72197-f886-42bd-8409-af85d69aa5d4
# ╠═be4acc06-2744-4371-a3ac-acd6aa8f261a
# ╠═fe7fba36-1e11-468b-8816-71aee0b577eb
# ╠═bc6d461d-53ab-45fa-af2c-5d8bf296984c
# ╠═09bb10d8-53b4-406d-95ed-57d9ca368a2c
# ╠═d5513ef7-e999-4d3e-b5a2-75db8c84dddf
