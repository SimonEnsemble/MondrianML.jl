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
    using MondrianML, Test, PlutoUI, CairoMakie, Random, LinearAlgebra
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
mf, Φ_train = sample_mondrian_featurization(
	X, [0.5, 0.5], M
)

# ╔═╡ c3c37294-794e-4d5d-a8bc-b61aad6a8eda
size(Φ_train)

# ╔═╡ be4acc06-2744-4371-a3ac-acd6aa8f261a
@test featurize(X, mf) ≈ Φ_train

# ╔═╡ 246cca43-0cc3-49fc-8971-cae674457e13
size(X)[1] * M

# ╔═╡ 9d0047f0-caae-41cb-b0b9-2997441504fa
@test all(sum(Φ_train, dims=2) .== M)

# ╔═╡ d397cd5d-58d2-4a81-9bc7-dd8b4c8cc512
@test all(Φ_train[:, end] .≈ 0.0)

# ╔═╡ a2978bfd-0ca0-42e6-bf70-4e7abffd0b3d
begin
	data_in_partition_counts = vec(sum(Φ_train, dims=1))
	hist(
		data_in_partition_counts,
		bins=(-0.5:1:maximum(data_in_partition_counts)+0.5),
		axis=(;
			  xlabel="# data inside", 
			  ylabel="# boxes", limits=(-0.5, nothing, 0, nothing),
			  xticks=1:maximum(data_in_partition_counts)
		)
	)
end

# ╔═╡ f0c72197-f886-42bd-8409-af85d69aa5d4
md"# 🔨 Bayesian linear regression"

# ╔═╡ fe7fba36-1e11-468b-8816-71aee0b577eb
begin
    # see Bayesian linear regression in Murphy and Bishop too.
    struct BLR
        μ₀::Float64
        σ₀::Float64 # variance of function itself / # partitions
        σ::Float64  # measurement noise
    end
    
    struct BLRFit
        μₙ::Vector{Float64}
        Λₙ::Cholesky{Float64, Matrix{Float64}}
        L⁻¹::LowerTriangular{Float64, Matrix{Float64}}  # cached L⁻¹
        σ::Float64
        log_ev::Float64
    end
    
    function fit(
        Φ::AbstractMatrix{Float64},
        y::Vector{Float64},
        blr::BLR
    )
        n, p   = size(Φ)
        σ₀_inv = 1.0 / blr.σ₀^2
        σ_inv  = 1.0 / blr.σ^2
        μ₀_vec = fill(blr.μ₀, p)
    
        Λₙ  = cholesky(Symmetric(σ₀_inv * I(p) + σ_inv * Φ' * Φ))
        rhs = σ₀_inv * μ₀_vec + σ_inv * Φ' * y
        μₙ  = Λₙ \ rhs
        L⁻¹ = inv(Λₙ.L)
    
        # log evidence
        log_det_Λ₀ = p * log(σ₀_inv)
        log_det_Λₙ = 2.0 * sum(log.(diag(Λₙ.L)))
        μₙᵀΛₙμₙ = sum(abs2, Λₙ.U * μₙ)
        μ₀ᵀΛ₀μ₀   = σ₀_inv * dot(μ₀_vec, μ₀_vec)
    
        log_ev = 0.5 * (log_det_Λ₀ - log_det_Λₙ)         # log|Λ₀|/|Λₙ|
               - 0.5 * n * log(2π * blr.σ^2)               # normaliser
               - 0.5 * σ_inv * (dot(y,y) + μ₀ᵀΛ₀μ₀ - μₙᵀΛₙμₙ)  # quadratics
    
        return BLRFit(μₙ, Λₙ, L⁻¹, blr.σ, log_ev)
    end
    
    function predict(
        Φ::AbstractMatrix{Float64},
        fit::BLRFit
    )
        μ_pred = Φ * fit.μₙ
    
        # Z = (L⁻¹ Φᵀ), so row-norms of Z give φᵢᵀ Λₙ⁻¹ φᵢ
        Z  = fit.L⁻¹ * Φ'                          # n_features × n_obs
        σ_pred = sqrt.(vec(sum(Z .^ 2, dims=1)) .+ fit.σ^2)
    
        return μ_pred, σ_pred
    end
end

# ╔═╡ bc6d461d-53ab-45fa-af2c-5d8bf296984c
forrester(x) = (6 * x - 2)^2 * sin(12 * x - 4)   # x ∈ [0, 1]

# ╔═╡ 13c0a584-79fb-4c18-a508-9ba46c719de3
begin
	# test with known regression
	w = [-1.0, 4.0]
	local n = 100
	local ϵ = 0.01
	X_lr = rand(n, 2) * 10
	y_lr = X_lr * w .+ randn(n)
	
	blr_test = BLR(1.0 * 0.5 + 4.0 * 0.5, 1.0, ϵ)
	model_test = fit(X_lr, y_lr, blr_test)
	@test isapprox(model_test.μₙ, w, atol=0.1)
end

# ╔═╡ 09bb10d8-53b4-406d-95ed-57d9ca368a2c
function test_1D(n::Int; M::Int=10, λ::Float64=1.0, seed::Int=1)
	xs = range(-0.2, 1.0, length=200)[2:end-1]

	# data
	Random.seed!(seed)
	X = rand(n, 1)
	y = [forrester(X[i, 1]) + 0.2 * randn() for i = 1:n]
	
	# model
	Random.seed!() # reset
	mf, Φ_train = sample_mondrian_featurization(
		X, [λ], M, box=Box([-0.2], [1.0])
	)
	println("# dims: ", mf.dims)
	blr = BLR(0.0, 10.0 / M, 0.2)
	model = fit(Φ_train, y, blr)
	X_test = collect(reshape(xs, length(xs), 1))
	Φ_test = featurize(X_test, mf)
	println("log evidence: ", model.log_ev)

	μ, σ = predict(Φ_test, model)

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
test_1D(10, M=10, λ=3.0, seed=6)

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
# ╠═c3c37294-794e-4d5d-a8bc-b61aad6a8eda
# ╠═be4acc06-2744-4371-a3ac-acd6aa8f261a
# ╠═246cca43-0cc3-49fc-8971-cae674457e13
# ╠═9d0047f0-caae-41cb-b0b9-2997441504fa
# ╠═d397cd5d-58d2-4a81-9bc7-dd8b4c8cc512
# ╠═a2978bfd-0ca0-42e6-bf70-4e7abffd0b3d
# ╟─f0c72197-f886-42bd-8409-af85d69aa5d4
# ╠═fe7fba36-1e11-468b-8816-71aee0b577eb
# ╠═bc6d461d-53ab-45fa-af2c-5d8bf296984c
# ╠═13c0a584-79fb-4c18-a508-9ba46c719de3
# ╠═09bb10d8-53b4-406d-95ed-57d9ca368a2c
# ╠═d5513ef7-e999-4d3e-b5a2-75db8c84dddf
