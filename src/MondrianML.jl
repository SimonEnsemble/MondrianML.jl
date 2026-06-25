module MondrianML

using Random, Statistics, StatsBase, CairoMakie, ColorSchemes, LinearAlgebra

include("mondrian_partition.jl")
include("viz.jl")
include("featurization.jl")

export MondrianPartition, Box, Split, MondrianNode, inside, sample_mondrian_partition,
    get_boxes, # mondrian_partition.jl
    viz, # viz.jl
    get_box_id, featurize, sample_mondrian_featurization # featurization.jl

end # module
