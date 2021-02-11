module TimeProbeSeismic

using Reexport

using LinearAlgebra, SegyIO, JOLI, PyCall, HDF5, Images, Random
@reexport using SegyIO, JOLI, JUDI, SlimOptim, PyPlot, JOLI

export h5read, forward, backprop, smooth, combine_probes, simil

# python imports
const dv = PyNULL()
const wu = PyNULL()
const ker = PyNULL()
const geom = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
    copy!(dv, pyimport("devito"))
    copy!(wu, pyimport("wave_utils"))
    copy!(ker,  pyimport("kernels"))
    copy!(geom, pyimport("geom_utils"))
end

# Propagators
include("propagators.jl")
include("interface.jl")

# JUDI functions
include("judi.jl")

# Utility functions
include("utils.jl")

end
