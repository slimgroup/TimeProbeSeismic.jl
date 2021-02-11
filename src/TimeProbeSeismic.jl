module TimeProbeSeismic

using Reexport

@reexport using LinearAlgebra, SegyIO, JOLI, PyPlot, PyCall, HDF5, Images
@reexport using SegyIO, JOLI, JUDI, SlimOptim

export h5read

# python imports
dv = pyimport("devito")
pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
wu = pyimport("wave_utils")
ker = pyimport("kernels")
geom = pyimport("geom_utils")

# Propagators
include("propagators.jl")

# JUDI functions
include("judi.jl")

h5read(filename, keys...) = read(h5open(filename, "r"), keys...)

end
