module TimeProbeSeismic

using Reexport

using PyCall
using Distributed
@reexport using LinearAlgebra, Images, HDF5, SegyIO, JOLI, JUDI, Serialization
@reexport using JLD2
@reexport using SlimOptim, PyPlot, JOLI, Printf, Random, Statistics

export h5read, forward, backprop, smooth, combine_probes, simil
export typedict, set_r!, set_ps!

# Overloaded functions import
import JUDI: judiJacobian, adjbornop, bornop, judiAbstractJacobian, judipmap, subsample
import JUDI: time_modeling, fwi_objective, lsrtm_objective

# python imports
const dv = PyNULL()
const wu = PyNULL()
const ker = PyNULL()
const geom = PyNULL()
const si = PyNULL()
const ut = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
    copy!(dv, pyimport("devito"))
    copy!(wu, pyimport("wave_utils"))
    copy!(ker,  pyimport("kernels"))
    copy!(geom, pyimport("geom_utils"))
    copy!(si, pyimport("sensitivity"))
    copy!(ut, pyimport("utils"))
end

# Propagators
include("propagators.jl")
include("interface.jl")

# JUDI functions
include("judi.jl")

# Utility functions
include("utils.jl")

end
