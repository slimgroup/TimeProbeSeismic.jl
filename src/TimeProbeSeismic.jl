module TimeProbeSeismic

using Reexport, Random, Statistics, LinearAlgebra

@reexport using JUDI
using JUDI.PyCall

import JUDI: judiAbstractJacobian, propagate, judiJacobian, multi_src_fg
import JUDI: time_resample, judiComposedPropagator, update!, fwi_objective, lsrtm_objective
import JUDI: MTypes, Dtypes, judiMultiSourceVector
import Base: adjoint, getindex

export forward, backprop, smooth, combine_probes, simil
export typedict, qr_data, datadir, plotsdir, wsave

TPSPath = dirname(pathof(TimeProbeSeismic))

# python imports
const dv = PyNULL()
const wf = PyNULL()
const ker = PyNULL()
const geom = PyNULL()
const si = PyNULL()
const ut = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
    copy!(dv, pyimport("devito"))
    copy!(wf, pyimport("fields"))
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
