module TimeProbeSeismic

using Reexport

@reexport using LinearAlgebra, SegyIO, JOLI, PyPlot, PyCall, HDF5, Images
@reexport using SegyIO, JOLI, JUDI, SlimOptim

export h5read

h5read(filename, keys...) = read(h5open(filename, "r"), keys...)

end
