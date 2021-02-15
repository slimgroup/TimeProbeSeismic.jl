# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#
using DrWatson
@quickactivate :TimeProbeSeismic

# Load starting model
~isfile(datadir("models", "overthrust_model.h5")) && download("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5", datadir("models", "overthrust_model.h5"))
n, d, o, m0, m = h5read(datadir("models", "overthrust_model.h5"), "n", "d", "o", "m0", "m")
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ model0.m)
vmin = ones(Float32,model0.n) .* 1.3f0
vmax = ones(Float32,model0.n) .* 6.5f0
vmin[:, 1:21] .= v0[:, 1:21]   # keep water column fixed
vmax[:, 1:21] .= v0[:, 1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data and create data vector
~isfile(datadir("data", "overthrust_2D.segy")) && download("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D.segy",datadir("data", "overthrust_2D.segy"))

block = segy_read(datadir("data", "overthrust_2D.segy"))
d_obs = judiVector(block)

# Set up wavelet and source vector
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.008f0) # 8 Hz wavelet
q = judiVector(src_geometry, wavelet)

########################################### FWI ####################################################

# Optimization parameters
fevals = 20
batchsize = 20
fvals = []
ps = 2

# Objective function for library
function objective_function(x, ps)
    model0.m .= x;

    # select batch          "elapsed_time", elapsed_time
    idx = randperm(d_obs.nsrc)[1:batchsize]
    f, g = fwi_objective(model0, q[idx], d_obs[idx], ps)
    g[:, 1:19] .= 0f0
    global fvals; fvals = [fvals; f]
    return f, vec(g.data/norm(g, Inf))   # normalize gradient for line search
end

# Bound projection
ProjBound(x) = median([mmin x mmax], dims=2)[1:end]

for i=1:8
    ps = 2^i
    objfun(x) = objective_function(x, ps)
    # FWI with SPG
    options = spg_options(verbose = 3, maxIter = fevals, memory = 3, iniStep = 1f0)
    sol = spg(objfun, vec(m0), ProjBound, options)

    # Save results
    # wsave
    wsave(datadir("fwi_overthrust", "fwi_ps$(ps).bson"), typedict(sol))
end
