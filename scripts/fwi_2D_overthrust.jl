# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#
using TimeProbeSeismic, SegyIO, Random, SlimOptim, JLD2, HDF5, Statistics, LinearAlgebra

# Load starting model
~isfile(datadir("models", "overthrust_model.h5")) && run(`curl -L ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5 --create-dirs -o $(datadir("models", "overthrust_model.h5"))`)
n, d, o, m0, m = read(h5open(datadir("models", "overthrust_model.h5"), "r"), "n", "d", "o", "m0", "m")
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ m0)
vmin = ones(Float32,model0.n) .* 1.3f0
vmax = ones(Float32,model0.n) .* 6.5f0
@show size(vmin), size(v0)
vmin[:, 1:21] .= v0[:, 1:21]   # keep water column fixed
vmax[:, 1:21] .= v0[:, 1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data and create data vector
~isfile(datadir("data", "overthrust_2D.segy")) && run(`curl -L ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D.segy --create-dirs -o $(datadir("data", "overthrust_2D.segy"))`)

block = segy_read(datadir("data", "overthrust_2D.segy"))
d_obs = judiVector(block)

# Set up wavelet and source vector
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.008f0) # 8 Hz wavelet
q = judiVector(src_geometry, wavelet)
q_dist = generate_distribution(q)

########################################### FWI ####################################################

# Optimization parameters
fevals = 20
batchsize = 20
fvals = []
g_const = 0
frequencies = Array{Any}(undef, batchsize)

# Objective function for library
function objective_function(x, ps; dft=false)
    Base.flush(stdout)
    model0.m .= x;

    # select batch          
    idx = randperm(d_obs.nsrc)[1:batchsize]
    # dft modew
    if dft
        for k=1:batchsize
            frequencies[k] = select_frequencies(q_dist; fmin=0.003, fmax=0.02, nf=ps)
        end
        opt = Options(frequencies=frequencies)
        f, g = fwi_objective(model0, q[idx], d_obs[idx]; options=opt)
    elseif isnothing(ps)
        f, g = fwi_objective(model0, q[idx], d_obs[idx])
    else
        f, g = fwi_objective(model0, q[idx], d_obs[idx], ps)
    end
    g[:, 1:19] .= 0f0
    g_const == 0 && global g_const = 1 / norm(g, Inf)
    return f, vec(g.data.*g_const)   # normalize gradient for line search
end

# Bound projection
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2),model0.n)
options = spg_options(verbose = 3, maxIter = fevals, memory = 3, iniStep = 1f0)

# FWI with probing
ps = 32
global g_const = 0
sol = spg(x->objective_function(x, ps), vec(m0), proj, options)

# Save results
# wsave
@save datadir("fwi_overthrust_basic", "fwi_ps$(ps).jld2") sol

# FWI with SPG
g_const = 0
sol = spg(x->objective_function(x, nothing), vec(m0), proj, options)

# Save results
@save datadir("fwi_overthrust_basic", "fwi_std.jld2") sol

