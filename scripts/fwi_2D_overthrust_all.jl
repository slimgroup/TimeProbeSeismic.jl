# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#
using TimeProbeSeismic, SegyIO, JLD2, SlimOptim

# Load starting model
~isfile(datadir("models", "overthrust_model.h5")) && run(`curl -L ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5 --create-dirs -o $(datadir("models", "overthrust_model.h5"))`)
n, d, o, m0, m = read(h5open(datadir("models", "overthrust_model.h5"), "r"), "n", "d", "o", "m0", "m")
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
    # dft mode
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
ProjBound(x) = median([mmin x mmax], dims=2)[1:end]

# FWI with SPG
options = spg_options(verbose = 3, maxIter = fevals, memory = 3, iniStep = 1f0)
g_const = 0
sol = spg(x->objective_function(x, nothing), vec(m0), ProjBound, options)

# Save results
@save datadir("fwi_overthrust", "fwi_std.jld2") sol

for i=1:8
    ps = 2^i
    # FWI with SPG
    global g_const = 0
    sol = spg(x->objective_function(x, ps), vec(m0), ProjBound, options)

    # Save results
    # wsave
    @save datadir("fwi_overthrust", "fwi_ps$(ps).jld2") sol
    # FWI with SPG
    global g_const = 0
    sol = spg(x->objective_function(x, ps; dft=true), vec(m0), ProjBound, options)

    # Save results
    @save datadir("fwi_overthrust", "fwi_dft$(ps).jld2") sol
end
