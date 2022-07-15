# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#
using TimeProbeSeismic, PyPlot, LinearAlgebra, HDF5, Images

# Load starting model
~isfile(datadir("models/", "overthrust_model.h5")) && run(`curl -L ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5 --create-dirs -o $(datadir("models", "overthrust_model.h5"))`)
n, d, o, m0, m = read(h5open(datadir("models", "overthrust_model.h5"), "r"), "n", "d", "o", "m0", "m")

n = Tuple(n)
o = Tuple(o)
d = Tuple(d)
m0[:, 20:end] = imfilter(m0[:, 20:end] ,Kernel.gaussian(5))
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

# Set up receiver geometry
nxrec = n[1]
xrec = range(0f0, stop=(n[1] - 1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(12.5f0, stop=12.5f0, length=nxrec)

# receiver sampling and recording time
timeD = 3000f0   # receiver recording time [ms]
dtD = 4f0  # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = div(n[1], 2) * d[1]
ysrc = 0f0
zsrc = 12.5f0

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet
f0 = 0.010f0     # kHz
wavelet = ricker_wavelet(timeD, dtD, f0)
q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

###################################################################################################
# Write shots as segy files to disk
opt = Options(space_order=16)

# Setup operators
F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)

# data
# Nonlinear modeling
dobs = F*q
d0 = F0*q
residual = d0 - dobs
δd = J*dm

# gradient
g = J'*residual

function fourierJ(F0, q, ps)
    # Generate probability density function from source spectrum
    q_dist = generate_distribution(q)
    J = judiJacobian(F0, q)
    # Select 20 random frequencies per source location
    J.options.frequencies = Array{Any}(undef, q.nsrc)
    for j=1:q.nsrc
        J.options.frequencies[j] = select_frequencies(q_dist; fmin=0.003, fmax=0.04, nf=ps)
    end
    return J
end

ge = Array{Any}(undef, 3, 8)

for (p, mode) in enumerate([:Fourier, :QR, :Rademacher])
    for ps=1:8
        Jp = mode == :Fourier ? fourierJ(F0, q, 2^ps) : judiJacobian(F0, q, 2^ps, dobs; mode=mode)
        ge[p, ps] = Jp'*residual
    end
end

for (p, mode) in enumerate([:Fourier, :QR, :Rademacher])
    clip = maximum(g)/10

    figure(figsize=(15, 10))
    subplot(331)
    imshow(g', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    title("Reference")
    for ps=1:8
        subplot(3,3,ps+1)
        imshow(ge[p, ps]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("$(mode) r=$(2^ps)")
    end
    tight_layout()
    savefig(plotsdir("geo-pros-paper/", "Gradient-$(mode).pdf"), bbox_inches=:tight, dpi=150)

    clip = maximum(g)/10

    figure(figsize=(15, 10))
    subplot(331)
    imshow(g', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    title("Reference")
    for ps=1:8
        subplot(3,3,ps+1)
        imshow(g' - ge[p, ps]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("$(mode) Difference r=$(2^ps)")
    end
    tight_layout()
    savefig(plotsdir("geo-pros-paper/", "Error-$(mode).pdf"), bbox_inches=:tight, dpi=150)
end
