# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#

using DrWatson
@quickactivate :TimeProbeSeismic

# Setup a 2 layers model
n = (101, 101)
d = (10., 10.)
o = (0., 0.)

m = 1.5f0^(-2) * ones(Float32, n)
m[:, 5:end] .= .25f0

model = Model(n, d, o, m)
model0 = smooth(model; sigma=5)

# Simple geometry
# Src/rec sampling and recording time
timeD = 1000f0   # receiver recording time [ms]
dtD = get_dt(model0)    # receiver sampling interval [ms]
nsrc = 1

nxrec = n[1]
xrec = range(0f0, stop=(n[1] - 1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(2*d[2], stop=2*d[2], length=nxrec)

xsrc = (n[1] - 1)*d[1]/2
ysrc = 0f0
zsrc = 2*d[2]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)
# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet
f0 = 0.01f0     # kHz
wavelet = ricker_wavelet(timeD, dtD, f0)
q = judiVector(srcGeometry, wavelet)

# Forward operator
opt = Options(space_order=16)
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)

# data
# Nonlinear modeling
dobs = F*q
d0 = F0*q
residual = d0 - dobs

# gradient
g = J'*residual

# Probe
ge = Array{Any}(undef, 8)
for ps=1:8
    d0, Q, eu = forward(model0, q, dobs; ps=2^ps)
    ev = backprop(model0, residual, Q)
    ge[ps] = combine_probes(ev, eu, model)
end

figure()
subplot(331)
imshow(g', cmap="seismic", vmin=-1e2, vmax=1e2, aspect=.5)
title("true gradient")
for ps=1:8
    subplot(3,3,ps+1)
    imshow(ge[ps]', cmap="seismic", vmin=-1e2, vmax=1e2, aspect=.5)
    title("Probed ps=$(2^ps)")
end
tight_layout()

plot([simil(g, ge[i]) for i=1:8])
