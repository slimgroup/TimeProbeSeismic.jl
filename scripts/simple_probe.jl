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

m = 1.5f0^(-2) * ones(Float32, n);
m[:, 51:end] .= .25f0;
collect(m[i, i÷2+30:end] .= .35f0 for i=1:101);

model = Model(n, d, o, m; nb=80)
model0 = smooth(model; sigma=5)

# Simple geometry
# Src/rec sampling and recording time
timeD = 1000f0   # receiver recording time [ms]
dtD = get_dt(model0)    # receiver sampling interval [ms]
nsrc = 3

nxrec = n[1]
xrec = range(0f0, stop=(n[1] - 1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(2*d[2], stop=2*d[2], length=nxrec)

xsrc = convertToCell(range(0f0, stop=(n[1] - 1)*d[1], length=nsrc))
ysrc =  convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc =  convertToCell(range(2*d[2], stop=2*d[2], length=nsrc))

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)
# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet
f0 = 0.015f0     # kHz
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
    ge[ps] = judiJacobian(F0, q, 2^ps, dobs)'*residual
end


clip = maximum(g)/10

figure()
subplot(331)
imshow(g', cmap="seismic", vmin=-clip, vmax=clip, aspect=.5)
title("Reference")
for ps=1:8
    subplot(3,3,ps+1)
    imshow(ge[ps]', cmap="seismic", vmin=-clip, vmax=clip, aspect=.5)
    title("ps=$(2^ps)")
end
tight_layout()

# Similarities
similar = [simil(g, ge[i]) for i=1:8]
# Similarities with muted wated
mw(x::PhysicalParameter) = mw(x.data)
mw(x::Array) = x[:, 33:end]
similar2 = [simil(mw(g), mw(ge[i])) for i=1:8]

figure()
plot(similar, "--r", label="Full gradient")
plot(similar2, "--b", label="Muted water layer")
title(L"$\frac{<g, g_e>}{||g|| ||g_e||}$")
legend()
