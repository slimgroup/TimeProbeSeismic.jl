# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#
using TimeProbeSeismic, PyPlot, LinearAlgebra, Printf

# Setup a 2 layers model
n = (101, 101)
d = (10., 10.)
o = (0., 0.)

m = 1.5f0^(-2) * ones(Float32, n);
m[:, 51:end] .= .25f0;

model = Model(n, d, o, m; nb=80)
model0 = smooth(model; sigma=5)
dm = model0.m - model.m

# Simple geometry
# Src/rec sampling and recording time
timeD = 1000f0   # receiver recording time [ms]
dtD = 2f0    # receiver sampling interval [ms]
nsrc = 1

nxrec = n[1]
xrec = range(0f0, stop=(n[1] - 1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(2*d[2], stop=2*d[2], length=nxrec)

xsrc = 100f0
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
opt = Options(space_order=16, sum_padding=true, dt_comp=dtD)
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)

# data
# Nonlinear modeling
dobs = F*q
d0 = F0*q
residual = d0 - dobs
δd = J*dm

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


clip = maximum(g)/10

figure()
subplot(331)
imshow(g', cmap="seismic", vmin=-clip, vmax=clip, aspect=.5)
title("Reference")
for ps=1:8
    subplot(3,3,ps+1)
    imshow(g' - ge[ps]', cmap="seismic", vmin=-clip, vmax=clip, aspect=.5)
    title("Difference ps=$(2^ps)")
end
tight_layout()

# Similarities
similar = [simil(g, ge[i]) for i=1:8]
# Similarities with muted wated
mw(x::PhysicalParameter) = mw(x.data)
mw(x::Array) = x[:, 30:end]
similar2 = [simil(mw(g), mw(ge[i])) for i=1:8]

figure()
semilogx([2^i for i=1:8], similar, "-or", label="Full gradient", base=2)
semilogx([2^i for i=1:8], similar2, "-ob", label="Muted water layer", base=2)
title(L"Similarity $\frac{<g, g_e>}{||g|| ||g_e||}$")
xlabel("Number of probing vectors")
legend()


# Adjoint test
at = Array{Any}(undef, 8)
J0 = dot(δd, residual)
for ps=1:8
    at[ps] = dot(ge[ps], dm)
    @printf("ps=%d,  <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n",
            2^ps, J0, at[ps], (J0 - at[ps])/(J0 + at[ps]))
end

att = [abs(1 - J0/at[i]) for i=1:8]
figure()
semilogx([2^i for i=1:8], att, base=2, label="Probed adjoint test")
semilogx([2^i for i=1:8], [0 for i=1:8], base=2, label="target")
title(L"$1 - \frac{<J \delta m, g_e>}{<J' \delta d, \delta m>}$")


# Probing vectors
figure();imshow(dobs.data[1]*dobs.data[1]', cmap="seismic", vmin=-10, vmax=10)
title(L"$d_{obs} d_{obs}^\top$")

figure()
for ps=1:9
    subplot(3,3,ps)
    Q = qr_data(dobs.data[1]*dobs.data[1]', 2^ps)
    imshow(Q, vmin=-.1, vmax=.1, cmap="seismic", aspect="auto")
    title("Probing vectors ps=$(2^ps)")
end
tight_layout()
