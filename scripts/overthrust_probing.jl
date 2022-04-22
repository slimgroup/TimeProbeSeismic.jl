# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#

using TimeProbeSeismic, PyPlot, LinearAlgebra, HDF5

# Load starting model
~isfile(datadir("models", "overthrust_model.h5")) && run(`curl -L ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5 --create-dirs -o $(datadir("models", "overthrust_model.h5"))`)
n, d, o, m0, m = read(h5open(datadir("models", "overthrust_model.h5"), "r"), "n", "d", "o", "m0", "m")

n = Tuple(n)
o = Tuple(o)
d = Tuple(d)
m0[:, 20:end] = imfilter(m0[:, 20:end] ,Kernel.gaussian(5))
dm = vec(m - m0)

# Setup info and model structure
nsrc = 2	# number of sources
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
xsrc = convertToCell(range(0f0, (n[1] - 1)*d[1]/2, length=nsrc))
ysrc = convertToCell(range(0f0, 0f0, length=nsrc))
zsrc = convertToCell(range(12.5f0, 12.5f0, length=nsrc))

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
g = Array{Any}(undef, 2)
g[1] = J[1]'*residual[1]
g[2] = J[2]'*residual[2]

# Probe
Jp = judiJacobian(F0, q, 2^1, dobs)
ge = Array{Any}(undef, 8, 2)
for r=1:8
    set_r!(Jp, 2^r)
    ge[r, 1] = Jp[1]'*residual[1]
    ge[r, 2] = Jp[2]'*residual[2]
end

for i=1:2
    clip = maximum(g[i])/10
    figure(figsize=(12, 8))
    subplot(331)
    imshow(g[i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    title("Reference")
    for r=1:8
        subplot(3,3,r+1)
        imshow(ge[r, i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("r=$(2^r)")
    end
    tight_layout()
    wsave(plotsdir("overthrust_single", "Probed_grad$i.png"), gcf())

    figure(figsize=(12, 8))
    subplot(331)
    imshow(g[i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    title("Reference")
    for r=1:8
        subplot(3,3,r+1)
        imshow(g[i]' - ge[r,i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("Difference r=$(2^r)")
    end
    tight_layout()
    wsave(plotsdir("overthrust_single", "Probed_err$i.png"), gcf())
end

# Similarities
similar = [simil(g[1], ge[i, 1]) for i=1:8]
similar2 = [simil(g[2], ge[i, 2]) for i=1:8]
# Similarities with muted wated
mw(x::PhysicalParameter) = mw(x.data)
mw(x::Array) = x[:, 19:end]
similar1 = [simil(mw(g[1]), mw(ge[i, 1])) for i=1:8]
similar12 = [simil(mw(g[2]), mw(ge[i, 2])) for i=1:8]

figure()
semilogx([2^i for i=1:8], similar, "-or", label="Full gradient turning", basex=2)
semilogx([2^i for i=1:8], similar2, "-ob", label="Full gradient reflections", basex=2)
semilogx([2^i for i=1:8], similar1, "-ob", label="Muted water layer turning", basex=2)
semilogx([2^i for i=1:8], similar12, "-ob", label="Muted water layer reflections", basex=2)
title(L"Similarity $\frac{<g, g_e>}{||g|| ||g_e||}$")
xlabel("Number of probing vectors")
legend()
wsave(plotsdir("overthrust_single", "simil.png"), gcf())

# Adjoint test
at = Array{Any}(undef, 8, 2)
J0 = dot(δd, residual)
for i=1:2
    for r=1:8
        at[r, i] = dot(ge[r, i], dm)
        @printf("r=%d,  <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n",
                2^r, J0, at[r, i], (J0 - at[r, i])/(J0 + at[r, i]))
    end
end

att1 = [abs(1 - J0/at[i, 1]) for i=1:8]
att2 = [abs(1 - J0/at[i, 2]) for i=1:8]
figure()
semilogx([2^i for i=1:8], att1, basex=2, label="Probed adjoint test")
semilogx([2^i for i=1:8], att2, basex=2, label="Probed adjoint test reflections")
semilogx([2^i for i=1:8], [0 for i=1:8], label="target")
title(L"$1 - \frac{<J \delta m, g_e>}{<J' \delta d, \delta m>}$")
wsave(plotsdir("overthrust_single", "adjtest.png"), gcf())

# Probing vectors
figure();imshow(dobs.data[1]*dobs.data[1]', cmap="seismic", vmin=-10, vmax=10)
title(L"$d_{obs} d_{obs}^\top$")

figure()
for r=1:3:7
    subplot(3,3,r)
    Q = qr_data(dobs.data[1], 2^(r+2))
    imshow(Q, vmin=-.1, vmax=.1, cmap="seismic", aspect="auto", interpolation="none")
    title(L"$\mathbf{Z}$ "*"r=$(2^r)")
    subplot(3, 3, r+1)
    imshow(Q'*Q,vmin=-1e-2, vmax=1e-2, cmap="seismic", aspect="auto")
    title(L"$\mathbf{Z}^\top \mathbf{Z}$")
    subplot(3, 3, r+2)
    imshow(Q*Q',vmin=-1e-2, vmax=1e-2, cmap="seismic", aspect="auto")
    title(L"$\mathbf{Z}\mathbf{Z}^\top$")
end
tight_layout()
wsave(plotsdir("overthrust_single", "Zi.png"), gcf())