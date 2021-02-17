# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#

using DrWatson
@quickactivate :TimeProbeSeismic
import TimeProbeSeismic: qr_data

# Load starting model
# Load starting model
~isfile(datadir("models", "overthrust_model.h5")) && run(`curl -L ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5 --create-dirs -o $(datadir("models", "overthrust_model.h5"))`)
n, d, o, m0, m = h5read(datadir("models", "overthrust_model.h5"), "n", "d", "o", "m0", "m")

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
zrec = range(19*d[2], stop=19*d[2], length=nxrec)

# receiver sampling and recording time
timeD = 3000f0   # receiver recording time [ms]
dtD = get_dt(model0)    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(0f0, (n[1] - 1)*d[1]/2, length=nsrc))
ysrc = convertToCell(range(0f0, 0f0, length=nsrc))
zsrc = convertToCell(range(2*d[2], 2*d[2], length=nsrc))

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet
f0 = 0.008f0     # kHz
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
ge = Array{Any}(undef, 8, 2)
for ps=1:8
    ge[ps, 1] = judiJacobian(F0[1], q[1], 2^ps, dobs[1])'*residual[1]
    ge[ps, 2] = judiJacobian(F0[2], q[2], 2^ps, dobs[2])'*residual[2]
end

for i=1:2
    clip = maximum(g[i])/10
    figure()
    subplot(331)
    imshow(g[i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    title("Reference")
    for ps=1:8
        subplot(3,3,ps+1)
        imshow(ge[ps, i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("ps=$(2^ps)")
    end
    tight_layout()

    figure()
    subplot(331)
    imshow(g[i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    title("Reference")
    for ps=1:8
        subplot(3,3,ps+1)
        imshow(g[i]' - ge[ps,i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("Difference ps=$(2^ps)")
    end
    tight_layout()
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
semilogx([2^i for i=1:8], similar1, "-or", label="Full gradient reflections", basex=2)
semilogx([2^i for i=1:8], similar12, "-ob", label="Muted water layer turning", basex=2)
title(L"Similarity $\frac{<g, g_e>}{||g|| ||g_e||}$")
xlabel("Number of probing vectors")
legend()


# Adjoint test
at = Array{Any}(undef, 8, 2)
J0 = dot(δd, residual)
for i=1:2
    for ps=1:8
        at[ps, i] = dot(ge[ps, i], dm)
        @printf("ps=%d,  <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n",
                2^ps, J0, at[ps, i], (J0 - at[ps, i])/(J0 + at[ps, i]))
    end
end

att1 = [abs(1 - J0/at[i, 1]) for i=1:8]
att2 = [abs(1 - J0/at[i, 2]) for i=1:8]
figure()
semilogx([2^i for i=1:8], att1, basex=2, label="Probed adjoint test turning")
semilogx([2^i for i=1:8], att2, basex=2, label="Probed adjoint test reflections")
semilogx([2^i for i=1:8], [0 for i=1:8], label="target")
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
