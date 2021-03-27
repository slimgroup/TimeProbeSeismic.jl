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

# Setup info and model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m)

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
xsrc = (n[1] - 1)*d[1]/2
ysrc = 0f0
zsrc = 12.5f0

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet
f0 = 0.010f0     # kHz
wavelet = ricker_wavelet(timeD, dtD, f0)
q = judiVector(srcGeometry, wavelet)
q_dist = generate_distribution(q)

###################################################################################################
# Write shots as segy files to disk
opt = Options(space_order=16)

# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)

# data
# Nonlinear modeling
dobs = F*q

# Probing vectors
figure();imshow(dobs.data[1]*dobs.data[1]', cmap="seismic", vmin=-10, vmax=10)
title(L"$d_{obs} d_{obs}^\top$")

timea = 0:4f0:3000f0

fig, axs = subplots(5, 3, sharex=:col, sharey=:row, gridspec_kw=Dict(:hspace=> 0, :wspace=> 0))
for (i, ps)=enumerate([4, 16, 32, 64, 256])
    @show i
    # QR
    Q = qr_data(dobs.data[1], ps)
    axs[i, 1].imshow(Q*Q',vmin=-.05, vmax=.05, cmap="seismic", aspect="auto")
    i == 1 && axs[i, 1].set_title(L"$\mathbf{Q}\mathbf{Q}^\top$")
    axs[i, 1].set_ylabel("r=$(ps)", rotation=0, labelpad=20)
    # gaussian
    E = rand([-1f0, 1f0], length(timea), ps)/sqrt(ps)
    axs[i, 2].imshow(E*E',vmin=-.25, vmax=.25, cmap="seismic", aspect="auto")
    i == 1 && axs[i, 2].set_title(L"$\mathbf{Z}\mathbf{Z}^\top$")
    # DFT
    freq = select_frequencies(q_dist; fmin=0.002, fmax=0.04, nf=ps)
    F = exp.(-2*im*pi*timea*freq')/sqrt(ps)
    axs[i, 3].imshow(real.(F*F'),vmin=-.5, vmax=.5, cmap="seismic", aspect="auto")
    i == 1 && axs[i, 3].set_title(L"$\mathbf{F}\mathbf{F}^\top$")
end
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

wsave(plotsdir("overthrust_single", "Zi.png"), gcf())
