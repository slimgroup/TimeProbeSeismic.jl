# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#
using TimeProbeSeismic, PyPlot, LinearAlgebra, HDF5, Images, SlimPlotting

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
timeD = 4000f0   # receiver recording time [ms]
dtD = 4f0  # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = div(n[1], 4) * d[1]
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
rlist = [4, 16, 32, 64, 128]
nr = length(rlist)
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

ge = Array{Any}(undef, 3, nr)

for (p, mode) in enumerate([:Fourier, :QR, :Rademacher])
    for ps=1:nr
        Jp = mode == :Fourier ? fourierJ(F0, q, rlist[ps]) : judiJacobian(F0, q, rlist[ps], dobs; mode=mode)
        ge[p, ps] = Jp'*residual
    end
end

for (p, mode) in enumerate([:Fourier, :QR, :Rademacher])
    clip = maximum(g)/10

    figure(figsize=(15, 10))
    subplot(331)
    imshow(g', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    title("Reference")
    for ps=1:nr
        subplot(3,3,ps+1)
        imshow(ge[p, ps]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("$(mode) r=$(2^ps)")
    end
    tight_layout()
    savefig(plotsdir("geo-pros-paper/", "Gradient-$(mode).png"), bbox_inches=:tight, dpi=150)

    clip = maximum(g)/10

    figure(figsize=(15, 10))
    subplot(331)
    imshow(g', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    title("Reference")
    for ps=1:nr
        subplot(3,3,ps+1)
        imshow(g' - ge[p, ps]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("$(mode) Difference r=$(2^ps)")
    end
    tight_layout()
    savefig(plotsdir("geo-pros-paper/", "Error-$(mode).png"), bbox_inches=:tight, dpi=150)
end


# Single plot
for (p, mode) in enumerate([:Fourier, :QR, :Rademacher])
    f, axarr = subplots(figsize=(30, 4), 2, nr)
    suptitle(mode)
    for ps=1:nr
        sca(axarr[1, ps])
        plot_simage(ge[p, ps]'; cmap="seismic", new_fig=false, name="r=$(rlist[ps])", perc=95)
        sca(axarr[2, ps])
        plot_simage(g' - ge[p, ps]'; cmap="seismic", new_fig=false, name="", perc=99)
        if ps > 1
            for r=1:2
                axarr[r, ps].set_yticklabels([])
                axarr[r, ps].set_yticks([])
                axarr[r, ps].set_ylabel("")
            end

        end
        axarr[1, ps].set_xticklabels([])
        axarr[1, ps].set_xticks([])
        axarr[1, ps].set_xlabel("")
    end
    axarr[2, 1].set_ylabel("Error \n \n Depth[m]")
    tight_layout()
    subplots_adjust(wspace=0, hspace=0)
    savefig(plotsdir("geo-pros-paper/", "Grads-$(mode).png"), bbox_inches=:tight, dpi=150)
end
# Plot errors

el2 = zeros(3, nr)
sim = zeros(3, nr)


for (p, mode) in enumerate([:Fourier, :QR, :Rademacher])
    for ps=1:nr
        el2[p, ps] = .5*norm(g - ge[p, ps])^2
        sim[p, ps] = simil(g, ge[p, ps])
    end
end


pis = rlist

figure(figsize=(8,4))
subplot(121)
plot(pis, el2[1, :], :r, label="DFT")
plot(pis, el2[2, :], :g, label="QR")
plot(pis, el2[3, :], :b, label="Rademacher")
xlabel("Number of probing vectors")
ylabel(L"$\ell_2$ error")
legend()
subplot(122)
plot(pis, sim[1, :], :r, label="DFT")
plot(pis, sim[2, :], :g, label="QR")
plot(pis, sim[3, :], :b, label="Rademacher")
xlabel("Number of probing vectors")
ylabel("Similarity")
legend()
savefig(plotsdir("geo-pros-paper/", "metrics.png"), bbox_inches=:tight, dpi=150)
