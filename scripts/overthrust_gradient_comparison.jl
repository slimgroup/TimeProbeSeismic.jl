using DrWatson
@quickactivate :TimeProbeSeismic

# Load starting model
!isfile(datadir("models", "overthrust_model.h5")) && download("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5", datadir("models", "overthrust_model.h5"))
n, d, o, m0, m = h5read(datadir("models", "overthrust_model.h5"), "n", "d", "o", "m0", "m")

n = Tuple(n)
o = Tuple(o)
d = Tuple(d)
m0[:, 20:end] = imfilter(m0[:, 20:end] ,Kernel.gaussian(5))
dm = vec(m - m0)

# Setup info and model structure
nsrc = 25	# number of sources
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
xsrc = convertToCell(range(0f0, (n[1] - 1)*d[1], length=nsrc))
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
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)

# data
# Nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q
d0 = Pr*F0*adjoint(Ps)*q
residual = d0 - dobs

figure()
subplot(131)
imshow(residual.data[1], vmin=-1, vmax=1, cmap="seismic", aspect="auto")
subplot(132)
imshow(residual.data[3], vmin=-1, vmax=1, cmap="seismic", aspect="auto")
subplot(133)
imshow(residual.data[5], vmin=-1, vmax=1, cmap="seismic", aspect="auto")

ps = 16

g  = zeros(Float32, n[1]+80, n[2]+80)
gpqr  = zeros(Float32, n[1]+80, n[2]+80)
gpqro  = zeros(Float32, n[1]+80, n[2]+80)
gpr  = zeros(Float32, n[1]+80, n[2]+80)

for i=1:nsrc
    @show i

    # # Adjoint
    u0 = diff(F0[i]*adjoint(Ps[i])*q[i], dims=1)
    v0 = diff(F0[i]'*adjoint(Pr[i])*residual[i], dims=1)

    nt, nx, nz = size(u0.data[1])

    global g[:, :] += sum(u0.data[1] .* v0.data[1], dims=1)[1,:,:]

    S = rand([-1f0, 1f0], nt, ps)
    G = rand([-1f0, 1f0], nt, ps)
    T = Diagonal(range(0f0, 3f0, length=nt).^1.4)
    AS = T * dobs[i].data[1] * (dobs[i].data[1]' * T' * S)

    Q, _ = qr(AS)
    Q = Matrix(Q)
    Qo = (I - Q*Q')*G

    @inbounds for i=1:nx, j=1:nz
        global gpqr[i, j] += tr(Q'*(u0.data[1][:, i, j]*(v0.data[1][:, i, j]'*Q)))
    end

    u0 = []
    v0 = []

    GC.gc()
end


nl(x) = x#/norm(x, Inf)
function plot_g(x, crop=true; scale=nothing)
    crop && (x = x[40:end-40, 40:end-40])
    x[:, 1:19] .= 0f0
    clip = isnothing(scale) ? norm(x, Inf)/5 : scale/5
    imshow(nl(x)', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
    colorbar()
end

figure()
subplot(221)
plot_g(g, true;scale=norm(g, Inf))
title("true g")
subplot(222)
plot_g(gpqr, true;scale=norm(g, Inf))
title("qr, ps=$(ps)")
subplot(223)
plot_g(g-gpqr, true;scale=norm(g, Inf))
title("qr - g, ps=$(ps)")
subplot(224)
plot_g(reshape(-dm, model.n), false)
title("true dm")
tight_layout()

simil(x, y) = dot(x[:], y[:]/(norm(x)*norm(y)))
s1 = simil(-dm, g[41:end-40, 41:end-40])
s2 = simil(-dm, gpqr[41:end-40, 41:end-40])
