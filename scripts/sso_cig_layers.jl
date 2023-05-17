# Author: Mathias Louboutin
# Date: June 2021
#

using JUDI, LinearAlgebra, Images, PyPlot
using DSP, ImageGather, Printf, TimeProbeSeismic
using SlimPlotting

# Set up model structure
n = (301, 301)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v =  2.5f0 .* ones(Float32,n)
v[:, 151:151] .= 4f0
v0 = 2.5f0 .* ones(Float32,n)
# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2

# Setup info and model structure
nsrc = 101	# number of sources
model = Model(n, d, o, m; nb=80)
model0 = Model(n, d, o, m0; nb=80)
dm = model.m - model0.m

# Set up receiver geometry
nxrec = 151
xrec = range(0f0, stop=(n[1] -1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(20f0, stop=20f0, length=nxrec)

# receiver sampling and recording time
timeD = 3000f0   # receiver recording time [ms]
dtD = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(0f0, stop=(n[1] -1)*d[1], length=nsrc))
ysrc = convertToCell(zeros(Float32, nsrc))
zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet
f0 = 0.015f0     # kHz
wavelet = ricker_wavelet(timeD, dtD, f0)
q = diff(judiVector(srcGeometry, wavelet))

###################################################################################################
opt = Options(space_order=12, isic=false, sum_padding=false)

# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
J0 = judiJacobian(F(model0), q)

# Nonlinear modeling
dD = J0*dm
dobs = F*q
rtm = J0'*dD

# Common surface offset image gather
offsets = -500f0:model.d[1]:500f0
# offsets = [0f0]
J = judiJacobian(F(model0), q, 32, dobs; offsets=offsets, mode=:QR)

ssodm = J'*dD

depth = zeros(301)
depth[41:end] .= 1f0

plot_simage(depth .* ssodm[:, 151, :]', (model.d[1], model.d[end]))