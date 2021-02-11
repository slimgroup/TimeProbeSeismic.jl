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
timeD = 3000f0   # receiver recording time [ms]
dtD = get_dt(model0)    # receiver sampling interval [ms]

nxrec = n[1]
xrec = range(0f0, stop=(n[1] - 1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(19*d[2], stop=19*d[2], length=nxrec)

xsrc = convertToCell(range(0f0, (n[1] - 1)*d[1], length=nsrc))
ysrc = convertToCell(range(0f0, 0f0, length=nsrc))
zsrc = convertToCell(range(2*d[2], 2*d[2], length=nsrc))

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
dobs, eu, _ = forward(model0, q[1], recGeometry)