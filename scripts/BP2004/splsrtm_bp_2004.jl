# Least-squares RTM of the BP synthetic 2007 data set
# Author: Philipp Witte
# Date: March 2018
#

using JUDI, SegyIO, JLD, PyPlot, JOLI, Random, LinearAlgebra, TimeProbeSeismic, MECurvelets
using Images, SlimOptim

# TO DO
# Set up path where data will be saved
# data_path = "/data/mlouboutin3/BP2004/bp_observed/"
data_path = "/mnt/c/Users/mathi/Dropbox (GaTech)/BP_synthetic_2004/SynData/"

using JUDI, SegyIO, JLD, PyPlot, JOLI, Random, LinearAlgebra, TimeProbeSeismic, MECurvelets

# Load velocity model(replace with correct paths)
if !isfile("bp_synthetic_2004_migration_velocity.jld")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/CompressiveLSRTM/bp_synthetic_2004_migration_velocity.jld`)
end
vp = load(join([pwd(), "/bp_synthetic_2004_migration_velocity.jld"]))["vp"] / 1f3

if !isfile("bp_synthetic_2004_water_bottom.jld")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/CompressiveLSRTM/bp_synthetic_2004_water_bottom.jld`)
end
water_bottom = load(join([pwd(), "/bp_synthetic_2004_water_bottom.jld"]))["wb"]

# Set up model structure
# d = (6.25, 6.25)
d = (10., 10.)
o = (0., 0.)
m0 = (1f0 ./ vp).^2
# n = size(m0)
n = (6744, 1195)
m0 = imresize(m0, n)
water_bottom = imresize(water_bottom, n)
model0 = Model(n, d, o, m0)

# Scan directory for segy files and create out-of-core data container
container = segy_scan(data_path, "bp_observed_data", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key = "SourceDepth")

# Set up source
src_geometry = Geometry(container; key = "source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.020)  # 27 Hz peak frequency
q = judiVector(src_geometry, wavelet)

###################################################################################################

# Set options
opt = Options(limit_m=true, buffer_size=3000f0, isic=true)

ps = 16
# Setup operators
F = judiModeling(model0, q.geometry, d_obs.geometry; options=opt)
J = judiJacobian(F, q, ps, d_obs)

# Right-hand preconditioners
D = judiDepthScaling(model0)
T = judiTopmute(model0.n, (1 .- water_bottom), [])
Mr = D*T

# Linearized Bregman parameters
x = zeros(Float32, prod(model0.n))
z = zeros(Float32, prod(model0.n))
batchsize = 2
niter = 20
fval = zeros(Float32, niter)

# Soft thresholding functions and Curvelet transform
C = joMECurvelet2D(model0.n[1], model0.n[2]; zero_finest=true, DDT=Float32)

function obj(x)
    flush(stdout)
    # Get batch
    i = randperm(d_obs.nsrc)[1:batchsize]
    d_sub = get_data(d_obs[i])
    Ml = judiMarineTopmute2D(35, d_sub.geometry)
    # Linearized residual
    d_pred = J[i]*Mr*x
    r = Ml*d_pred - Ml*d_sub
    # grad
    g = adjoint(Mr)*adjoint(J[i])*adjoint(Ml)*r
    return g
end

x0 = zeros(Float32, prod(model0.n))

g = obj(x0)

bregopt = bregman_options(maxIter=niter, verbose=2, quantile=.9, alpha=.1, antichatter=false, spg=true)
# solb = bregman(obj, x0, bregopt, C);
