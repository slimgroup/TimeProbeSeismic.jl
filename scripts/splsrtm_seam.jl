# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#
using TimeProbeSeismic, Serialization, MECurvelets, SlimOptim

# Download SEAM model and data
~isfile(datadir("models", "seam_model.bin")) && run(`curl -L https://www.dropbox.com/s/nyazel8g6ah4cld/seam_model.bin\?dl\=0 --create-dirs -o $(datadir("models", "seam_model.bin"))`)
~isfile(datadir("data", "seam_obn_1km.jld")) && run(`curl -L https://www.dropbox.com/s/6hzgel1itkuqb20/seam_obn_1km.jld\?dl\=0 --create-dirs -o $(datadir("data", "seam_obn_1km.jld"))`)

# Read model and set up background
vp, rho, n, d, o = deserialize(datadir("models", "seam_model.bin"))

m = vp.^(-2)
m0 = Float32.(imfilter(m, Kernel.gaussian(15)))
rho0 = Float32.(imfilter(rho[:, :], Kernel.gaussian(15)))
dm = m0 - m

# Setup info and model structure
model0 = Model(n, d, o, m0, rho=rho0; nb=40)

# Load data
@load datadir("data", "seam_obn_1km.jld") dobs q
convgeom = x -> GeometryIC{Float32}([getfield(x.geometry, s) for s=fieldnames(GeometryIC)]...)
convdata = x -> convert(Array{Array{Float32, 2}, 1}, x.data)
conv_to_new_jv = x -> judiVector(convgeom(x), convdata(x))

q = conv_to_new_jv(q)
dobs = conv_to_new_jv(dobs)

#################################################################################################

opt = Options(isic=true, space_order=8)
ps = 32

# Right-hand preconditioners (model topmute)
idx_wb = find_water_bottom(vp .- minimum(vp))
Tm = judiTopmute(model0.n, idx_wb, 10)  # Mute water column
S = judiDepthScaling(model0)
Mr = S*Tm

# Curevelet transform
C = joMECurvelet2D(model.n)

# Setup random shot selection
batchsize = 4
nsrc = 44
ind_left = collect(1:nsrc÷2)
ind_right = collect(nsrc÷2+1:nsrc)

function breg_obj(x)
    # Force log update
    Base.flush(stdout)
    # Select batch and set up left-hand preconditioner
    if length(ind_left) == 0 || length(ind_right) == 0
        println("Made one pass through data, reseting source sampling")
        global ind_left = collect(1:nsrc÷2)
        global ind_right = collect(nsrc÷2+1:nsrc)
    end
    i = vcat(sample_indices(ind_left, batchsize), sample_indices(ind_right, batchsize))
    batchsize == 4  && (global batchsize = 2)

    f, g = lsrtm_objective(model0, q[i], dobs[i], Mr*x, ps; options=opt, nlind=true)
    return f, Mr'*g
end

# breg_opt = bregman_options(maxIter=5, verbose=2, quantile=.975, alpha=1, store_trace=true, antichatter=true, spg=true)
# sol = bregman(breg_obj, 0f0.*vec(m0), breg_opt, C)

# @save datadir("seam_rtm", "splsrtm.jld2") sol

f, g = breg_obj(0f0.*vec(m0))