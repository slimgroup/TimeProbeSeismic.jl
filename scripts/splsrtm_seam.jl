# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#

using DrWatson
@quickactivate :TimeProbeSeismic

# Load Sigsbee model
~isfile(datadir("models", "seam_model.bin")) && run(`curl -L https://www.dropbox.com/s/nyazel8g6ah4cld/seam_model.bin\?dl\=0 --create-dirs -o $(datadir("models", "seam_model.bin"))`)
~isfile(datadir("data", "seam_obn_data.bin"))&& run(`curl -L https://www.dropbox.com/s/ap2y4ny5n98kd5c/seam_obn_data.bin\?dl\=0 --create-dirs -o $(datadir("data", "seam_obn_data.bin"))`)
# Read model and set up background
vp, rho, n, d, o = deserialize(datadir("models", "bin.jld"))

@warn "Untested script"

m = 1/vp
m0 = Float32.(imfilter(m, Kernel.gaussian(15)))
m0 = m0.^2
rho0 = Float32.(imfilter(rho[:, :], Kernel.gaussian(15)))

n = size(m0)
d = (12.5, 10.)

# Setup info and model structure
model0 = Model(n, d, o, m0, rho=rho0; nb=40)

# Load data
dobs, q = deserialize(datadir("data", "seam_obn_data.bin"))

#################################################################################################

opt = Options(isic=true, space_order=8)

# Right-hand preconditioners (model topmute)
idx_wb = find_water_bottom(vp .- minimum(vp))
Tm = judiTopmute(model0.n, idx_wb, 10)  # Mute water column
S = judiDepthScaling(model0)
Mr = S*Tm

# Curevelet transform
C0 = joCurvelet2D(model0.n[1], 2*model0.n[2]; zero_finest=false, DDT=Float32, RDT=Float32)
function C_fwd(im, C, n)
    im = hcat(reshape(im, n), reshape(im, n)[:, end:-1:1])
    coeffs = C*vec(im)
    return coeffs
end

function C_adj(coeffs, C, n)
    im = reshape(C'*coeffs, n[1], 2*n[2])
    return vec(im[:, 1:n[2]] .+ im[:, end:-1:n[2]+1])
end

C = joLinearFunctionFwd_T(size(C0, 1), n[1]*n[2],
                          x -> C_fwd(x, C0, n),
                          b -> C_adj(b, C0, n),
                          Float32,Float32, name="Cmirrorext")

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
    println("Iteration: $(j), imaging sources $(i)")

    f, g = lsrtm_objective(model0, q[i], dobs[i], Mr*x, ps;optins=opt, nlind=true)
    return f, Mr'*g
end

opt = bregman_options(maxIter=20, verbose=2, quantile=.95, alpha=1, antichatter=true, spg=true)
sol = bregman(breg_obj, 0f0.*vec(m0), opt, C)