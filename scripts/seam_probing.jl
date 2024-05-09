# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#

using TimeProbeSeismic, Serialization, PyPlot

# Load Sigsbee model
~isfile(datadir("models", "seam_model.bin")) && run(`curl -L https://www.dropbox.com/s/nyazel8g6ah4cld/seam_model.bin\?dl\=0 --create-dirs -o $(datadir("models", "seam_model.bin"))`)
~isfile(datadir("data", "seam_obn_data.bin"))&& run(`curl -L https://www.dropbox.com/s/ap2y4ny5n98kd5c/seam_obn_data.bin\?dl\=0 --create-dirs -o $(datadir("data", "seam_obn_data.bin"))`)
# Read model and set up background
vp, rho, n, d, o = deserialize(datadir("models", "bin.jld"))

m = vp.^(-2)
m0 = Float32.(imfilter(m, Kernel.gaussian(15)))
rho0 = Float32.(imfilter(rho[:, :], Kernel.gaussian(15)))
dm = m0 - m

n = size(m0)
d = (12.5, 10.)

# Setup info and model structure
model0 = Model(n, d, o, m0, rho=rho0; nb=40)

# Load data
dobs, q = deserialize(datadir("data", "seam_obn_data.bin"))
dobs = dobs[[1, 22]]
q = q[[1, 22]]

# Info structure for linear operators
ntComp = get_computational_nt(q.geometry, dobs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), dobs.nsrc, ntComp)

#################################################################################################
# Write shots as segy files to disk
opt = Options(space_order=16, isic=true)

# Setup operators
F0 = judiModeling(info, model0, q.geometry, dobs.geometry; options=opt)
J = judiJacobian(F0, q)

# data
# Nonlinear modeling
d0 = F0*q
residual = d0 - dobs

# Probe
ge = Array{Any}(undef, 6, 2)
for ps=1:6
    ge[ps, 1] = judiJacobian(F0[1], q[1], 2^ps, dobs[1])'*residual[1]
    ge[ps, 2] = judiJacobian(F0[2], q[2], 2^ps, dobs[2])'*residual[2]
end

for i=1:2
    clip = maximum(ge[ps, i])/10
    figure()
    for ps=1:6
        subplot(2,3,ps)
        imshow(ge[ps, i]', cmap="seismic", vmin=-clip, vmax=clip, aspect="auto")
        title("ps=$(2^ps)")
    end
    tight_layout()
end

# Probing vectors
figure();imshow(dobs.data[1]*dobs.data[1]', cmap="seismic", vmin=-10, vmax=10)
title(L"$d_{obs} d_{obs}^\top$")

figure()
for ps=1:9
    subplot(3,3,ps)
    Q = qr_data(dobs.data[1]*dobs.data[1]', 2^ps)
    imshow(Q, vmin=-.1, vmax=.1, cmap="seismic", aspect="auto")
    title("Probing vectors ps=$(2^ps)")
ends
tight_layout()
