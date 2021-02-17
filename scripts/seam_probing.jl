# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#

using DrWatson
@quickactivate :TimeProbeSeismic

# Load Sigsbee model
~isfile(datadir("models", "seam_model.jld")) && run(`curl -L https://www.dropbox.com/s/9s8vnj087ttfhu6/seam_model.jld\?dl\=0 --create-dirs -o $(datadir("models", "seam_model.jld"))`)
~isfile(datadir("data", "acou_data_nofs.jld"))&& run(`curl -L https://www.dropbox.com/s/zx1bjxeu5qgtr2l/acou_data_nofs.jld\?dl\=0 --create-dirs -o $(datadir("data", "acou_data_nofs.jld"))`)

# Read model and set up background
@load datadir("models", "seam_model.jld") n d o vp rho
vp = imresize(vp, (3501, 1501))
rho = imresize(rho, (3501, 1501))
m = vp.^(-2)
m0 = Float32.(imfilter(m, Kernel.gaussian(15)))
rho0 = Float32.(imfilter(rho[:, :], Kernel.gaussian(15)))
dm = m0 - m

n = size(m0)
d = (12.5, 10.)

# Setup info and model structure
model0 = Model(n, d, o, m0, rho=rho0; nb=40)

# Load data
@load datadir("data", "acou_data_nofs.jld") dobs q
dobs = dobs[[1, 22]]
q = q[[1, 22]]
# Info structure for linear operators
ntComp = get_computational_nt(q.geometry, dobs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), dobs.nsrc, ntComp)

#################################################################################################
# Write shots as segy files to disk
opt = Options(space_order=16)

# Setup operators
F0 = judiModeling(info, model0, q.geometry, dobs.geometry; options=opt)
J = judiJacobian(F0, q)

# data
# Nonlinear modeling
d0 = F0*q
residual = d0 - dobs
δd = J*dm

# Probe
ge = Array{Any}(undef, 6, 2)
for ps=1:6
    ge[ps, 1] = judiJacobian(F0[1], q[1], 2^ps, dobs[1])'*residual[1]
    ge[ps, 2] = judiJacobian(F0[2], q[2], 2^ps, dobs[2])'*residual[2]
end

for i=1:2
    clip = maximum(g[i])/10
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
end
tight_layout()
