# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#

using DrWatson
@quickactivate :TimeProbeSeismic

# Load Sigsbee model
~isfile(datadir("models", "seam_model.jld")) && run(`wget https://www.dropbox.com/s/9s8vnj087ttfhu6/seam_model.jld\?dl\=0 -P $(datadir("models", "seam_model.jld"))`)
~isfile(datadir("data", "acou_data_nofs.jld"))&& run(`wget https://www.dropbox.com/s/zx1bjxeu5qgtr2l/acou_data_nofs.jld\?dl\=0 -P $(datadir("data", "acou_data_nofs.jld"))`)

# Read model and set up background
@load datadir("models", "seam_model.jld")
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
@load datadir("data", "acou_data_nofs.jld")

# Info structure for linear operators
ntComp = get_computational_nt(q.geometry, dobs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), dobs.nsrc, ntComp)

#################################################################################################

opt = Options(isic=true)

# Setup operators
F0 = judiModeling(info, model0, q.geometry, dobs.geometry; options=opt)

# Right-hand preconditioners (model topmute)
idx_wb = find_water_bottom(reshape(dm, model0.n))
Tm = judiTopmute(model0.n, idx_wb, 10)  # Mute water column
S = judiDepthScaling(model0)
Mr = S*Tm

# Linearized Bregman parameters
x = zeros(Float32, info.n)
z = zeros(Float32, info.n)
batchsize = 100
niter = 20
fval = zeros(Float32, niter)
t = 2f-5
ps = 32

# Soft thresholding functions and Curvelet transform
soft_thresholding(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float64, lambda), 0.0)
soft_thresholding(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float32, lambda), 0f0)
C = joCurvelet2D(model0.n[1], model0.n[2]; zero_finest=true, DDT=Float32, RDT=Float64)
lambda = []

# Main loop
for j=1:niter
    println("Iteration: ", j)

    # Compute residual and gradient
    i = randperm(d_lin.nsrc)[1:batchsize]
    d_sub = get_data(d_lin[i])    # load shots into memory
    J = judiJacobian(F0[i], q[i], ps, d_sub)
    r = j==1 ? -1*d_sub : J*Mr*x - d_sub
    g = Mr'*J'*r

    # Step size and update variable
    fval[j] = .5*norm(r)^2
    isempty(t) && (global t = norm(r)^2/norm(g)^2)
    j==1 && (global lambda = 0.01*norm(C*t*g, Inf))   # estimate thresholding parameter in 1st iteration

    # Update variables and save snapshot
    global z -= t*g
    global x = adjoint(C)*soft_thresholding(C*z, lambda)

    # Save snapshot
    # save(join(["splsrtm_checkpointing_iteration_", string(j),".jld"]), "x", reshape(x, model0.n), "z", reshape(z, model0.n))
end
