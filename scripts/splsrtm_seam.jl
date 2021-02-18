# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: mlouboutin3@gatech.edu
# Date: February 2021
#

using DrWatson
@quickactivate :TimeProbeSeismic

# Load Sigsbee model
~isfile(datadir("models", "seam_model.bin")) && run(`curl -L https://www.dropbox.com/s/nyazel8g6ah4cld/seam_model.bin\?dl\=0 --create-dirs -o $(datadir("models", "seam_model.jld"))`)
~isfile(datadir("data", "seam_obn_data.bin"))&& run(`curl -L https://www.dropbox.com/s/ap2y4ny5n98kd5c/seam_obn_data.bin\?dl\=0 --create-dirs -o $(datadir("data", "acou_data_nofs.jld"))`)

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

# Info structure for linear operators
ntComp = get_computational_nt(q.geometry, dobs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), dobs.nsrc, ntComp)

#################################################################################################

opt = Options(isic=false)

# Setup operators
F0 = judiModeling(info, model0, q.geometry, dobs.geometry; options=opt)

# Right-hand preconditioners (model topmute)
idx_wb = find_water_bottom(reshape(dm, model0.n))
Tm = judiTopmute(model0.n, idx_wb, 10)  # Mute water column
S = judiDepthScaling(model0)
Mr = S*Tm

# Curevelet transform
C0 = joCurvelet2D(model0.n[1], 2*model0.n[2]; zero_finest = false, DDT = Float32, RDT = Float64)
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
                          Float32,Float64, name="Cmirrorext")

# Linearized Bregman parameters
x = zeros(Float32, info.n)
z = zeros(Float32, info.n)

batchsize = 8
niter = 5
ps = 32
fval = zeros(Float32, niter)
tk = zeros(Float32, info.n)

# Soft thresholding functions and Curvelet transform
soft_thresholding(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float64, lambda), 0.0)
soft_thresholding(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float32, lambda), 0f0)
lambda = 0f0
t = 1f-4

# Main loop
nsrc = 44
ind_left = collect(1:nsrc÷2)
ind_right = collect(nsrc÷2+1:nsrc)

for j = 1:niter
    # Select batch and set up left-hand preconditioner
    j == 1 ? batchs = batchsize : batchs = batchsize÷2
    if length(ind_left) == 0 || length(ind_right) == 0
        println("Made one pass through data, reseting source sampling")
        global ind_left = collect(1:nsrc÷2)
        global ind_right = collect(nsrc÷2+1:nsrc)
    end
    i = vcat(sample_indices(ind_left, batchs), sample_indices(ind_right, batchs))
    println("Iteration: $(j), imaging sources $(i)")
    flush(Base.stdout)
    
    phi, g = lsrtm_objective(model0, q[i], d_obs[i], Mr*x, ps; nlind=true, options=opt)

    # Step size and update variable
    g = adjoint(Mr)*g
    fval[j] = phi

    α = Float32(phi/norm(g)^2)
    # sign
    tk[:] .+= sign.(-g)
    
    # Chatter correction
    inds_z = findall(abs.(z) .> lambda)
    stepg .= α
    stepg[inds_z] .*= abs.(tk[inds_z])/j

    @printf("At iteration %d function value is %2.2e and step length is %2.2e \n", j, fval[j], α)
    flush(Base.stdout)
    # Update variables and save snapshot
    global z -= stepg[j, :] .* g
    
    # Thresholding value
    cz = C * z
    (j-1)%10 == 0 && (global lambda = quantile(abs.(cz), .925))   # estimate thresholding parameter in 1st iteration
    @printf("Lambda is %2.2e \n", lambda)

    global x = adjoint(C)*soft_thresholding(cz, lambda)

    savedict = @dict x z g lambda fval
    wsave(datadir("lsrtm", "iter_$(j).bson"), savedict)
end


