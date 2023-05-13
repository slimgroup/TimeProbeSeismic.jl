using TimeProbeSeismic, SegyIO, JLD2, SlimOptim, Statistics, HDF5, PyPlot

# Load starting model
~isfile(datadir("models", "overthrust_model.h5")) && run(`curl -L ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5 --create-dirs -o $(datadir("models", "overthrust_model.h5"))`)
n, d, o, m0, m = read(h5open(datadir("models", "overthrust_model.h5"), "r"), "n", "d", "o", "m0", "m")

n = size(m)
d = (d[1], d[2])
o = (0., 0.)

model0 = Model(n, d , o, m0)
model = Model(n, d , o, m)

# Load data and create data vector
~isfile(datadir("data", "overthrust_2D.segy")) && run(`curl -L ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D.segy --create-dirs -o $(datadir("data", "overthrust_2D.segy"))`)

block = segy_read(datadir("data", "overthrust_2D.segy"))
d_obs = judiVector(block)[1:4:end]

# Set up wavelet and source vector
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.008f0) # 8 Hz wavelet
q = judiVector(src_geometry, wavelet)[1:4:end]

###################################################################################################
# Write shots as segy files to disk
opt = Options(space_order=8)

# Setup operators
F = judiModeling(model, q.geometry, d_obs.geometry; options=opt)
F0 = judiModeling(model0, q.geometry, d_obs.geometry; options=opt)

d0 = F0 * q

ps = [1,2,4,8,16,32,64,128,256]
nr = length(ps)
si_s = div(q.nsrc, 5)

gs = judiJacobian(F0, q)[si_s]' *  (d0[si_s] - d_obs[si_s])
g = judiJacobian(F0, q)' *  (d0 - d_obs)

gsr = Vector{Any}(undef, nr)
gsg = Vector{Any}(undef, nr)
gsq = Vector{Any}(undef, nr)

for (pi, p) in enumerate(ps)
    gsr[pi] = judiJacobian(F0, q[si_s], p, d_obs[si_s]; mode=:Rademacher)' *  (d0[si_s] - d_obs[si_s])
    gsg[pi] = judiJacobian(F0, q[si_s], p, d_obs[si_s]; mode=:Gaussian)' *  (d0[si_s] - d_obs[si_s])
    gsq[pi] = judiJacobian(F0, q[si_s], p, d_obs[si_s]; mode=:QR)' *  (d0[si_s] - d_obs[si_s])
end

gr = Vector{Any}(undef, nr)
gg = Vector{Any}(undef, nr)
gq = Vector{Any}(undef, nr)

for (pi, p) in enumerate(ps)
    gr[pi] = judiJacobian(F0, q, p, d_obs; mode=:Rademacher)' * (d0 - d_obs)
    gg[pi] = judiJacobian(F0, q, p, d_obs; mode=:Gaussian)' * (d0 - d_obs)
    gq[pi] = judiJacobian(F0, q, p, d_obs; mode=:QR)' * (d0 - d_obs)
end

function meanerr(gs, g)
    n = (length(gs) * abs(mean(g.data)))
    mg = zeros(Float32, length(gs))
    for (i, gi) in enumerate(gs)
        err  = abs.((gi .- g).data)
        mg[i] = mean(err) / n
    end
    mg
end

mean_eg = meanerr(gg, g)
mean_er = meanerr(gr, g)
mean_eq = meanerr(gq, g)

mean_egs = meanerr(gsg, gs)
mean_ers = meanerr(gsr, gs)
mean_eqs = meanerr(gsq, gs)

C1 = max(maximum(mean_eg), maximum(mean_er), maximum(mean_eq))

close("all")
figure(figsize=(12, 5))
loglog(ps, mean_eg./C1, "-*r", base=10, label=L"$\mathcal{N}(0, 1)$")
loglog(ps, mean_er./C1, "-ob", base=10, label=L"$[-1, 1]$")
loglog(ps, mean_eq./C1, "-vg", base=10, label="QR")
loglog(ps, 1.25 ./ sqrt.(ps), "-xk", base=10, label=L"$\mathcal{O}(\frac{1}{\sqrt{r}})$")
xlabel("r (number of probing vector)")
ylabel("Relative rror")
ylim([.005, 1.5])
legend()
savefig("//Users/mathiaslouboutin/research/papers/RandomTraceSeismic/figures/convergence.png", dpi=150, bbox_inches="tight")

C2 = max(maximum(mean_egs), maximum(mean_ers), maximum(mean_eqs))

figure(figsize=(12, 5))
loglog(ps, mean_egs./C2, "-*r", base=10, label=L"$\mathcal{N}(0, 1)$")
loglog(ps, mean_ers./C2, "-ob", base=10, label=L"$[-1, 1]$")
loglog(ps, mean_eqs./C2, "-vg", base=10, label="QR")
loglog(ps, 1.25 ./ sqrt.(ps), "-xk", base=10, label=L"$\mathcal{O}(\frac{1}{\sqrt{r}})$")
xlabel("r (number of probing vector)")
ylabel("Relative error")
ylim([.005, 1.5])
legend()
savefig("//Users/mathiaslouboutin/research/papers/RandomTraceSeismic/figures/convergencef.png", dpi=150, bbox_inches="tight")
