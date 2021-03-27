using DrWatson
@quickactivate :TimeProbeSeismic

close("all")
ee = (0, .8*25, .206*25, 0)

n, d, m, m0 = h5read(datadir("models", "overthrust_model.h5"), "n", "d", "m", "m0")
m0[:, 20:end] = imfilter(m0[:, 20:end] ,Kernel.gaussian(5));
n = Tuple(n)
d = Tuple(d)
vp_t = m'.^(-.5);
vp_0 = m0'.^(-.5);

inds = Dict(j=>i for (i,j)=enumerate([2^k for k=1:8]))

plt_dict = (cmap=:gist_ncar, vmin=1.5, vmax=6.0, extent=ee, aspect=:auto)

files = readdir(datadir("fwi_overthrust"))

dfts = [f for f=files if occursin("dft", f)]
probed = [f for f=files if occursin("ps", f)]

trinds = [200, 480, 680]
# True and initial model
figure(figsize=(12, 9))
subplot(2,1,1)
imshow(vp_t; plt_dict...)
# title("True")
xlabel("X (km)")
ylabel("Depth (km)")
vlines(x=[t*d[1]/1000 for t=trinds], colors=:k, ymin=0, ymax=ee[3])
subplot(2,1,2)
imshow(vp_0; plt_dict...)
# title("Initial")
xlabel("X (km)")
ylabel("Depth (km)")
tight_layout()
wsave(plotsdir("fwi_overthrust", "Init.png"), gcf())

# Load true
res = wload(datadir("fwi_overthrust", "fwi_std.bson"))
vp_std = reshape(res[:x], n)'.^(-.5)
ϕ_std = res[:ϕ_trace]

# DFT 
ϕ_dft = Array{Any}(undef, 8)
vp_dft = Array{Any}(undef, 8)
figure(figsize=(12, 9))
subplot(3,3,1)
imshow(vp_std; plt_dict...)
title("FWI")
xlabel("X (km)")
ylabel("Depth (km)")
for f in dfts
    res = wload(datadir("fwi_overthrust", f))
    num = parse(Int, split(split(f, "_")[2], ".")[1][4:end])
    vp_dft[inds[num]] = reshape(res[:x], n)'.^(-.5)
    subplot(3, 3, inds[num]+1)
    imshow(vp_dft[inds[num]]; plt_dict...)
    title("OTDFT $(num)")
    xlabel("X (km)")
    ylabel("Depth (km)")
    ϕ_dft[inds[num]] = res[:ϕ_trace]
end
tight_layout()
wsave(plotsdir("fwi_overthrust", "DFT_fwi.png"), gcf())

# probed
ϕ_ps = Array{Any}(undef, 8)
vp_ps = Array{Any}(undef, 8)
figure(figsize=(12, 9))
subplot(3,3,1)
imshow(vp_std; plt_dict...)
title("Standard")
xlabel("X (km)")
ylabel("Depth (km)")
for f in probed
    res = wload(datadir("fwi_overthrust", f))
    num = parse(Int, split(split(f, "_")[2], ".")[1][3:end])
    vp_ps[inds[num]] = reshape(res[:x], n)'.^(-.5)
    subplot(3, 3, inds[num]+1)
    imshow(vp_ps[inds[num]]; plt_dict...)
    title("PFWI $(num)")
    xlabel("X (km)")
    ylabel("Depth (km)")
    ϕ_ps[inds[num]] = res[:ϕ_trace]
end
tight_layout()
wsave(plotsdir("fwi_overthrust", "probed_fwi.png"), gcf())
# Traces

depth = range(0, ee[3], length=n[2])

for t in trinds
    fig, axs = subplots(nrows=2, ncols=4, figsize=(9, 5), sharex=true, sharey=true)
    fig.subplots_adjust(wspace=0, hspace=0)
    for i=1:8
        axs[i].plot(vp_t[:, t], depth, label="True")
        axs[i].plot(vp_std[:, t], depth, label="FWI")
        axs[i].plot(vp_ps[i][:, t], depth, label="PFWI $(2^i)")
        axs[i].plot(vp_dft[i][:, t], depth, label="OTDFT $(2^i)")
        axs[i].set_ylim(depth[end], depth[1])
        axs[i].legend(loc="lower left")
        i%2 == 0 && axs[i].set_xlabel(L"$V_p$")
        i < 3 && axs[i].set_ylabel("Depth (km)")
    end
    title("X=$((t-1)*25/1000) km")
    # tight_layout()
    wsave(plotsdir("fwi_overthrust", "vertical_trace_$(t).png"), gcf())
end

for t in trinds
    fig, axs = subplots(nrows=1, ncols=3, figsize=(12, 9), sharey=true)
    fig.subplots_adjust(wspace=0)
    for (n,i)=enumerate([1, 4, 5])
        axs[n].plot(vp_t[:, t], depth, label="True")
        axs[n].plot(vp_std[:, t], depth, label="FWI")
        axs[n].plot(vp_ps[i][:, t], depth, label="PFWI $(2^i)")
        axs[n].plot(vp_dft[i][:, t], depth, label="OTDFT $(2^i)")
        axs[n].set_ylim(depth[end], depth[1])
        axs[n].legend(loc="lower left")
        n==1 && axs[n].set_ylabel("Depth (km)")
        n==2 && axs[n].set_xlabel(L"$V_p$")
        n==2 && axs[n].set_title("X=$((t-1)*25/1000) km")
    end
    # tight_layout()
    wsave(plotsdir("fwi_overthrust", "vertical_trace_$(t)_select.png"), gcf())
end

# Convergence
cols = ["b", "r", "y", "c", "g", "m", :lime, :indigo]

figure(figsize=(12, 9))
for i=1:8
    plot(ϕ_ps[i]/ϕ_ps[i][1], color=cols[i], linestyle=:dashed, marker="o", label="PFWI $(2^i)")
end
plot(ϕ_std/ϕ_std[1], label="STD")
for i=1:8
    plot(ϕ_dft[i]/ϕ_dft[i][1], color=cols[i], linestyle=:dashed, marker="^", label="OTDFT $(2^i)")
end
legend(loc="upper right", ncol=2)
xlabel("Iteration")
ylabel("Normalized objective")

wsave(plotsdir("fwi_overthrust", "convergence.png"), gcf())


# FWI plot for SEG abstract
figure(figsize=(12, 9))

subplot(3,3,1)
imshow(vp_t; plt_dict...)
title("True")
xlabel("X (km)")
ylabel("Depth (km)")
vlines(x=[t*d[1]/1000 for t=trinds], colors=:k, ymin=0, ymax=ee[3])

subplot(3,3,2)
imshow(vp_0; plt_dict...)
title("Initial")
xlabel("X (km)")
ylabel("Depth (km)")

subplot(3,3,3)
imshow(vp_std; plt_dict...)
title("FWI")
xlabel("X (km)")
ylabel("Depth (km)")

for (j, ps)=zip([4,5,6], [4, 16, 64])
    i = inds[ps]
    subplot(3, 3, j)
    imshow(vp_ps[i]; plt_dict...)
    title("PFWI $(ps)")
    xlabel("X (km)")
    ylabel("Depth (km)")

    subplot(3, 3, j+3)
    imshow(vp_dft[i]; plt_dict...)
    title("OTDFT $(ps)")
    xlabel("X (km)")
    ylabel("Depth (km)")
end
tight_layout()


wsave(plotsdir("fwi_overthrust", "probed_vs_dft.png"), gcf())
