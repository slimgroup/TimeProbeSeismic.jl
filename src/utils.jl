
typedict(x::T) where {T} = Dict(fn=>getfield(x, fn) for fn ∈ fieldnames(T))

function qr_data(d::Array{Float32,2}, ps::Integer; seed=nothing, ts=0)
    !isnothing(seed) && Random.seed!(seed)
    S = rand([-1f0, 1f0], size(d, 1), ps)
    ts > 0 && (d = circshift(d, (-ts, 0)))
    AS = d * (d' * S) 
    Q, _ = qr(AS)
    return Matrix(Q)
end

simil(x, y) = dot(x, y)/(norm(x)*norm(y))

# Model smoother
function smooth(m::Model; sigma=3)
    nm = deepcopy(m)
    inds = [1:ni for ni ∈ m.n]
    @inbounds for i ∈ CartesianIndices(tuple(inds...))
        s = CartesianIndex(max.(1, Tuple(i) .- sigma))
        e = CartesianIndex(min.(m.n, Tuple(i) .+ sigma))
        nm.m.data[i] = mean(m.m.data[s:e])
    end
    return nm
end

function get_model(src_geom::Geometry, rec_geom::Geometry, model::Model, options::JUDIOptions)
    ~options.limit_m && return model
    return limit_model_to_receiver_area(src_geom, rec_geom, deepcopy(model), options.buffer_size)[1]
end

function get_model(src_geom::Geometry, rec_geom::Geometry, model::Model, options::JUDIOptions, dm)
    ~options.limit_m && return (model, dm)
    return limit_model_to_receiver_area(src_geom, rec_geom, deepcopy(model), options.buffer_size; pert=dm)
end

datadir(s::Vararg{String, N}) where N = jointpath([dirname(pathof(TimeProbeSeismic)),"/../data/", s...])
plotsdir(s::Vararg{String, N}) where N = jointpath([dirname(pathof(TimeProbeSeismic)),"/../plots/", s...])
wsave(s, fig::Figure; dpi::Int=150, kw...) = fig.savefig(s, bbox_inches="tight", dpi=dpi, kw...)