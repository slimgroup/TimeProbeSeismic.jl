
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

function normalize!(d::judiVector)
    for i=1:d.nsrc
        d.data[i] ./= mapslices(norm, d.data[i], dims=1)
    end
end

normalize(d::Vector) = d / norm(d)

function loss(d0::judiVector, d1::judiVector)
    loss = 0
    for i=1:d0.nsrc
        for r=1:size(d0.data[i], 2)
            loss += dot(d0.data[i][:, r], normalize(d0.data[i][:, r]) - normalize(d1.data[i][:, r]))
        end
    end
    loss
end

# Model smoother
function smooth(m::Model; sigma=3)
    nm = deepcopy(m)
    inds = [sigma+1:ni-sigma for ni ∈ m.n]
    for i ∈ CartesianIndices(tuple(inds...))
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