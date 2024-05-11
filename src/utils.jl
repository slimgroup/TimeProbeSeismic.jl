
typedict(x::T) where {T} = Dict(fn=>getfield(x, fn) for fn ∈ fieldnames(T))

function qr_data(d::Array{Float32,2}, ps::Integer; seed=nothing, ts=0, mode=:QR)
    !isnothing(seed) && Random.seed!(seed)
    S = rand([-1f0, 1f0], size(d, 1), ps)
    if mode == :Gaussian
        Q = randn(Float32, size(d, 1), ps) ./ Float32(sqrt(ps))
    elseif mode == :Rademacher
        return S ./ Float32(sqrt(ps))
    elseif mode == :QR
        ts > 0 && (d = circshift(d, (-ts, 0)))
        AS = d * (d' * S) 
        Q, _ = qr(AS)
        return Matrix(Q)
    elseif mode == :hutchpp
        ts > 0 && (d = circshift(d, (-ts, 0)))
        # Split S
        m = 3 * ps / 2
        mid = div(ps, 2)
        S1, S2 = S[:, 1:mid], S[:, mid+1:end]
        AS = d * (d' * S2)
        Q, _ = qr(AS)
        Q = Matrix(Q)
        G = sqrt(3 / m) * (S1 - Q * (Q'* S1))   
        Qhutch = hcat(Q, G)
        return convert(Matrix{Float32}, Qhutch)
    else
        throw(ArgumentError("Probing vector mode unrecognized: $(mode), must be `:QR`, `:Rademacher`, `:Gaussian` or `:hutchpp`"))
    end
end

simil(x, y) = dot(x, y)/(norm(x)*norm(y))

# Model smoother
function smooth(m::AbstractModel; sigma=3)
    nm = deepcopy(m)
    inds = [1:ni for ni ∈ size(m)]
    for i ∈ CartesianIndices(tuple(inds...))
        s = CartesianIndex(max.(1, Tuple(i) .- sigma))
        e = CartesianIndex(min.(size(m), Tuple(i) .+ sigma))
        nm.m.data[i] = mean(m.m.data[s:e])
    end
    return nm
end

function get_model(src_geom::Geometry, rec_geom::Geometry, model::AbstractModel, options::JUDIOptions)
    ~options.limit_m && return model
    return limit_model_to_receiver_area(src_geom, rec_geom, deepcopy(model), options.buffer_size)[1]
end

function get_model(src_geom::Geometry, rec_geom::Geometry, model::AbstractModel, options::JUDIOptions, dm)
    ~options.limit_m && return (model, dm)
    return limit_model_to_receiver_area(src_geom, rec_geom, deepcopy(model), options.buffer_size; pert=dm)
end

datadir(s::Vararg{String, N}) where N = join([TPSPath,"/../data/", s...])
plotsdir(s::Vararg{String, N}) where N = join([TPSPath,"/../plots/", s...])
wsave(s, fig; dpi::Int=150, kw...) = fig.savefig(s, bbox_inches="tight", dpi=dpi, kw...)


function get_dt_data(dobs, q, dt_Comp)
    # Interpolate input data to computational grid
    q_data = time_resample(make_input(q), q.geometry, dt_Comp)
    d_data = time_resample(make_input(dobs), dobs.geometry, dt_Comp)
    if size(d_data, 1) != size(q_data, 1)
        dsize = size(q_data, 1) - size(d_data, 1)
        dt0 = t0(dobs.geometry, 1) - t0(q.geometry, 1)
        @assert dt0 != 0 && sign(dsize) == sign(dt0)
        if dt0 > 0
            d_data = vcat(zeros(Float32, dsize, size(d_data, 2)), d_data)
        else
            q_data = vcat(zeros(Float32, -dsize,  size(q_data, 2)), q_data)
        end
    end
    return d_data, q_data
end   