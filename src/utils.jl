
typedict(x::T) where {T} = Dict(fn=>getfield(x, fn) for fn ∈ fieldnames(T))

h5read(filename, keys...) = read(h5open(filename, "r"), keys...)

function qr_data(d::Array{Float32,2}, ps::Integer; seed=nothing, ts=0)
    !isnothing(seed) && Random.seed!(seed)
    S = rand([-1f0, 1f0], size(d, 1), ps)
    ts > 0 && (d = circshift(d, (-ts, 0)))
    AS = d * (d' * S) 
    Q, _ = qr(AS)
    return Matrix(Q)
end

simil(x, y) = dot(x, y)/(norm(x)*norm(y))


_wsave(s, fig::Figure; dpi::Int=150) = fig.savefig(s, bbox_inches="tight", dpi=dpi)


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