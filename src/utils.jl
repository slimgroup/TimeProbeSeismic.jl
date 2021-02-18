
typedict(x::T) where {T} = Dict(fn=>getfield(x, fn) for fn ∈ fieldnames(T))

h5read(filename, keys...) = read(h5open(filename, "r"), keys...)

function qr_data(d::Array{Float32,2}, ps::Integer; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    S = rand([-1f0, 1f0], size(d, 1), ps)
    AS = d * (d' * S)
    Q, _ = qr(AS)
    return Matrix(Q)
end

simil(x, y) = dot(x[:], y[:])/(norm(x)*norm(y))
