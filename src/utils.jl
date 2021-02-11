
h5read(filename, keys...) = read(h5open(filename, "r"), keys...)

function combine_probes(eu::Array{Float32, N}, ev::Array{Float32, N}, model; true_adjoint=false) where N
    g = -sum(eu.*ev, dims=N)
    return remove_padding(g, get_pad(model); true_adjoint=true_adjoint)
end

function qr_data(d::Array{Float32,2}, ps::Integer; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    S = rand([-1f0, 1f0], size(d, 1), ps)
    AS = d * (d' * S)
    Q, _ = qr(AS)
    return Matrix(Q)
end

simil(x, y) = dot(x[:], y[:])/(norm(x)*norm(y))