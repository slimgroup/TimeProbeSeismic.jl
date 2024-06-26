p_dims = Dict()

# Basic caching of probing dim
function pdim(ne::Integer)
    if ne in keys(p_dims)
        p_e = p_dims[ne]
    else
        p_e = dv.CustomDimension(name="p_e", 0, ne-1, ne)
        p_dims[ne] = p_e
    end
    return p_e
end

# Forward propagation
function forward(model::PyObject, src_coords::Array{Float32}, rcv_coords::Array{Float32},
                 wavelet::Array{Float32}, e::Array{Float32}, space_order::Integer=8, isic::Bool=false)
    # Number of time steps
    nt = size(wavelet, 1)
    r = size(e, 2)

    # Setting forward wavefield
    u = wf.wavefield(model, space_order)

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, u)
    eu, probe_eq = time_probe(e, u; isic=isic)

    # Setup source and receiver
    geom_expr = geom.geom_expr(model, u, src_coords=src_coords, nt=nt,
                               rec_coords=rcv_coords, wavelet=wavelet)
    _, rcv = geom.src_rec(model, u, src_coords, rcv_coords, wavelet, nt)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(vcat(pde, geom_expr, probe_eq), subs=subs, name="forwardp$(r)")

    kw = Dict(Symbol(rcv.name) => rcv)
    summary = op(dt=model."critical_dt"; kw...)

    # Output
    return rcv.data, eu, summary
end


function backprop(model::PyObject, y::Array{Float32}, rcv_coords::Array{Float32},
                  e::Array{Float32}, space_order::Integer=8; pe=nothing, d_probe=nothing)
    # Number of time steps
    nt = size(y, 1)
    r = size(e, 2)

    # Setting adjoint wavefield
    v = wf.wavefield(model, space_order, fw=false)

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, v, fw=false)
    ev, probe_eq = time_probe(e, v; fw=false, pe=pe, d_probe=d_probe)

    # Setup source and receiver
    geom_expr = geom.geom_expr(model, v, src_coords=rcv_coords, nt=nt, wavelet=y, fw=false)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(vcat(pde, geom_expr, probe_eq), subs=subs, name="adjointp$(r)")

    # Run operator
    summary = op(dt=model."critical_dt")

    return ev, summary
end


# Forward propagation
function born(model::PyObject, src_coords::Array{Float32}, rcv_coords::Array{Float32},
              wavelet::Array{Float32}, e::Array{Float32}, space_order::Integer=8, isic::Bool=false)
    # Number of time steps
    nt = size(wavelet, 1)
    r = size(e, 2)

    # Setting forward wavefield
    u = wf.wavefield(model, space_order)
    ul = wf.wavefield(model, space_order, name="l")

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, u)
    pdel = model.dm == 0 ? [] : ker.wave_kernel(model, ul, q=si.lin_src(model, u, isic=isic))
    eu, probe_eq = time_probe(e, u; isic=isic)

    # Setup source and receiver
    geom_expr = geom.geom_expr(model, u, src_coords=src_coords, nt=nt,
                               rec_coords=rcv_coords, wavelet=wavelet)
    geom_exprl = geom.geom_expr(model, ul, rec_coords=rcv_coords, nt=nt)
    _, rcv = geom.src_rec(model, u, rec_coords=rcv_coords, nt=nt)
    _, rcvl = geom.src_rec(model, ul, rec_coords=rcv_coords, nt=nt)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(vcat(pde, pdel, probe_eq, geom_expr, geom_exprl), subs=subs,
                     name="bornp$(r)")

    kw = Dict(Symbol(rcv.name) => rcv, Symbol(rcvl.name) => rcvl)
    summary = op(dt=model."critical_dt"; kw...)

    # Output
    return rcv.data, rcvl.data, eu, summary
end


function time_probe(e::Array{Float32, 2}, wf::PyObject; fw=true, isic=false, pe=nothing, d_probe=nothing)
    ne = size(e, 2)
    p_e = isnothing(d_probe) ? dv.DefaultDimension(name="p_e", default_value=ne) : d_probe
    # sub_time
    t_sub = wf.grid.time_dim
    s = fw ? t_sub.spacing : 1
    # Probing vector
    nt = size(e, 1)
    q = dv.TimeFunction(name="Q", grid=wf.grid, dimensions=(t_sub, p_e),
                        shape=(nt, ne), time_order=0, initializer=e, space_order=0)
    # Probed output
    if isnothing(pe)
        pe = dv.Function(name="$(wf.name)e", grid=wf.grid, dimensions=(p_e, wf.grid.dimensions...),
                        shape=(ne, wf.grid.shape...), space_order=wf.space_order)
    end
    probing = [dv.Eq(pe, pe + s*q*ic(wf, fw, isic))]
    return pe, probing
end


function time_probe(e::Array{Float32, 2}, wf::Tuple{PyCall.PyObject, PyCall.PyObject}; fw=true, isic=false, pe=nothing)
    ne = size(e, 2)
    p_e = isnothing(d_probe) ? dv.DefaultDimension(name="p_e", default_value=ne) : d_probe
    # sub_time
    t_sub = wf[1].grid.time_dim
    s = fw ? t_sub.spacing : 1
    # Probing vector
    nt = size(e, 1)
    q = dv.TimeFunction(name="Q", grid=wf[1].grid, dimensions=(t_sub, p_e),
                        shape=(nt, ne), time_order=0, initializer=e, space_order=0)
    # Probed output
    if isnothing(pe)
        pe = dv.Function(name="$(wf[1].name)e", grid=wf[1].grid, dimensions=(p_e, wf[1].grid.dimensions...),
                            shape=(ne, wf[1].grid.shape...), space_order=wf[1].space_order)
    end
    probing = [dv.Eq(pe, pe + s*q*ic(wf, fw, isic))]
    return pe, probing
end


function ic(wf::PyObject, fw::Bool, isic::Bool)
    return (isic || ~fw) ? wf : wf.dt2
end


function ic(wf::Tuple{PyCall.PyObject, PyCall.PyObject}, fw::Bool, isic::Bool)
    return (isic || ~fw) ? (wf[1] + wf[2]) : (wf[1].dt2 + wf[2].dt2)
end


function combine(eu::PyObject, ev::PyObject, offsets::Number, isic::Bool=false)
    @assert offsets == 0
    g = dv.Function(name="ge", grid=eu.grid, space_order=0)
    eq = isic ? (eu * ev.laplace + si.inner_grad(eu, ev)) : eu * ev
    op = dv.Operator(dv.Inc(g, -eq), subs=eu.grid.spacing_map)
    op()
    g
end


function combine(eu::PyObject, ev::PyObject, offsets::Vector{Int64}, isic::Bool=false)
    @assert (length(offsets) % 2 == 1 || offsets[1] == 0)
    # Dimension for offsets
    nh = offsets[1] == 0 ? (length(offsets)-1) : div(length(offsets), 2)
    offs = dv.CustomDimension("hdim", 0, 2*nh, 2*nh+1)
    oh = dv.Function(name="hvals", grid=eu.grid, space_order=0, initializer=offsets,
                     dimensions=(offs,), shape=(2*nh+1,), dtype=np.int32)

    # IC with space shifts
    g = dv.Function(name="ge", grid=eu.grid, shape=(2*nh+1, eu.grid.shape...),
                    dimensions=(offs, eu.grid.dimensions...), space_order=0)
    x = eu.grid.dimensions[1]

    ev = ev._subs(x, x+oh)
    eu = eu._subs(x, x-oh)
    eq = isic ? (eu * ev.laplace + si.inner_grad(eu, ev)) : eu * ev

    op = dv.Operator(dv.Inc(g, -eq), subs=eu.grid.spacing_map)
    xs = maximum(abs.(offsets))
    xe = eu.grid.shape[1] - xs
    op(x_m=xs, x_M=xe)
    g
end
