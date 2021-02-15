# Forward propagation
function forward(model::PyObject, src_coords::Array{Float32}, rcv_coords::Array{Float32},
                 wavelet::Array{Float32}, e::Array{Float32}, space_order::Integer=8, isic::Bool=false)
    # Number of time steps
    nt = size(wavelet, 1)

    # Setting forward wavefield
    u = wu.wavefield(model, space_order)

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, u)
    eu, probe_eq = time_probe(e, u, model; isic=isic)

    # Setup source and receiver
    geom_expr, _, rcv = geom.src_rec(model, u, src_coords=src_coords, nt=nt,
                                     rec_coords=rcv_coords, wavelet=wavelet)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(vcat(pde, geom_expr, probe_eq), subs=subs, name="forwardp")

    summary = op()

    # Output
    return rcv.data, eu.data, summary
end


function backprop(model::PyObject, y::Array{Float32}, rcv_coords::Array{Float32},
                 e::Array{Float32}, space_order::Integer=8)
    # Number of time steps
    nt = size(y, 1)

    # Setting adjoint wavefield
    v = wu.wavefield(model, space_order, fw=false)

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, v, fw=false)
    ev, probe_eq = time_probe(e, v, model; fw=false)

    # Setup source and receiver
    geom_expr, _, _ = geom.src_rec(model, v, src_coords=rcv_coords, nt=nt,
                                     wavelet=y, fw=false)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(vcat(pde, geom_expr, probe_eq), subs=subs, name="adjointp")

    # Run operator
    summary = op()

    return ev.data, summary
end


function time_probe(e::Array{Float32, 2}, wf::PyObject, model::PyObject; fw=true, isic=false)
    p_e = dv.DefaultDimension(name="p_e", default_value=size(e, 2))
    s = fw ? wf.grid.time_dim.spacing : 1
    # Probing vector
    nt = size(e, 1)
    q = dv.TimeFunction(name="Q", grid=wf.grid, dimensions=(wf.grid.time_dim, p_e),
                        shape=(nt, size(e, 2)), time_order=0, initializer=e)
    # Probed output
    pe = dv.Function(name="pe", grid=wf.grid, dimensions=(wf.grid.dimensions..., p_e),
                     shape=(wf.grid.shape..., size(e, 2)))
    probing = dv.Inc(pe, s*q*ic(wf, fw, model, isic))
    return pe, probing
end

function ic(wf::PyObject, fw::Bool, model::PyObject, isic::Bool)
    ~fw && (return wf)
    ~isic && (return wf.dt2)
    return wf.dt2 * model.m + wf.laplace
end
