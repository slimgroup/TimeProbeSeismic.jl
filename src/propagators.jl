

function forward(model::Model, q::judiVector, dobs::judiVector; options=Options())
    return forward(devito_model(model, options), args...)
end
adjoint(model::Model, args...; options=Options()) = adjoint(devito_model(model, options), args...)

# Forward propagation
function forward(model::PyObject, src_coords::Array{Float32}, rcv_coords::Array{Float32},
                 wavelet::Array{Float32}, e::Array{Float32}, space_order=8)
    """
    Low level propagator, to be used through `interface.py`
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Number of time steps
    nt = wavelet.shape[0]

    # Setting forward wavefield
    u = wu.wavefield(model, space_order)

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, u)
    eu, probe_eq = time_probe(e, u)

    # Setup source and receiver
    geom_expr, _, rcv = geom.src_rec(model, u, src_coords=src_coords, nt=nt,
                                     rec_coords=rcv_coords, wavelet=wavelet)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(pde + geom_expr + probe_eq, subs=subs, name="forwardp")

    summary = op()

    # Output
    return rcv.data, eu.data, summary
end


function adjoint(model, y, src_coords, rcv_coords, space_order=8, e)
    """
    Low level propagator, to be used through `interface.py`
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Number of time steps
    nt = y.shape[0]

    # Setting adjoint wavefield
    v = wu.wavefield(model, space_order, fw=False)

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, v, fw=False)
    eu, probe_eq = time_probe(e, v)

    # Setup source and receiver
    geom_expr, _, rcv = geom.src_rec(model, v, src_coords=rcv_coords, nt=nt,
                                     rec_coords=src_coords, wavelet=y, fw=False)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(pde + geom_expr + probe_eq, subs=subs, name="adjointp")

    # Run operator
    summary = op()

    return ev.data, summary
end


function time_probe(e::Array{Float32}, wf::PyObject)
    p_e = dv.DefaultDimension(name="p_e", default_value=size(e, 2))

    # Probing vector
    nt = size(e, 1)
    q = dv.TimeFunction(name="Q", grid=wf.grid, dimensions=(wf.grid.time_dim, p_e),
                        shape=(nt, size(e, 2)))
    q.data[:, :] = e

    # Probed output
    pe = dv.Function(name="pe", grid=wf.grid, dimensions=(grid.dimensions..., p_e),
                     shape=(wf.grid.shape..., size(e, 2)))
    probing = dv.Inc(pe, q*wf)
    return pe, probing
end
