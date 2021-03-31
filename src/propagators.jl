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
    op = dv.Operator(vcat(pde, probe_eq, geom_expr), subs=subs, name="forwardp", opt=ut.opt_op(model))

    summary = op()

    # Output
    return rcv.data, eu, summary
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
    op = dv.Operator(vcat(pde, probe_eq, geom_expr), subs=subs, name="adjointp", opt=ut.opt_op(model))

    # Run operator
    summary = op()

    return ev, summary
end

# Forward propagation
function born(model::PyObject, src_coords::Array{Float32}, rcv_coords::Array{Float32},
              wavelet::Array{Float32}, e::Array{Float32}, space_order::Integer=8, isic::Bool=false)
    # Number of time steps
    nt = size(wavelet, 1)

    # Setting forward wavefield
    u = wu.wavefield(model, space_order)
    ul = wu.wavefield(model, space_order, name="l")

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, u)
    pdel = model.dm == 0 ? [] : ker.wave_kernel(model, ul, q=si.lin_src(model, u, isic=isic))
    eu, probe_eq = time_probe(e, u, model; isic=isic)

    # Setup source and receiver
    geom_expr, _, rcv = geom.src_rec(model, u, src_coords=src_coords, nt=nt,
                            rec_coords=rcv_coords, wavelet=wavelet)
    geom_exprl, _, rcvl = geom.src_rec(model, ul, rec_coords=rcv_coords, nt=nt)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(vcat(pde, pdel, probe_eq, geom_expr, geom_exprl), subs=subs,
                     name="bornp", opt=ut.opt_op(model))

    summary = op()

    # Output
    return rcv.data, rcvl.data, eu, summary
end

# Forward propagation
function born_with_back(model::PyObject, src_coords::Array{Float32}, rcv_coords::Array{Float32},
                        wavelet::Array{Float32}, e::Array{Float32}, y::Array{Float32, 2},
                        space_order::Integer=8, isic::Bool=false)
    # Number of time steps
    nt = size(wavelet, 1)

    # Setting forward wavefield
    u = wu.wavefield(model, space_order)
    ul = wu.wavefield(model, space_order, name="l")
    v = wu.wavefield(model, space_order, name="b")

    # Set up PDE expression and rearrange
    pde = ker.wave_kernel(model, u)
    pdev = ker.wave_kernel(model, v)
    pdel = model.dm == 0 ? [] : ker.wave_kernel(model, ul, q=si.lin_src(model, u, isic=isic))
    eu, ev, probe_eq = time_probe_sim(e, u, v, model; isic=isic)

    # Setup source and receiver
    geom_expr, _, rcv = geom.src_rec(model, u, src_coords=src_coords, nt=nt,
                                     rec_coords=rcv_coords, wavelet=wavelet)
    geom_exprl, _, rcvl = geom.src_rec(model, ul, rec_coords=rcv_coords, nt=nt)
    geom_exprv, _, _ = geom.src_rec(model, v, src_coords=rcv_coords, nt=nt, wavelet=y)

    # Create operator and run
    subs = model.spacing_map
    op = dv.Operator(vcat(pde, pdel, pdev, probe_eq, geom_expr, geom_exprl, geom_exprv),
                     subs=subs, name="bornback", opt=ut.opt_op(model))

    summary = op()

    # Output
    return rcvl.data, eu, ev, summary
end

function time_probe(e::Array{Float32, 2}, wf::PyObject, model::PyObject; fw=true, isic=false, pe=nothing)
    p_e = dv.DefaultDimension(name="p_e", default_value=size(e, 2))
    s = fw ? wf.grid.time_dim.spacing : 1
    # Probing vector
    nt = size(e, 1)
    q = dv.TimeFunction(name="Q", grid=wf.grid, dimensions=(wf.grid.time_dim, p_e),
                        shape=(nt, size(e, 2)), time_order=0, initializer=e)
    # Probed output
    pe = dv.Function(name="$(wf.name)e", grid=wf.grid, dimensions=(wf.grid.dimensions..., p_e),
                     shape=(wf.grid.shape..., size(e, 2)), space_order=wf.space_order)

    if size(e, 2) < 17
        probing = [dv.Inc(pe, s*q*ic(wf, fw, isic)).xreplace(Dict(p_e => i)) for i=1:size(e, 2)]
    else
        probing = [dv.Inc(pe, s*q*ic(wf, fw, isic))]
    end
    return pe, probing
end

function time_probe_sim(e::Array{Float32, 2}, wfu::PyObject, wfv::PyObject,model::PyObject; isic=false)
    p_e = dv.DefaultDimension(name="p_e", default_value=size(e, 2))
    s = wfu.grid.time_dim.spacing
    # Probing vector
    nt = size(e, 1)
    q = dv.TimeFunction(name="Q", grid=wfu.grid, dimensions=(wfu.grid.time_dim, p_e),
                        shape=(nt, size(e, 2)), time_order=0, initializer=e)
    q_r = q._subs(wfu.grid.time_dim, wfu.grid.time_dim.symbolic_max - wfu.grid.time_dim)
    # Probed output
    peu = dv.Function(name="$(wfu.name)e", grid=wfu.grid, dimensions=(wfu.grid.dimensions..., p_e),
                      shape=(wfu.grid.shape..., size(e, 2)), space_order=wfu.space_order)
    pev = dv.Function(name="$(wfv.name)e", grid=wfv.grid, dimensions=(wfv.grid.dimensions..., p_e),
                      shape=(wfv.grid.shape..., size(e, 2)), space_order=wfv.space_order)

    if size(e, 2) < 17
        probing = [dv.Inc(peu, s*q*ic(wfu, true, isic)).xreplace(Dict(p_e => i)) for i=1:size(e, 2)]
        probing = vcat(probing, [dv.Inc(pev, s*q_r*ic(wfv, false, isic)).xreplace(Dict(p_e => i)) for i=1:size(e, 2)])
    else
        probing = [dv.Inc(peu, s*q*ic(wfu, true, isic)), dv.Inc(pev, s*q_r*ic(wfv, false, isic))]
    end
    return peu, pev, probing
end

function ic(wf::PyObject, fw::Bool, isic::Bool)
    return (isic || ~fw) ? wf : wf.dt2
end

function combine(eu::PyObject, ev::PyObject, isic::Bool=false)
    g = dv.Function(name="ge", grid=eu.grid, space_order=0)
    eq = isic ? (eu * ev.laplace + si.inner_grad(eu, ev)) : eu * ev
    eq = eq.subs(ev.indices[end], eu.indices[end])
    op = dv.Operator(dv.Inc(g, -eq), subs=eu.grid.spacing_map)
    op()
    g
end