
function forward(model::Model, q::judiVector, dobs::judiVector; options=Options(), ps=16, modelPy=nothing, mode=:QR)
    dt_Comp = get_dt(model)
    # Python model
    isnothing(modelPy) && (modelPy = devito_model(model, options))
    # Interpolate input data to computational grid
    q_data = time_resample(q[1], dt_Comp)
    d_data = time_resample(dobs[1], dt_Comp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    ts = trunc(Int, abs(src_coords[1, end] - rec_coords[1, end])/(1.5f0*dt_Comp))
    Q = qr_data(d_data, ps; ts=ts, mode=mode)

    rec, eu, _ = forward(modelPy, src_coords, rec_coords, q_data, Q, options.space_order, options.isic)
    rec = time_resample(rec, dt_Comp, dobs.geometry)
    return judiVector(dobs.geometry, rec), Q, eu
end

function born(model::Model, q::judiVector, dobs::judiVector, dm; options=Options(), ps=16, modelPy=nothing)
    dt_Comp = get_dt(model)
    # Python model
    isnothing(modelPy) && (modelPy = devito_model(model, options; dm=dm))

    # Interpolate input data to computational grid
    q_data = time_resample(q[1], dt_Comp)
    d_data = time_resample(dobs[1], dt_Comp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    ts = trunc(Int, abs(src_coords[1, end] - rec_coords[1, end])/(1.5f0*dt_Comp))
    Q = qr_data(d_data, ps; ts=ts)

    recnl, recl, eu, _ = born(modelPy, src_coords, rec_coords, q_data, Q, options.space_order, options.isic)
    recnl = time_resample(recnl, dt_Comp, dobs.geometry)
    recl = time_resample(recl, dt_Comp, dobs.geometry)
    return judiVector(dobs.geometry, recnl), judiVector(dobs.geometry, recl), Q, eu
end

function backprop(model::Model, residual::judiVector, Q::Array{Float32}, eu::PyObject;
                 options=Options(), ps=16, modelPy=nothing)
    dt_Comp = get_dt(model)
    # Python mode
    isnothing(modelPy) && (modelPy = devito_model(model, options))
    # Interpolate input data to computational grid
    d_data = time_resample(residual[1], dt_Comp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(residual.geometry[1], modelPy.shape)

    ev, _ = backprop(modelPy, d_data, rec_coords, Q, options.space_order)
    g = combine(eu, ev, options.isic)
    return remove_padding(g.data, modelPy.padsizes; true_adjoint=options.sum_padding)
end
