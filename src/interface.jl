
function forward(model::Model, q::judiVector, dobs::judiVector; options=Options(), ps=16, modelPy=nothing)
    # Python model
    isnothing(modelPy) && (modelPy = devito_model(model, options))
    # Interpolate input data to computational grid
    q_data = time_resample_data(q[1], get_dt(model))
    d_data = time_resample_data(dobs[1], get_dt(model))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    ts = trunc(Int, abs(src_coords[1, end] - rec_coords[1, end])/(1.5f0*get_dt(model)))
    Q = qr_data(d_data, ps; ts=ts)

    rec, eu, I, _ = forward(modelPy, src_coords, rec_coords, q_data, Q, options.space_order, options.isic)
    rec = time_resample_data(rec, dobs.geometry.dt[1], get_dt(model), dobs.geometry.t[1])

    I = remove_padding(I.data, modelPy.padsizes)

    return judiVector(dobs.geometry, rec), Q, eu, I
end

function born(model::Model, q::judiVector, dobs::judiVector, dm; options=Options(), ps=16, modelPy=nothing)
    # Python model
    isnothing(modelPy) && (modelPy = devito_model(model, options; dm=dm))

    # Interpolate input data to computational grid
    q_data = time_resample_data(q[1], get_dt(model))
    d_data = time_resample_data(dobs[1], get_dt(model))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    ts = trunc(Int, abs(src_coords[1, end] - rec_coords[1, end])/(1.5f0*get_dt(model)))
    Q = qr_data(d_data, ps; ts=ts)

    recnl, recl, eu, I, _ = born(modelPy, src_coords, rec_coords, q_data, Q, options.space_order, options.isic)
    recnl = time_resample_data(recnl, dobs.geometry.dt[1], get_dt(model), dobs.geometry.t[1])
    recl = time_resample_data(recl, dobs.geometry.dt[1], get_dt(model), dobs.geometry.t[1])

    I = remove_padding(I.data, modelPy.padsizes)

    return judiVector(dobs.geometry, recnl), judiVector(dobs.geometry, recl), Q, eu, I
end

function born_with_back(model::Model, q::judiVector, dobs::judiVector, dm; options=Options(), ps=16, modelPy=nothing)
    # Python model
    isnothing(modelPy) && (modelPy = devito_model(model, options; dm=dm))
    # Interpolate input data to computational grid
    q_data = time_resample_data(q[1], get_dt(model))
    d_data = time_resample_data(dobs[1], get_dt(model))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    ts = trunc(Int, abs(src_coords[1, end] - rec_coords[1, end])/(1.5f0*get_dt(model)))
    Q = qr_data(d_data, ps; ts=ts)

    recl, eu, ev, _  = born_with_back(modelPy, src_coords, rec_coords, q_data, Q, d_data[end:-1:1, :],
                                      options.space_order, options.isic)
    recl = time_resample_data(recl, dobs.geometry.dt[1], get_dt(model), dobs.geometry.t[1])
    g = combine(eu, ev, options.isic)
    return remove_padding(g.data, modelPy.padsizes; true_adjoint=options.sum_padding), judiVector(dobs.geometry, recl)
end

function backprop(model::Model, residual::judiVector, Q::Array{Float32}, eu::PyObject;
                 options=Options(), ps=16, modelPy=nothing)
    # Python mode
    isnothing(modelPy) && (modelPy = devito_model(model, options))
    # Interpolate input data to computational grid
    d_data = time_resample_data(residual[1], get_dt(model))

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(residual.geometry[1], modelPy.shape)

    ev, _ = backprop(modelPy, d_data, rec_coords, Q, options.space_order)
    g = combine(eu, ev, options.isic)
    return remove_padding(g.data, modelPy.padsizes; true_adjoint=options.sum_padding)
end
