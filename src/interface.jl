
function forward(model::Model, q::judiVector, dobs::judiVector; options=Options(), ps=16)
    # Python model
    modelPy = devito_model(model, options)
    # Interpolate input data to computational grid
    q_data = time_resample_data(q[1], get_dt(model))
    d_data = time_resample_data(dobs[1], get_dt(model))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    Q = qr_data(d_data, ps)

    rec, eu, _ = forward(modelPy, src_coords, rec_coords, q_data, Q, options.space_order, options.isic)
    rec = time_resample_data(rec, dobs.geometry.dt[1], get_dt(model), dobs.geometry.t[1])
    return judiVector(dobs.geometry, rec), Q, eu
end

function born(model::Model, q::judiVector, dobs::judiVector, dm; options=Options(), ps=16)
    # Python model
    modelPy = devito_model(model, options)
    update_dm(modelPy, dm, options)

    # Interpolate input data to computational grid
    q_data = time_resample_data(q[1], get_dt(model))
    d_data = time_resample_data(dobs[1], get_dt(model))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    Q = qr_data(d_data, ps)

    recnl, recl, eu, _ = born(modelPy, src_coords, rec_coords, q_data, Q, options.space_order, options.isic)
    recnl = time_resample_data(recnl, dobs.geometry.dt[1], get_dt(model), dobs.geometry.t[1])
    recl = time_resample_data(recl, dobs.geometry.dt[1], get_dt(model), dobs.geometry.t[1])
    return judiVector(dobs.geometry, recnl), judiVector(dobs.geometry, recl), Q, eu
end

function backprop(model::Model, residual::judiVector, Q::Array{Float32}; options=Options(), ps=16)
    # Python model
    modelPy = devito_model(model, options)
    # Interpolate input data to computational grid
    d_data = time_resample_data(residual[1], get_dt(model))

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(residual.geometry[1], modelPy.shape)

    ev, _ = backprop(modelPy, d_data, rec_coords, Q, options.space_order)
    return ev
end
