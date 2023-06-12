
function forward(model::AbstractModel, q::judiVector, dobs::judiVector; options=Options(), ps=16, modelPy=nothing, mode=:QR)
    dt_Comp = get_dt(model)
    # Python model
    isnothing(modelPy) && (modelPy = devito_model(model, options))
    # Interpolate input data to computational grid
    q_data = time_resample(make_input(q), q.geometry, dt_Comp)
    d_data = time_resample(make_input(dobs), dobs.geometry, dt_Comp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    ts = trunc(Int, abs(src_coords[1, end] - rec_coords[1, end])/(1.5f0*dt_Comp))
    Q = qr_data(d_data, ps; ts=ts, mode=mode)

    rec, eu, _ = forward(modelPy, src_coords, rec_coords, q_data, Q, options.space_order,  options.IC == "isic")
    rec = time_resample(rec, dt_Comp, dobs.geometry)
    return judiVector(dobs.geometry, rec), Q, eu
end

function born(model::AbstractModel, q::judiVector, dobs::judiVector, dm; options=Options(), ps=16, modelPy=nothing)
    dt_Comp = get_dt(model)
    # Python model
    isnothing(modelPy) && (modelPy = devito_model(model, options; dm=dm))

    # Interpolate input data to computational grid
    q_data = time_resample(make_input(q), q.geometry, dt_Comp)
    d_data = time_resample(make_input(dobs), dobs.geometry, dt_Comp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(q.geometry[1], modelPy.shape)
    rec_coords = setup_grid(dobs.geometry[1], modelPy.shape)

    # QR probing vector
    ts = trunc(Int, abs(src_coords[1, end] - rec_coords[1, end])/(1.5f0*dt_Comp))
    Q = qr_data(d_data, ps; ts=ts)

    recnl, recl, eu, _ = born(modelPy, src_coords, rec_coords, q_data, Q, options.space_order, options.IC == "isic")
    recnl = time_resample(recnl, dt_Comp, dobs.geometry)
    recl = time_resample(recl, dt_Comp, dobs.geometry)
    return judiVector(dobs.geometry, recnl), judiVector(dobs.geometry, recl), Q, eu
end


function backprop(model::AbstractModel, residual::judiVector, Q::Array{Float32}, eu::PyObject;
                 options=Options(), ps=16, modelPy=nothin, offsets=0f0, pe=nothing)
    dt_Comp = get_dt(model)
    # Python mode
    isnothing(modelPy) && (modelPy = devito_model(model, options))
    # Interpolate input data to computational grid
    d_data = time_resample(make_input(residual), residual.geometry, dt_Comp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(residual.geometry[1], modelPy.shape)

    ev, _ = backprop(modelPy, d_data, rec_coords, Q, options.space_order; pe=pe)
    inds = Int64.(offsets ./ model.d[1])
    g = combine(eu, ev, inds, options.IC == "isic")
    return remove_padding(g.data, modelPy.padsizes, offsets; true_adjoint=options.sum_padding)
end


remove_padding(gradient, nb, ::Number; true_adjoint::Bool=false) = remove_padding(gradient, nb; true_adjoint=true_adjoint)


function remove_padding(gradient::AbstractArray{DT}, nb::NTuple{Nd, NTuple{2, Int64}}, ::Vector{DT}; true_adjoint::Bool=false) where {DT, Nd}
    no = ndims(gradient) - length(nb)
    N = size(gradient)[no+1:end]
    hd = tuple([Colon() for _=1:no]...)
    if true_adjoint
        for (dim, (nbl, nbr)) in enumerate(nb)
            diml = dim+no
            selectdim(gradient, diml, nbl+1) .+= dropdims(sum(selectdim(gradient, diml, 1:nbl), dims=diml), dims=diml)
            selectdim(gradient, diml, N[dim]-nbr) .+= dropdims(sum(selectdim(gradient, diml, N[dim]-nbr+1:N[dim]), dims=diml), dims=diml)
        end
    end
    out = gradient[hd..., [nbl+1:nn-nbr for ((nbl, nbr), nn) in zip(nb, N)]...]
    return out
end