time_resample(q::judiVector, dt) = begin @assert q.nsrc == 1 ;time_resample(make_input(q), q.geometry.dt[1], dt) end

#########################
struct judiJacobianP{D, O, FT} <: judiAbstractJacobian{D, O, FT}
    m::AbstractSize
    n::AbstractSize
    F::FT
    q::judiMultiSourceVector
    r::Integer
    mode::Symbol
    dobs::judiVector
end

# Jacobian with probing
function judiJacobian(F::judiComposedPropagator{D, O}, q::judiMultiSourceVector, r::Integer, dobs::judiMultiSourceVector; options=nothing, mode=:QR) where {D, O}
    mode ∈ [:QR, :Rademacher] || throw(ArgumentError("Probing vector mode unrecognized, must be `:QR` or `:Rademacher`"))
    update!(F.F.options, options)
    return judiJacobianP{D, :born, typeof(F)}(F.m, space(F.model.n), F, q, r, mode, dobs)
end

adjoint(J::judiJacobianP{D, O, FT}) where {D, O, FT} = judiJacobianP{D, adjoint(O), FT}(J.n, J.m, J.F, J.q, J.r, J.mode, J.dobs)
getindex(J::judiJacobianP{D, O, FT}, i) where {D, O, FT} = judiJacobianP{D, O, FT}(J.m[i], J.n[i], J.F[i], J.q[i], J.r, J.mode, J.dobs[i])

process_input_data(::judiJacobianP{D, :born, FT}, q::dmType{D}) where {D<:Number, FT} = q
function make_input(J::judiJacobianP{D, :born, FT}, q::dmType{D}) where {D<:Number, FT}
    srcGeom, srcData = make_src(J.q, J.F.qInjection)
    return srcGeom, srcData, J.F.rInterpolation.data[1], nothing, reshape(q, J.model.n)
end 

function propagate(J::judiJacobianP{D, :adjoint_born, FT}, residual::AbstractArray{T}) where {T, D, FT}
    J.q.geometry = Geometry(J.q.geometry)
    residual.geometry = Geometry(residual.geometry)
    J.dobs.geometry = Geometry(J.dobs.geometry)

    model_loc = get_model(J.q.geometry, residual.geometry, J.model, J.options)
    modelPy = devito_model(model_loc, J.options)
    _, Q, eu = forward(model_loc, J.q, J.dobs; ps=J.r, options=J.options, modelPy=modelPy, mode=J.mode)
    ge = backprop(model_loc, residual, Q, eu; options=J.options, modelPy=modelPy)
    J.options.limit_m && (ge = extend_gradient(J.model, model_loc, ge))
    return PhysicalParameter(ge./J.r, J.model.d, J.model.o)
end

fwi_objective(model::MTypes, q::Dtypes, dobs::Dtypes, r::Integer; options=Options()) =
    fwi_objective(model, q, dobs; options=options, r=r)

lsrtm_objective(model::MTypes, q::Dtypes, dobs::Dtypes, dm::dmType, r::Integer; options=Options(), nlind=false) =
    lsrtm_objective(model, q, dobs, dm; options=options, nlind=nlind, r=r)

function multi_src_fg(model::Model, q::judiVector, dobs::judiVector, dm, options::JUDIOptions, nlind::Bool, lin::Bool, r::Integer)
    q.geometry = Geometry(q.geometry)
    dobs.geometry = Geometry(dobs.geometry)

    model_loc, dm = get_model(q.geometry, dobs.geometry, model, options, dm)
    modelPy = devito_model(model_loc, options, dm)
    if lin
        dnl, dl, Q, eu = born(model_loc, q, dobs, dm; ps=r, options=options, modelPy=modelPy)
    else
        dl, Q, eu = forward(model_loc, q, dobs; ps=r, options=options, modelPy=modelPy)
    end
    residual = nlind ? dl - (dobs - dnl) : dl - dobs
    ge = backprop(model_loc, residual, Q, eu; options=options, modelPy=modelPy)
    options.limit_m && (ge = extend_gradient(model, model_loc, ge))
    return Ref{Float32}(.5f0*norm(residual)^2), PhysicalParameter(ge, model.d, model.o)
end