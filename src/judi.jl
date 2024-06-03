#########################
struct judiJacobianP{D, O, FT} <: judiAbstractJacobian{D, O, FT}
    m::AbstractSize
    n::AbstractSize
    F::FT
    q::judiMultiSourceVector
    r::Integer
    probing_mode::Symbol
    dobs::judiVector
    offsets::Union{D, Vector{D}}
end

_accepted_modes = (:QR, :Rademacher, :Gaussian, :hutchpp)

# Jacobian with probing
function judiJacobian(F::judiComposedPropagator{D, O}, q::judiMultiSourceVector, r::Integer, dobs::judiMultiSourceVector;
                      options=nothing, mode=:QR, offsets=0f0) where {D, O}
    mode ∈ _accepted_modes || throw(ArgumentError("Probing vector mode unrecognized, must be in $(_accepted_modes)"))
    update!(F.F.options, options)
    return judiJacobianP{D, :born, typeof(F)}(F.m, space(size(F.model)), F, q, r, mode, dobs, asvec(offsets))
end

asvec(x::T) where T<:Number = x
asvec(x) = collect(x)

adjoint(J::judiJacobianP{D, O, FT}) where {D, O, FT} = judiJacobianP{D, adjoint(O), FT}(J.n, J.m, J.F, J.q, J.r, J.probing_mode, J.dobs, J.offsets)
getindex(J::judiJacobianP{D, O, FT}, i) where {D, O, FT} = judiJacobianP{D, O, FT}(J.m[i], J.n[i], J.F[i], J.q[i], J.r, J.probing_mode, J.dobs[i], J.offsets)

process_input_data(::judiJacobianP{D, :born, FT}, q::dmType{D}) where {D<:Number, FT} = q

function make_input(J::judiJacobianP{D, :born, FT}, q::dmType{D}) where {D<:Number, FT}
    srcGeom, srcData = make_src(J.q, J.F.qInjection)
    return srcGeom, srcData, J.F.rInterpolation.data[1], nothing, reshape(q, size(J.model))
end

function multi_src_propagate(F::judiJacobianP{D, O, FT}, q::AbstractArray{D}) where {D<:Number, FT, O}
    GC.gc(true)
    dv.clear_cache()

    q = process_input_data(F, q)
    # Number of sources and init result
    nsrc = get_nsrc(F, q)
    pool = JUDI._worker_pool()
    arg_func = i -> (F[i], JUDI.src_i(F, q, i))
    # Distribute source
    res = JUDI.run_and_reduce(propagate, pool, nsrc, arg_func)
    if length(F.offsets) < 2    
        res = JUDI._project_to_physical_domain(res, F.model)
    end
    res = JUDI.update_illum(res, F)
    res = JUDI.as_vec(res, Val(F.options.return_array))
    return res
end

function propagate(J::judiJacobianP{D, :adjoint_born, FT}, residual::AbstractArray{T}) where {T, D, FT}
    GC.gc(true)
    dv.clear_cache()

    J.q.geometry = Geometry(J.q.geometry)
    residual.geometry = Geometry(residual.geometry)
    J.dobs.geometry = Geometry(J.dobs.geometry)

    model_loc = get_model(J.q.geometry, residual.geometry, J.model, J.options)
    modelPy = devito_model(model_loc, J.options)
    _, Q, eu = forward(model_loc, J.q, J.dobs; ps=J.r, options=J.options, modelPy=modelPy, mode=J.probing_mode)
    ge = backprop(model_loc, J.q, residual, Q, eu; options=J.options, modelPy=modelPy, offsets=J.offsets)
    J.options.limit_m && (ge = extend_gradient(J.model, model_loc, ge))

    if isa(J.offsets, Number)
        g = PhysicalParameter(ge, spacing(J.model), origin(J.model))
    else
        ncig = (size(ge, 1), size(model_loc)...)
        o = (minimum(J.offsets), origin(model_loc)...)
        d = (abs(diff(J.offsets)[1]), spacing(model_loc)...)
        g = PhysicalParameter(ncig, d, o, ge)
    end
    return g
end

fwi_objective(model::MTypes, q::Dtypes, dobs::Dtypes, r::Integer; options=Options(), offsets=0f0) =
    fwi_objective(model, q, dobs; options=options, r=r, offsets=offsets, func=multi_src_fg_r)

lsrtm_objective(model::MTypes, q::Dtypes, dobs::Dtypes, dm::dmType, r::Integer; options=Options(), nlind=false, offsets=0f0) =
    lsrtm_objective(model, q, dobs, dm; options=options, nlind=nlind, r=r, offsets=offsets, func=multi_src_fg_r)


function multi_src_fg_r(model::AbstractModel, q::judiVector, dobs::judiVector, dm, options::JUDIOptions;
                      nlind::Bool=false, lin::Bool=false, misfit::Function=mse, illum::Bool=false, r::Integer=32,
                      data_precon=nothing, model_precon=LinearAlgebra.I)
    GC.gc(true)
    dv.clear_cache()

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
    ge = backprop(model_loc, q, residual, Q, eu; options=options, modelPy=modelPy, offsets=offsets)
    options.limit_m && (ge = extend_gradient(model, model_loc, ge))
    return Ref{Float32}(.5f0*norm(residual)^2), PhysicalParameter(ge, spacing(model), origin(model))
end
