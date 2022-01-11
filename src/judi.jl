import JUDI: lsrtm_objective, time_modeling, fwi_objective
import JUDI: judiJacobian, adjbornop, bornop, judiAbstractJacobian, subsample

# Model smoother
function smooth(m::Model; sigma=3)
    new_model = deepcopy(m)
    new_model.m.data[:] = imfilter(m.m.data, Kernel.gaussian(sigma))
    return new_model
end

SincInterpolation(Y, S, Up) = sinc.( (Up .- S') ./ (S[2] - S[1]) ) * Y

function time_resample_data(d::judiVector, dt_new)
    return time_resample_data(d.data[1], dt_new, d.geometry.dt[1], d.geometry.t[1])
end

time_resample_data(d::SeisCon, dt_new, dt_old, t) = time_resample_data(Float32.(d[1].data), dt_new, dt_old, t)

function time_resample_data(d::Array, dt_new, dt_old, t)
    if dt_new == dt_old
        return d
    else
        timeAxis = 0:dt_old:t
        timeInterp = 0:dt_new:t
        dataInterp = Float32.(SincInterpolation(d, timeAxis, timeInterp))

        return dataInterp
    end
end

Base.getindex(g::GeometryIC, i::Int64) = subsample(g, i)

function get_pad(model::Model)
    pm = load_pymodel()
    modelPy = pm."Model"(origin=model.o, spacing=model.d, shape=model.n,
                         m=.44, nbl=model.nb)
    return modelPy.padsizes
end

function get_model(src_geom::Geometry, rec_geom::Geometry, model::Model, options::Options)
    ~options.limit_m && return model
    return limit_model_to_receiver_area(src_geom, rec_geom, deepcopy(model), options.buffer_size)[1]
end

function get_model(src_geom::Geometry, rec_geom::Geometry, model::Model, options::Options, dm)
    ~options.limit_m && return (model, dm)
    return limit_model_to_receiver_area(src_geom, rec_geom, deepcopy(model), options.buffer_size; pert=dm)
end

# Time modeling 
function time_modeling(model::Model, q::judiVector, residual::judiVector, dobs::judiVector, options::Options, ps::Integer)
    q.geometry = Geometry(q.geometry)
    residual.geometry = Geometry(residual.geometry)
    dobs.geometry = Geometry(dobs.geometry)

    model_loc = get_model(q.geometry, residual.geometry, model, options)
    modelPy = devito_model(model_loc, options)
    d0, Q, eu, I = forward(model_loc, q, dobs; ps=ps, options=options, modelPy=modelPy)
    ge = backprop(model_loc, residual, Q, eu; options=options, modelPy=modelPy)
    options.limit_m && (ge = extend_gradient(model, model_loc, ge))
    options.limit_m && (I = extend_gradient(model, model_loc, I))
    return PhysicalParameter(ge, model.d, model.o), PhysicalParameter(I, model.d, model.o)
end

# FWI and lsrtm function

function fwi_objective(model::Model, q::judiVector, dobs::judiVector, options::Options, ps::Integer)
    dobs = get_data(dobs)
    normalize!(dobs)
    q.geometry = Geometry(q.geometry)
    dobs.geometry = Geometry(dobs.geometry)

    model_loc = get_model(q.geometry, dobs.geometry, model, options)
    modelPy = devito_model(model_loc, options)
    d0, Q, eu, I = forward(model_loc, q, dobs; ps=ps, options=options, modelPy=modelPy)
    normalize!(d0)
    residual = d0 - dobs
    ge = backprop(model_loc, residual, Q, eu; options=options, modelPy=modelPy)
    options.limit_m && (ge = extend_gradient(model, model_loc, ge))
    options.limit_m && (I = extend_gradient(model, model_loc, I))

    obj = loss(d0, residual)
    return obj, PhysicalParameter(ge, model.d, model.o), PhysicalParameter(I, model.d, model.o)
end

# LSRTM

function lsrtm_objective(model::Model, q::judiVector, dobs::judiVector, dm::Union{Array, PhysicalParameter}, options::Options, nlind::Bool, ps::Integer)
    q.geometry = Geometry(q.geometry)
    dobs.geometry = Geometry(dobs.geometry)

    model_loc, dm = get_model(q.geometry, dobs.geometry, model, options, dm)
    modelPy = devito_model(model_loc, options; dm=dm)
    dnl, dl, Q, eu, I = born(model_loc, q, dobs, dm; ps=ps, options=options, modelPy=modelPy)
    residual = nlind ? dl - (dobs - dnl) : dl - dobs
    ge = backprop(model_loc, residual, Q, eu; options=options, modelPy=modelPy)
    options.limit_m && (ge = extend_gradient(model, model_loc, ge))
    options.limit_m && (I = extend_gradient(model, model_loc, I))
    return .5f0*norm(residual)^2, PhysicalParameter(ge, model.d, model.o), PhysicalParameter(I, model.d, model.o)
end

#########################

mutable struct judiJacobianP{DDT, RDT} <: judiAbstractJacobian{DDT, RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    source::judiVector
    dobs::judiVector
    ps::Integer
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end

Base.getproperty(J::judiJacobianP, sym::Symbol) = sym == :recGeometry ? J.dobs.geometry : getfield(J, sym)

set_r!(J::judiJacobianP, ps::Integer) = (J.ps = ps)
set_ps!(J::judiJacobianP, ps::Integer) = (J.ps = ps)

# Jacobian with probing
function JUDI.judiJacobian(F::judiPDEfull, source::judiVector, ps::Int64, dobs::judiVector; DDT::DataType=Float32, RDT::DataType=DDT, options=nothing)
    # JOLI wrapper for nonlinear forward modeling
    compareGeometry(F.srcGeometry, source.geometry) == true || judiJacobianException("Source geometry mismatch")
    (DDT == Float32 && RDT == Float32) || throw(judiJacobianException("Domain and range types not supported"))
    m = typeof(F.recGeometry) <: GeometryOOC ? sum(F.recGeometry.nsamples) : sum([length(F.recGeometry.xloc[j])*F.recGeometry.nt[j] for j=1:source.nsrc])
    n = F.info.n

    isnothing(options) && (options = F.options)
    return J = judiJacobianP{Float32,Float32}("linearized wave equation", m, n, 
        F.info, F.model, source, dobs, ps, options, bornop, adjbornop)
end


judiJacobian(J::judiJacobianP{DDT,RDT}; name=J.name, m=J.m, n=J.n, info=J.info, model=J.model, source=J.source,
     dobs=J.dobs, ps=J.ps, opt=J.options, fop=J.fop, fop_T=J.fop_T) where {DDT, RDT} =
            judiJacobianP{DDT,RDT}(name, m, n, info, model, source, dobs, ps, opt, fop, fop_T)

# Subsample Jacobian
function subsample(J::judiJacobianP{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    nsrc = typeof(srcnum) <: Int ? 1 : length(srcnum)
    info = Info(J.info.n, nsrc, J.info.nt[srcnum])
    dobs_loc = J.dobs[srcnum]
    m = dobs_loc.m

    return judiJacobian(J; m=m, info=info, source=J.source[srcnum], dobs=dobs_loc, ps=J.ps, opt=subsample(J.options, srcnum))
end

function adjbornop(J::judiJacobianP, w)
    srcnum = 1:J.info.nsrc
    rtm, I = time_modeling(J.model, J.source, w, srcnum, J.dobs, J.options, J.ps)
    J.model.params[:illum] = I
    return rtm
end

function bornop(J::judiJacobianP, v)
    srcnum = 1:J.info.nsrc
    return time_modeling(J.model, J.source.geometry, J.source.data, J.dobs.geometry, nothing, v, srcnum, 'J', 1, J.options)
end


###### JUDI fix to be moved there
Base.similar(x::Array{T, N}, m::Model) where {T<:Real, N} = zeros(T, m.n)