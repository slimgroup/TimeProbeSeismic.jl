
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

# Time modeling 
function time_modeling(model::Model, q::judiVector, dat::judiVector, srcnum::UnitRange{Int64}, ps::Integer, dobs::judiVector, options)
    p = default_worker_pool()

    # Process shots from source channel asynchronously
    results = judipmap(j -> time_modeling(model, q[j], dat[j], ps, dobs[j], subsample(options, j)), p, srcnum)
    argout1 = results[1]
    for j=2:length(srcnum)
        argout1 += results[j]
    end
    return argout1
end


function time_modeling(model::Model, q::judiVector, residual::judiVector, ps::Integer, dobs::judiVector, options)
    modelPy = devito_model(model, options)
    d0, Q, eu = forward(model, q, dobs; ps=ps, options=options, modelPy=modelPy)
    ge = backprop(model, residual, Q, eu; options=options, modelPy=modelPy)
    return PhysicalParameter(ge, model.d, model.o)
end

time_modeling(model::Model, q::judiVector, dat::judiVector, srcnum::Int64, ps::Integer, dobs::judiVector, options) = time_modeling(model, q, dat, ps, dobs, options)

# FWI and lsrtm function
function fwi_objective(model::Model, source::judiVector, dObs::judiVector, ps::Integer; options=Options())
    
    # fwi_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.
    p = default_worker_pool()
    results = judipmap(j -> fwi_objective_ps(model, source[j], dObs[j], ps, subsample(options, j)), p, 1:dObs.nsrc)
    
    # Collect and reduce gradients
    objective = 0f0
    gradient = PhysicalParameter(zeros(Float32, model.n), model.d, model.o)

    for j=1:dObs.nsrc
        gradient .+= results[j][2]
        objective += results[j][1]
    end
    # first value corresponds to function value, the rest to the gradient
    return objective, gradient
end


function fwi_objective_ps(model::Model, q::judiVector, dobs::judiVector, ps::Integer, options)
    modelPy = devito_model(model, options)
    d0, Q, eu = forward(model, q, dobs; ps=ps, options=options, modelPy=modelPy)
    residual = d0 - dobs
    ge = backprop(model, residual, Q, eu; options=options, modelPy=modelPy)
    return .5f0*norm(residual)^2, PhysicalParameter(ge, model.d, model.o)
end

# LSRTM
function lsrtm_objective(model::Model, source::judiVector, dObs::judiVector, dm, ps; options=Options(), nlind=false)
    # fwi_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.
    p = default_worker_pool()
    results = judipmap(j -> lsrtm_objective_ps(model, source[j], dObs[j], dm, ps; options=subsample(options, j), nlind=nlind), p, 1:dObs.nsrc)
    # Collect and reduce gradients
    objective = 0f0
    gradient = PhysicalParameter(zeros(Float32, model.n), model.d, model.o)

    for j=1:dObs.nsrc
        gradient .+= results[j][2]
        objective += results[j][1]
    end
    # first value corresponds to function value, the rest to the gradient
    return objective, gradient
end

function lsrtm_objective_ps(model::Model, q::judiVector, dobs::judiVector, dm, ps::Integer; options=Options(), nlind=false)
    modelPy = devito_model(model, options)
    dnl, dl, Q, eu = born(model, q, dobs, dm; ps=ps, options=options, modelPy=modelPy)
    residual = nlind ? dl - (dobs - dnl) : dl - dobs
    ev = backprop(model, residual, Q, eu; options=options, modelPy=modelPy)
    return .5f0*norm(residual)^2, PhysicalParameter(ge, model.d, model.o)
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

set_ps!(J::judiJacobianP, ps::Integer) = (J.ps = ps)
# Jacobian with probing
function judiJacobian(F::judiPDEfull, source::judiVector, ps::Int64, dobs::judiVector; DDT::DataType=Float32, RDT::DataType=DDT, options=nothing)
    # JOLI wrapper for nonlinear forward modeling
    compareGeometry(F.srcGeometry, source.geometry) == true || judiJacobianException("Source geometry mismatch")
    (DDT == Float32 && RDT == Float32) || throw(judiJacobianException("Domain and range types not supported"))
    m = typeof(F.recGeometry) == GeometryOOC ? sum(F.recGeometry.nsamples) : sum([length(F.recGeometry.xloc[j])*F.recGeometry.nt[j] for j=1:source.nsrc])
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
    return time_modeling(J.model, J.source, w, J.ps, J.dobs, J.options)
end

function bornop(J::judiJacobianP, v)
    srcnum = 1:J.info.nsrc
    return time_modeling(J.model, J.source.geometry, J.source.data, J.dobs.geometry, nothing, v, srcnum, 'J', 1, J.options)
end