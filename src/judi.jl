
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

# Jacobian with probing
function JUDI.judiJacobian(F::judiPDEfull, source::judiVector, ps::Int64, dobs::judiVector; DDT::DataType=Float32, RDT::DataType=DDT, options=nothing)
    # JOLI wrapper for nonlinear forward modeling
        compareGeometry(F.srcGeometry, source.geometry) == true || judiJacobianException("Source geometry mismatch")
        (DDT == Float32 && RDT == Float32) || throw(judiJacobianException("Domain and range types not supported"))
        if typeof(F.recGeometry) == GeometryOOC
            m = sum(F.recGeometry.nsamples)
        else
            m = 0
            for j=1:F.info.nsrc m += length(F.recGeometry.xloc[j])*F.recGeometry.nt[j] end
        end
        n = F.info.n
        srcnum = F.info.nsrc > 1 ? (1:F.info.nsrc) : 1

        isnothing(options) && (options = F.options)

        return J = judiJacobian{Float32,Float32}("linearized wave equation", m, n, 
            F.info, F.model, F.srcGeometry, F.recGeometry, source.data, options,
            v -> JUDI.time_modeling(F.model, F.srcGeometry, source.data, F.recGeometry, nothing, v, srcnum, 'J', 1, options),
            w -> time_modeling(F.model, source, w, srcnum, ps, dobs, options)
        )
end

# Time modeling 
function time_modeling(model::Model, q::judiVector, dat::judiVector, srcnum::UnitRange{Int64}, ps::Integer, dobs::judiVector, options)
    p = default_worker_pool()

    # Process shots from source channel asynchronously
    results = pmap(j -> time_modeling(model, q[j], dat[j], ps, dobs[j], subsample(options, j)), p, srcnum)
    argout1 = results[1]
    for j=2:numSources
        argout1 += results[j]
    end
    return argout1
end


function time_modeling(model::Model, q::judiVector, residual::judiVector, ps::Integer, dobs::judiVector, options)
    d0, Q, eu = forward(model, q, dobs; ps=ps, options=options, residual=residual)
    ev = backprop(model, residual, Q; options=options)
    ge = combine_probes(ev, eu, model)
    return PhysicalParameter(ge, model.d, model.o)
end

time_modeling(model::Model, q::judiVector, dat::judiVector, srcnum::Int64, ps::Integer, dobs::judiVector, options) = time_modeling(model, q, dat, ps, dobs, options)

# FWI and lsrtm function
function fwi_objective(model::Model, source::judiVector, dObs::judiVector, ps::Integer; options=Options())
    
    # fwi_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.
    p = default_worker_pool()
    results = pmap(j -> fwi_objective_ps(model, source[j], dObs[j], ps, subsample(options, j)), p, 1:dObs.nsrc)
    
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
    d0, Q, eu = forward(model, q, dobs; ps=ps, options=options)
    residual = d0 - dobs
    ev = backprop(model, residual, Q; options=options)
    ge = combine_probes(ev, eu, model)
    return .5*norm(residual)^2, PhysicalParameter(ge, model.d, model.o)
end
