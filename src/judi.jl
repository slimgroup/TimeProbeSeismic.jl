

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
        numTraces = size(data, 2)
        timeAxis = 0:dt_old:t
        timeInterp = 0:dt_new:t
        dataInterp = Float32.(SincInterpolation(data, timeAxis, timeInterp))

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