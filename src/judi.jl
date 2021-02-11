

# Model smoother
function smooth(m::Model; sigma=3)
    new_model = deepcopy(m)
    new_model.m.data[:] = imfilter(m.m.data, Kernel.gaussian(sigma))
    return new_model
end

SincInterpolation(Y, S, Up) = sinc.( (Up .- S') ./ (S[2] - S[1]) ) * Y

function time_resample_data(d::judiVector, dt_new)

    if dt_new==geometry_in.dt[1]
        return d.data
    else
        data, geom = d.data[1], d.geometry[1]
        numTraces = size(data,2)
        timeAxis = 0:geom.dt[1]:geom.t[1]
        timeInterp = 0:dt_new:geom.t[1]
        dataInterp = Float32.(SincInterpolation(data, timeAxis, timeInterp))

        return dataInterp
    end
end
