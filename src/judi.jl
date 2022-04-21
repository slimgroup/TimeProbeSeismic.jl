#########################
struct judiJacobianP{D, O, FT} <: judiAbstractJacobian{D, O, FT}
    m::AbstractSize
    n::AbstractSize
    F::FT
    q::judiMultiSourceVector
    r::Integer
    dobs::judiVector
end

# Jacobian with probing
function judiJacobian(F::judiComposedPropagator{D, O}, q::judiMultiSourceVector, r::Integer, dobs::judiMultiSourceVector; options=nothing) where {D, O}
    update!(F.F.options, options)
    return judiJacobianP{D, :born, typeof(F)}(F.m, space(F.model.n), F, q, r, dobs)
end

adjoint(J::judiJacobianP{D, O, FT}) where {D, O, FT} = judiJacobianP{D, adjoint(O), FT}(J.n, J.m, J.F, J.q, J.r, J.dobs)
getindex(J::judiJacobianP{D, O, FT}, i) where {D, O, FT} = judiJacobianP{D, O, FT}(J.m[i], J.n[i], J.F[i], J.q[i], J.r, J.dobs[i])

function propagate(J::judiJacobianP{D, :adjoint_born, FT}, residual::AbstractArray{T}) where {T, D, FT}
    J.q.geometry = Geometry(J.q.geometry)
    residual.geometry = Geometry(residual.geometry)
    J.dobs.geometry = Geometry(J.dobs.geometry)

    model_loc = get_model(J.q.geometry, residual.geometry, J.model, J.options)
    modelPy = devito_model(model_loc, J.options)
    _, Q, eu = forward(model_loc, J.q, J.dobs; ps=J.r, options=J.options, modelPy=modelPy)
    ge = backprop(model_loc, residual, Q, eu; options=J.options, modelPy=modelPy)
    options.limit_m && (ge = extend_gradient(J.model, model_loc, ge))
    return PhysicalParameter(ge, model.d, model.o)
end