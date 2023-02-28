###########
# Overdot #
###########

# This is a draft, or maybe even an experiment. Things are implemented in a variety of ways
# to see which work out best. Don't expect a ton of consistency if you read through this
# stuff. Expect question marks and TODOs.

module Overdot

# TODO
# [ ] Regularly-sampled systems need to remember to set t_next in the f0_out. That's easy to
#     forget, and it's more-or-less redundant with the subsequent use of zohx/y.
# [ ] Implement an adaptive-step integrator for general utility.
# [ ] Add a multi-threaded batch function.
# [ ] Add a tool to save a time history to disk in a portable format.
# [ ] Make it so that time information can be specified as Float64 and converted to a
#     rational. Typing rationals is just kind of a pain when the end user is thinking in
#     floats.

using Random

export simulate, zohx, zohxy, zoh

#############
# Utilities #
#############

# We want to accept a variety of inputs, but we need a rational. Is it ok to call
# rationalize rather than relying on a straight conversion?
Base.@inline maybe_rationalize(x::AbstractFloat)::Rational{Int64}   = rationalize(x)
Base.@inline maybe_rationalize(x::Rational{Int64})::Rational{Int64} = x
Base.@inline maybe_rationalize(x)::Rational{Int64}                  = Rational{Int64}(x)

################
# Time History #
################

"""
Stores the time-history of constants, states, derivatives, and outputs for a model.
"""
struct ModelHistory{C,TC,XC,YC,TD,XD,YD,M}
    constants::C
    tc::TC
    xc::XC
    xc_dot::XC # same type as xc
    yc::YC
    td::TD
    xd::XD
    yd::YD
    models::M
end

# Records the continuous-time part of the time history from the output of fc.
function recordc(t, h, fc_out::T, x) where {T <: NamedTuple}
    push!(h.tc, float(t))
    for k in keys(h.xc)
        push!(getfield(h.xc, k), getfield(x, k))
        push!(getfield(h.xc_dot, k), getfield(fc_out.overdot, k))
    end
    if haskey(fc_out, :yc)
        push!(h.yc, fc_out.yc)
    end
    if haskey(fc_out, :models)
        for k in keys(fc_out.models)
            recordc(t, getfield(h.models, k), getfield(fc_out.models, k), getfield(x, k))
        end
    end
end

# When a "model" is actually a tuple or array, we need to expand it.
function recordc(t, h, fc_out::T, x) where {T <: Union{Tuple, AbstractArray}}
    for (h_i, fc_out_i, x_i) in zip(h, fc_out, x)
        recordc(t, h_i, fc_out_i, x_i)
    end
end

# Record the discrete-time part of the time history from the output of fd.
function recordd(t, h, fd_out::T) where {T <: NamedTuple}
    if !haskey(fd_out, :record) || fd_out.record == true
        push!(h.td, float(t))
        if haskey(fd_out, :update)
            for k in keys(h.xd)
                push!(getfield(h.xd, k), getfield(fd_out.update, k))
            end
        end
        if haskey(fd_out, :yd)
            push!(h.yd, fd_out.yd)
        end
    end
    if haskey(fd_out, :models)
        for k in keys(fd_out.models)
            recordd(t, getfield(h.models, k), getfield(fd_out.models, k))
        end
    end
end

# WHen a "model" is actually a tuplue or array, we need to expand it.
function recordd(t, h, fd_out::T) where {T <: Union{Tuple, AbstractArray}}
    for (h_i, fd_out_i) in zip(h, fd_out)
        recordd(t, h_i, fd_out_i)
    end
end

####################
# Zero-Order Holds #
####################

# These are all bonus utilities for the user. Overdot doesn't do anything with these.

"""
This evaluates the given function when it's trigger time and otherwise retrieves the
"last value" from the specified field of the model. It's useful for little things. E.g.:

```
# Set up ξ to only take a draw when required:
wd = (;
    ξ = (t, model) -> zoh(() -> randn(rng), t, model.sample_rate, model, :ξ)
)
```
"""
Base.@inline function zoh(f::Function, t::Real, sample_rate::Real, thing, field::Symbol)
    if mod(t, inv(sample_rate)) == 0
        return f()
    else
        return getproperty(thing, field)
    end
end

"""
This evaluates the given function when it's trigger time, returning the results in an
"update" block along with `record` and `t_next` as appropriate to model a regularly-sampled
system. It can work for multiple states.
"""
Base.@inline function zohx(f::Function, t::Real, sample_rate::Real, thing::T, fields::NTuple{N,Symbol}) where {T,N}

    # It's important to inline this function so that the fields are known exactly and not
    # just as a tuple.

    # Create the output type.
    OutType = NamedTuple{fields, Tuple{(fieldtype(T, field) for field in fields)...}}

    # See if it's time to trigger and, if so, run the function. Otherwise, get the last
    # values. Both forms return named tuples in the same order as `fields`.
    if mod(t, inv(sample_rate)) == 0
        return (;
            update = OutType(f()),
            record = true,
            t_next = t + inv(sample_rate),
        )
    else
        return (;
            update = OutType(getproperty(thing, field) for field in fields),
            record = false,
            t_next = ceil(t * sample_rate) / sample_rate,
        )
    end

end

"""
This evaluates the given function when it's trigger time, returning the results in an
"update" block along with `record` and `t_next` and an "yd" block, as appropriate to model a
system with regularly-sampled state and outputs. It can work for multiple states. The
"last output" (yd_field) is expected to be available in the state. There's no need to
include the yd_field in the tuple of xd_fields.
"""
Base.@inline function zohxy(f::Function, t::Real, sample_rate::Real, thing::T, xd_fields::NTuple{N,Symbol}, yd_field::Symbol) where {T,N}

    # It's important to inline this function so that the fields are known exactly and not
    # just as a tuple.

    # Create the output type.
    OutType = NamedTuple{(xd_fields..., yd_field), Tuple{(fieldtype(T, field) for field in xd_fields)..., fieldtype(T, yd_field)}}

    # See if it's time to trigger and, if so, run the function. Otherwise, get the last
    # values. Both forms return named tuples in the same order as `fields`.
    if mod(t, inv(sample_rate)) == 0
        out = OutType(f())
        return (;
            update = out,
            yd     = getproperty(out, yd_field),
            record = true,
            t_next = t + inv(sample_rate),
        )
    else
        return (;
            update = OutType(getproperty(thing, field) for field in (xd_fields..., yd_field)),
            yd     = getproperty(thing, yd_field),
            record = false,
            t_next = ceil(t * sample_rate) / sample_rate,
        )
    end

end

#############
# Recursion #
#############

# This first runs recurisvely on all sub-models, and then it runs the given function with
# the current model and the results of the process for all sub-models.
function recursively(f::F, model::NamedTuple) where {F}
    downstream = hasproperty(model, :models) ? (recursively(f, m_i) for m_i in model.models) : (;)
    return f(model, downstream)
end

function recursively(f::F, models::Tuple) where {F}
    return ((recursively(f, m_i) for m_i in models)...,)
end

function recursively(f::F, models::Vector) where {F}
    return [recursively(f, m_i) for m_i in models]
end

######################
# State Manipulation #
######################

# Given a NamedTuple with the appropriate names (such as the wd block of an f0_out), this
# copies the appropriate values from the prior into a new NamedTuple. In this case, the
# types of the values are known from the spec (e.g., xc or xd).
Base.@inline @generated function copy_from_prior_with_types(spec::NamedTuple{N,T}, prior::PT) where {N,T,PT}
    get_exprs = []
    for k in N
        push!(get_exprs, :(getproperty(prior, $(QuoteNode(k)))))
    end
    return :(NamedTuple{N,T}(($(get_exprs...),)))
end

# This is the same as the above, but the types are unknown (e.g., wc or wd).
Base.@inline @generated function copy_from_prior(spec::NamedTuple{N,T}, prior::PT) where {N,T,PT}
    get_exprs = []
    for k in N
        push!(get_exprs, :(getproperty(prior, $(QuoteNode(k)))))
    end
    return :(NamedTuple{N}(($(get_exprs...),)))
end

#############
# update_xc #
#############

# This creates an updated xc for the current model. It runs f (the update function) on
# each field of the state with corresponding fields from the derivatives in fc_outs (a
# tuple of outputs from fc, such as k1, k2, k3, k4 for RK4). It's a generated function
# so that it can use known field names/types, removing the need for allocations.
Base.@inline @generated function get_updated_xc(f, f0_out::NamedTuple{N,T}, prior, fc_outs) where {N,T}
    expr = []
    for k in N
        fc_outs_expr = []
        for i in 1:length(fc_outs.types)
            push!(fc_outs_expr, :(getproperty(fc_outs[$i].overdot, $(QuoteNode(k)))))
        end
        push!(expr, :(f(getproperty(prior, $(QuoteNode(k))), ($(fc_outs_expr...),))))
    end
    return :(NamedTuple{N,T}(($(expr...),)))
end

# This produces a new model-form by updating xc and copying everything else.
function update_xc(f, f0_out::NamedTuple{N,T}, fc_in_km1, fc_outs) where {N,T}

    # Copy/update.
    xc     = :xc     in N ? get_updated_xc(f, f0_out.xc, fc_in_km1, fc_outs) : (;)
    xd     = :xd     in N ? copy_from_prior_with_types(f0_out.xd, fc_in_km1) : (;)
    wc     = :wc     in N ? copy_from_prior(f0_out.wc, fc_in_km1) : (;)
    wd     = :wd     in N ? copy_from_prior(f0_out.wd, fc_in_km1) : (;)
    models = :models in N ? update_xc_for_submodels(f, f0_out.models, fc_in_km1, fc_outs) : (;)

    # Collect all over those NamedTuples together into one big NamedTuple with unique names.
    nt = (;
        (:constants in N ? f0_out.constants : (;))...,
        xc...,
        wc...,
        xd...,
        wd...,
        models...,
    )

    # Return either the named tuple or the user's requested type.
    if :type in N
        return f0_out.type(; nt...)
    else
        return nt
    end
end

# These just help expand when a "model" is really a tuple or vector.
Base.@inline function update_xc(f, f0_out::A, fc_in_km1::B, fc_outs) where {A <: Tuple, B <: Tuple}
    return ((update_xc(f, f0_out_i, fc_in_km1_i, fc_outs_i) for (f0_out_i, fc_in_km1_i, fc_outs_i...) in zip(f0_out, fc_in_km1, fc_outs...))...,)
end
Base.@inline function update_xc(f, f0_out::A, fc_in_km1::B, fc_outs) where {A <: Vector, B <: Vector}
    return [update_xc(f, f0_out_i, fc_in_km1_i, fc_outs_i) for (f0_out_i, fc_in_km1_i, fc_outs_i...) in zip(f0_out, fc_in_km1, fc_outs...)]
end

# This is how we implement recursion on the sub-models for the update_xc process.
Base.@inline function update_xc_for_submodels(f, f0_out_models::NamedTuple{N,T}, prior, fc_outs) where {N,T}

    # If there's no "models" output in fc_outs, then clearly there's nothing to update here.
    # Just copy directly from the prior.
    if !haskey(fc_outs[1], :models)
        return NamedTuple{N}(getproperty(prior, m) for m in N)
    end

    # There's a "models" output. For each model key, see if it exists in the fc_outs. If
    # not, return the appropriate field from the prior. If so, call update_xc with the 
    # appropriate model fields from f0_out, the prior, and each of the fc_outs.
    return NamedTuple{N}(
        haskey(fc_outs[1].models, m) ?
            update_xc(
                f,
                getproperty(f0_out_models, m),
                getproperty(prior, m),
                ((getproperty(fc_out.models, m) for fc_out in fc_outs)...,),
            ) 
        : getproperty(prior, m)
        for m in N)

end

#############
# update_xd #
#############

# This creates an updated xd for the current model. It's only run if there's discrete state,
# and if there's discrete state, we expect to always find an update for it. But we might
# possibly allow fd to "skip" updates.
Base.@inline @generated function get_updated_xd(f0_out_xd::NamedTuple{N,T}, prior, fd_out) where {N,T}
    expr = []
    for k in N
        push!(expr, :(getproperty(fd_out.update, $(QuoteNode(k)))))
    end
    return :(NamedTuple{N,T}(($(expr...),)))
end

# This produces a new model-form by updating xd and copying everything else.
function update_xd(f0_out::NamedTuple{N,T}, fd_in::B, fd_out::C) where {N, T, B, C}

    # Copy/update.
    xc     = :xc     in N ? copy_from_prior_with_types(f0_out.xc, fd_in) : (;)
    xd     = :xd     in N ? get_updated_xd(f0_out.xd, fd_in, fd_out) : (;)
    wd     = :wd     in N ? copy_from_prior(f0_out.wd, fd_in) : (;)
    models = :models in N ? update_xd_for_submodels(f0_out.models, fd_in, fd_out) : (;)

    # Collect all over those NamedTuples together into one big NamedTuple with unique names.
    nt = (;
        (:constants in N ? f0_out.constants : (;))...,
        xc...,
        xd...,
        wd...,
        models...,
    )
    
    # Return either the named tuple or the user's requested type.
    if :type in N
        return f0_out.type(; nt...)
    else
        return nt
    end
end

# Tuple/array expanders
Base.@inline function update_xd(f0_out::A, fd_in::B, fd_out::C) where {A <: Tuple, B <: Tuple, C}
    return ((update_xd(args...) for args in zip(f0_out, fd_in, fd_out))...,)
end
Base.@inline function update_xd(f0_out::A, fd_in::B, fd_out::C) where {A <: Vector, B <: Vector, C}
    return [update_xd(args...) for args in zip(f0_out, fd_in, fd_out)]
end

# This is how we implement recursion on the sub-models for the fd process.
Base.@inline function update_xd_for_submodels(f0_out_models::NamedTuple{N,T}, prior, fd_out) where {N,T}

    # If there's no "models" output in fd_out, then clearly there's nothing to update here.
    # Just return the appropriate fields from the prior.
    if !haskey(fd_out, :models)
        return NamedTuple{N}(getproperty(prior, m) for m in N)
    end

    # There's a "models" output. For each key, see if it exists in the fd_out. If so,
    # keep updating. If not, return the appropriate field from the prior.
    return NamedTuple{N}(
        haskey(fd_out.models, m) ?
            update_xd(
                getproperty(f0_out_models, m),
                getproperty(prior, m),
                getproperty(fd_out.models, m),
            )
        : getproperty(prior, m)
        for m in N)

end

#############
# update_wd #
#############

# This takes a draw for each field of the wd block.
Base.@inline @generated function draw_wd(wd_block::NamedTuple{N,T}, fd_in, t) where {N,T}
    expr = []
    for k in N
        push!(expr, :(getproperty(wd_block, $(QuoteNode(k)))(t, fd_in)))
    end
    return :(NamedTuple{N}(($(expr...),)))
end

# This produces a new model-form by taking new discrete draws and copying everything else.
function update_wd(f0_out::NamedTuple{N,T}, fd_in::B, t) where {N, T, B}

    # Copy/update.
    xc     = haskey(f0_out, :xc)     ? copy_from_prior_with_types(f0_out.xc, fd_in) : (;)
    xd     = haskey(f0_out, :xd)     ? copy_from_prior_with_types(f0_out.xd, fd_in) : (;)
    wd     = haskey(f0_out, :wd)     ? draw_wd(f0_out.wd, fd_in, t) : (;)
    models = haskey(f0_out, :models) ? update_wd_for_submodels(f0_out.models, fd_in, t) : (;)

    # Collect all over those NamedTuples together into one big NamedTuple with unique names.
    nt = (;
        (:constants in N ? f0_out.constants : (;))...,
        xc...,
        xd...,
        wd...,
        models...,
    )
    
    # Return either the named tuple or the user's requested type.
    if :type in N
        return f0_out.type(; nt...)
    else
        return nt
    end
end

# Tuple/array expanders
Base.@inline function update_wd(f0_out::A, fd_in::B, t) where {A <: Tuple, B <: Tuple}
    return ((update_wd(f0_out_i, fd_in_i, t) for (f0_out_i, fd_in_i) in zip(f0_out, fd_in))...,)
end
Base.@inline function update_wd(f0_out::A, fd_in::B, t) where {A <: Vector, B <: Vector}
    return [update_wd(f0_out_i, fd_in_i, t) for (f0_out_i, fd_in_i) in zip(f0_out, fd_in)]
end

# This implements recursion for the update_wd process. It's a generated function because it
# was easy to write. It's not clear if this is useful.
Base.@inline @generated function update_wd_for_submodels(f0_out_models::NamedTuple{N,T}, prior, t) where {N,T}
    expr = []
    for k in N
        push!(expr, :(update_wd(getproperty(f0_out_models, $(QuoteNode(k))), getproperty(prior, $(QuoteNode(k))), t)))
    end
    return :(NamedTuple{N}(($(expr...),)))
end

#############
# update_wc #
#############

# This takes a draw for each field of the wc block.
Base.@inline @generated function draw_wc(wc_block::NamedTuple{N,T}, fc_in, t_km1, t_k) where {N,T}
    expr = []
    for k in N
        push!(expr, :(getproperty(wc_block, $(QuoteNode(k)))(t_km1, t_k, fc_in)))
    end
    return :(NamedTuple{N}(($(expr...),)))
end

# This produces a new model-form by taking new continuous draws and copying everything else.
function update_wc(f0_out::NamedTuple{N,T}, fc_in::B, t_km1, t_k) where {N, T, B}

    # Copy/update
    xc     = haskey(f0_out, :xc)     ? copy_from_prior_with_types(f0_out.xc, fc_in) : (;)
    xd     = haskey(f0_out, :xd)     ? copy_from_prior_with_types(f0_out.xd, fc_in) : (;)
    wc     = haskey(f0_out, :wc)     ? draw_wc(f0_out.wc, fc_in, t_km1, t_k) : (;)
    wd     = haskey(f0_out, :wd)     ? copy_from_prior(f0_out.wd, fc_in) : (;)
    models = haskey(f0_out, :models) ? update_wc_for_submodels(f0_out.models, fc_in, t_km1, t_k) : (;)

    # Collect all of those NamedTuples together into one big NamedTuple with unique names.
    nt = (;
        (:constants in N ? f0_out.constants : (;))...,
        xc...,
        wc...,
        xd...,
        wd...,
        models...,
    )
    
    # Return either the named tuple or the user's requested type.
    if :type in N
        return f0_out.type(; nt...)
    else
        return nt
    end
end

# Tuple/array expanders
Base.@inline function update_wc(f0_out::A, fc_in::B, t_km1, t_k) where {A <: Tuple, B <: Tuple}
    return ((update_wc(f0_out_i, fc_in_i, t_km1, t_k) for (f0_out_i, fc_in_i) in zip(f0_out, fc_in))...,)
end
Base.@inline function update_wc(f0_out::A, fc_in::B, t_km1, t_k) where {A <: Vector, B <: Vector}
    return [update_wc(f0_out_i, fc_in_i, t_km1, t_k) for (f0_out_i, fc_in_i) in zip(f0_out, fc_in)]
end

# This implements recursion for the update_wc process. It's a generated function because it
# was easy to write. It's not clear if this is useful.
Base.@inline @generated function update_wc_for_submodels(f0_out_models::NamedTuple{N,T}, prior, t_km1, t_k) where {N,T}
    expr = []
    for k in N
        push!(expr, :(update_wc(getproperty(f0_out_models, $(QuoteNode(k))), getproperty(prior, $(QuoteNode(k))), t_km1, t_k)))
    end
    return :(NamedTuple{N}(($(expr...),)))
end

# Notice how similar the structure is for update_xc, update_xd, update_wc, and update_wc? It
# would be nice to consolidate, but considering how many are generated functions, this got
# daunting.

###############
# Integrators #
###############

"""
The typical Runge-Kutta 4th order numerical integration method -- a great general choice for
fixed-step solvers.
"""
function rk4(
    fc::FCT,    # The continuous-time function
    f0_out,     # The initialization output (contains the structure)
    t_km1,      # Time at sample k-1
    t_k,        # Time at sample k
    x_km1,      # State at k-1
    fc_out_km1, # The outputs of fc at k-1 (already known)
) where {FCT}

    dt = t_k - t_km1
    k1 = fc_out_km1 # k1 is given.
    k2 = fc(t_km1 + dt/2, update_xc((xc_km1, (k1,)) -> xc_km1 + dt/2 * k1, f0_out, x_km1, (k1,)))
    k3 = fc(t_km1 + dt/2, update_xc((xc_km1, (k2,)) -> xc_km1 + dt/2 * k2, f0_out, x_km1, (k2,)))
    k4 = fc(t_km1 + dt,   update_xc((xc_km1, (k3,)) -> xc_km1 + dt   * k3, f0_out, x_km1, (k3,)))

    # Now assemble the model at t_k^-.
    x_km = update_xc((xc_km1, (k1, k2, k3, k4)) -> xc_km1 + dt/6 * (k1 + 2 * (k2 + k3) + k4), f0_out, x_km1, (k1, k2, k3, k4))

    return x_km

end

"""
This simple trapezoid-rule-like 2nd-order integration is useful when the time step is kept
small by discrete systems, s.t. the continuous-time dynamics are easily modeled by a 2nd-
order polynomial.
"""
function heun(fc::FCT, f0_out, t_km1, t_k, x_km1, fc_out_km1) where {FCT}

    dt = t_k - t_km1    
    k1 = fc_out_km1 # k1 is given.
    k2 = fc(t_km1 + dt, update_xc((xc_km1, (k1,)) -> xc_km1 + dt * k1, f0_out, x_km1, (k1,)))

    # Now assemble the model at t_k^-.
    x_km = update_xc((xc_km1, (k1, k2)) -> xc_km1 + dt/2 * (k1 + k2), f0_out, x_km1, (k1, k2))

    return x_km

end

##############
# Simulation #
##############

""" Steps from one time to the next. """
function step(integrator::IT, t_km1, t_k, f0_out, x_km1, fc::FCT, fd::FDT, fc_out_km1) where {IT, FCT, FDT}

    if fc !== nothing

        # Run the integrator to update from t_{k-1}^+ to t_k^-.
        x_km = integrator(fc, f0_out, t_km1, t_k, x_km1, fc_out_km1)
        # x_km still has wc in it, but that will be dropped by update_wd, next.

    else
        x_km = x_km1
    end

    if fd !== nothing

        # Take the draws for the discrete-time process.
        x_km = update_wd(f0_out, x_km, t_k)

        # Get the discrete outputs at t_k.
        fd_out = fd(t_k, x_km)

        # Update from t_k^- to t_k^+.
        x_kp = update_xd(f0_out, x_km, fd_out)

    else
        x_kp = x_km
        fd_out = nothing
    end

    # We return the fully-updated "input" version of the model.
    return (x_kp, fd_out)

end

# Recurisvely finds the minimum values stored in `field` among all models.
Base.@inline function find_min(out::NamedTuple{N,T}, x, field::Val{F}) where {N,T,F}
    if F in N
        x = min(x, maybe_rationalize(getfield(out, F)))
    end
    if :models in N
        for m in out.models
            x = find_min(m, x, field)
        end
    end
    return x
end
Base.@inline function find_min(out::Union{Tuple,AbstractArray}, x, field::Val{F}) where {F}
    for o in out
        x = find_min(o, x, field)
    end
    return x
end

""" Loops over all time steps in the simulation. """
function loop!(h, fc::FCT, fd::FDT, dt_max, t_end, x_k, f0_out, integrator) where {FCT, FDT}

    # Figure out the maximum step size we're allowed to take.
    dt_max = min(dt_max, t_end) # Never step past the end.
    dt_max = find_min(f0_out, dt_max, Val(:dt_max)) # Never step more than the smallest request.

    # The initial conditions define t_k^+ = 0^+, so we'll set up the kth and k+1th times.
    t_k   = 0//1
    t_kp1 = find_min(f0_out, dt_max, Val(:t_next))

    # Loop until we're starting a step at the end time.
    while t_k <= t_end

        # Move the last k to k-1.
        t_km1 = t_k
        x_km1 = x_k
        t_k   = t_kp1 # We'll check to see if we need a smaller step.
        
        if fc !== nothing
                
            # Take the random draws for the continuous-time process for the given step size.
            # On the final pass, t_km1 will be t_end, as will t_k, making for an
            # infinitessimal step. That will mess with continuous-time draws. So, here,
            # we'll pretend like we're taking a step at dt_max on the final sample.
            x_km1 = update_wc(f0_out, x_km1, t_km1, t_k > t_km1 ? t_k : t_km1 + dt_max)

            # Get the derivative at k-1 and log the continuous-time stuff.
            fc_out_km1 = fc(t_km1, x_km1)

            if h !== nothing
                recordc(t_km1, h, fc_out_km1, x_km1)
            end

        else
            fc_out_km1 = nothing
        end

        # See if we're done.
        if t_km1 == t_end
            return (t_km1, x_km1)
        end

        # Update.
        (x_k, fd_out) = step(integrator, t_km1, t_k, f0_out, x_km1, fc, fd, fc_out_km1)

        # Log the discrete-time stuff.
        if h !== nothing && fd !== nothing
            recordd(t_k, h, fd_out)
        end
        
        # Figure out the next time step.
        t_kp1 = min(t_k + dt_max, t_end) # Assume we step as far as possible.
        if fd !== nothing
            t_kp1 = find_min(fd_out, t_kp1, Val(:t_next))
        end

    end

    return (t_k, x_k) # Should be unreachable, but for type stability...

end

"""
This simulates a model described by the output of the f0 function. That model will be
updated over time by the continuous-time process described by the fc function and the
discrete-time process described by the fd function.
"""
function simulate(
    f0, fc, fd, t_end;
    dt_max = t_end,
    integrator::Function = rk4,
    record::Bool = true,
    seed = 1,
)
    # Let's take care of some type conversions. Internally, we use Rational{Int64} for time.
    dt_max = maybe_rationalize(dt_max)
    t_end  = maybe_rationalize(t_end)

    # Create the hierarchical structure of wrappers.
    rng = Xoshiro(seed)
    f0_out = f0 isa NamedTuple ? f0 : f0(rng)

    # Build the state at t_{k-1}^+ (ignoring random draws).
    x_0 = recursively(f0_out) do f0_out_i, downstream

        # Build the named tuple form.
        nt = (;
            # Pull out the constants and state.
            (haskey(f0_out_i, :constants) ? f0_out_i.constants : (;))...,
            (haskey(f0_out_i, :xc)        ? f0_out_i.xc        : (;))...,
            (haskey(f0_out_i, :xd)        ? f0_out_i.xd        : (;))...,
            # Add in the sub-models (already done).
            (haskey(f0_out_i, :models) ? NamedTuple{keys(f0_out_i.models)}(downstream) : (;))...,
        )

        # Return either the named tuple or the user's requested type.
        if haskey(f0_out_i, :type)
            return f0_out_i.type(; nt...)
        else
            return nt
        end

    end

    # Now add on the initial wd draws.
    x_0 = update_wd(f0_out, x_0, 0//1)

    # Create the time histories.
    if record
        h = recursively(f0_out) do f0_out_i, downstream
            ModelHistory(
                haskey(f0_out_i, :constant) ? f0_out_i.constants : (;),
                # We'll fill in the continuous-time stuff after the first fc evaluation, so
                # these all start empty.
                Float64[], # TODO: Only fill this in if there's continuous stuff. Or make all of these point to the same thing?
                haskey(f0_out_i, :xc) ? NamedTuple{keys(f0_out_i.xc)}(typeof(xci)[] for xci in f0_out_i.xc) : (;), # Make an array for each element of the state.
                haskey(f0_out_i, :xc) ? NamedTuple{keys(f0_out_i.xc)}(typeof(xci)[] for xci in f0_out_i.xc) : (;), # xc_dot uses the same everything as xc.
                haskey(f0_out_i, :yc) ? typeof(f0_out_i.yc)[] : nothing,
                # The first fd evaluation happens at the end of the first time step. We'll
                # start all of the discrete time histories at t=0.
                [0.,], # TODO: Only fill this in if there's discrete stuff.
                haskey(f0_out_i, :xd) ? NamedTuple{keys(f0_out_i.xd)}(typeof(xdi)[xdi,] for xdi in f0_out_i.xd) : (;),
                haskey(f0_out_i, :yd) ? typeof(f0_out_i.yd)[f0_out_i.yd,] : nothing,
                haskey(f0_out_i, :models) ? NamedTuple{keys(f0_out_i.models)}(downstream) : (;),
            )
        end
    else
        h = nothing
    end

    # Run the loop.
    (t_k, x_k) = loop!(h, fc, fd, dt_max, t_end, x_0, f0_out, integrator)

    # Return lots of things and let the user choose which they want.
    return (;
        history = h,
        t       = t_k,
        x       = x_k,
    )

end

end # module Overdot
