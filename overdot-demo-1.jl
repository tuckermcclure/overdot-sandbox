include("overdot.jl")
using .Overdot
using LinearAlgebra

############################
# Basic Controller Example #
############################

# This is called to set up everything in the sim. It is given a fresh random number
# generator; we can use that for whatever we like. The output describes to Overdot what's
# in our model. We'll call this "the model description".
function init(rng)
    return (;
        constants = (;      # "We'd like these constants to be part of our model."
            A = [-1.,  0.], # Plant dynamics
            K = [-2., -4.], # Controller gain matrix
        ),
        xc = (;             # "These are my continuous-time states."
            r = 1.,         # Position
            v = 0.,         # Velocity
        ),
        xd = (;             # "These are my discrete-time states."
            f = 0.,         # Controller-applied force
        ),
        dt_max = 1//10,     # Maximum step size
    )
end

# Overdot will use these outputs to build a NamedTuple with fields corresponding to what we
# called out above. E.g.:
#
# (;
#     A = [-1.,  0.],
#     K = [-2., -4.],
#     r = 1.,
#     v = 0.,
#     f = 0.,
# )
#
# Call this the "model" or "model form".

# This is called for the integration of the continuous-time states. Its inputs are time and
# the model. Its output describes how the continuous-time states are changing.
function ode(t, model)
    return (;
        overdot = (; # "The following are the derivatives of the continuous-time states."
            r = model.v,
            v = model.A ⋅ [model.r, model.v] + model.f, # Plant dynamics + controller force
        ),
    )
end

# Over time, Overdot will create updated models, and these will be the inputs to subsequent
# function calls.

# This is called for each discrete step -- at the end of integration. Its output describes
# how the discrete-time states should update.
function step(t, model)
    return (;
        update = (; # "The following are the new values for the discrete-time states."
            f = model.K ⋅ [model.r, model.v], # Feedback control
        ),
    )
end

# By looping over the ode and step functions, the model is updated.

# We can simulate this like so:
history, t_f, x_f = simulate(init, ode, step, 10//1)

# Here we can examine the updated model.
dump(x_f)

# Here, we see the time history of the continuous state `r`:
@show history.xc.r

# The corresponding time steps are history.tc.
@show history.tc

# Overdot takes the convention that the continuous-time process updates the model from
# t_{k-1}^+ to t_k^- -- from "just after" the last time step to "just before" the current
# time step. The discrete process updates from t_k^- to t_k^+. The initial conditions
# describe the model at t=0^+. The sim ends at t_n^+. Hence, in simulating from 0s to 10s
# with a step of 0.1s, the discrete function will run 100 times (not 101 because it doesn't
# run at 0s).**

####################
# Models of Models #
####################

# Let's redo the above, but let's make the plant and controller their own models. Our model
# will now look like this:
#
# (;
#     plant = (;
#         A = ...,
#         r = ...,
#         v = ...,
#     ),
#     controller = (;
#         K = ...,
#         f = ...,
#     ),
# )
#
# That means we want the output of the initialization function to look like this:
#
# (;
#     models = (;
#         plant = (;
#             constants = (;
#                 A = ...,
#             ),
#             xc = (;
#                 r = ...,
#                 v = ...,
#             ),
#         ),
#         controller = (;
#             constants = (;
#                 K = ...,
#             ),
#             xd = (;
#                 f = ...,
#             ),
#         ),
#     ),
# )
# 
# But to make it easier to construct models of models, we'll make some functions so that
# each sub-model can output just the part it's reponsible for.

# The plant has parts in initialization and the ODE.

plant_init(rng) = (;
    constants = (;
        A = [-1., 0.],
    ),
    xc = (;
        r = 1.,
        v = 0.,
    ),
)

plant_ode(t, plant, f) = (;
    overdot = (;
        r = plant.v,
        v = plant.A ⋅ [plant.r, plant.v] + f,
    ),
)

# The controller is present in initialization and the discrete update, so we'll need to
# write those parts.

controller_init(rng) = (;
    constants = (;
        K = [-2., -4.],
    ),
    xd = (;
        f = 0.,
    ),
    dt_max = 1//10,
)

controller_step(t, controller, plant) = (;
    update = (;
        f = controller.K ⋅ [plant.r, plant.v], # Update the state.
    ),
)

# The overall system simply combines all of these.

system_init(rng) = (;
    models = (;
        plant      = plant_init(rng),
        controller = controller_init(rng),
    ),
)

system_ode(t, system) = (;
    models = (;
        plant = plant_ode(t, system.plant, system.controller.f),
    ),
)

system_step(t, system) = (;
    models = (;
        controller = controller_step(t, system.controller, system.plant),
    ),
)

# We can now simulate this system of systems.
(history, t_f, x_f) = simulate(system_init, system_ode, system_step, 10//1)

# We can see that the resulting state is the same as before.
dump(x_f)

# Here's how we can access the time history of `r`:
@show history.models.plant.xc.r

# With these simple patterns, it becomes easy to build clearn, modular systems of any depth.
# Let's now look at some patterns we can build on top of this.

############################
# Regularly-Sampled Models #
############################

# The discrete function gets called on _all_ time steps. If we only want it to update at
# some regular period, then we need to check to see if it's "trigger time" and otherwise to
# output the last value. We also need to explicitly tell Overdot to take a step when we want
# one. Further, we don't want the intermediate (non-trigger-time) samples to be logged.
# Here's a manual way to do all of that:

controller_init(rng) = (;
    constants = (;
        K = [-2., -4.],
        sample_rate = 10//1, # We'll add a field for the sample rate (Hz).
    ),
    xd = (;
        f = 0.,
    ),
    t_next = 1//10, # Make sure Overdot stops at the next relevant time.
)

function controller_step(t, controller, plant)
    # If it's "trigger time", update stuff.
    if mod(t, inv(controller.sample_rate)) == 0
        return (;
            update = (;
                f = controller.K ⋅ [plant.r, plant.v], # Update the state.
            ),
            # We want to record this sample in the log:
            record = true,
            # Make sure Overdot stops at the next relevant time:
            t_next = t + inv(controller.sample_rate)
        )
    else # Otherwise, return the same old value for f.
        return (;
            update = (;
                f = controller.f, # No change
            ),
            # We don't want to record this sample.
            record = false,
            # Make sure Overdot stops at the next relevant time:
            t_next = ceil(t * controller.sample_rate) / controller.sample_rate,
        )
    end
end

# The above works perfectly well, and since we've overwritten our existing functions, we can
# run this directly:

history, t_f, x_f = simulate(system_init, system_ode, system_step, 10//1)

# Let's show the sample times for the discrete controller stuff:
@show history.models.controller.td

# This pattern is common, so Overdot provides a convenience function to construct this
# if-triggering... pattern for us: zohx for "zero-order hold on x". Let's give it a shot:

function controller_step(t, controller, plant)
    # This reads as "perform a zero-order hold at the current `t` with the given
    # `sample_rate`, where the field of `controller`` being updated is called `:f`."
    # We could add as many states in here as we like.
    return zohx(t, controller.sample_rate, controller, (:f,)) do
        (;
            f = controller.K ⋅ [plant.r, plant.v], # Update the state on trigger samples.
        )
    end
end

# This does _exactly_ what we typed above, but it's clearly much more convenient.
#
# (If you're not familiar with the foo(args..) do ... end pattern, it's a Julia feature. The
# part between do and end defines a function. That function is implicitly the first argument
# to foo, so in fact this is the same as: bar() = ..., foo(bar, args...).)

history, t_f, x_f = simulate(system_init, system_ode, system_step, 10//1)

@show history.models.controller.td

# To be clear: Overdot doesn't understand anything about regularly-sampled systems because
# it doesn't need to. Rather, we can construct a regularly-sampled system using more general
# mechanisms. Because this particular pattern is so common, however, Overdot does provide a
# convenience function to help us out.

#################################
# Constructors for Initializers #
#################################

# In order to parameterize the many things in an initialization function, it's nice to
# create a function that _returns_ an initialization function. Consider our controller,
# above, but assume we want to parameterize things with some helpful defaults.

function make_controller(;
    K           = [-2., -4.], # Gain matrix
    sample_rate = 10//1,      # Sample rate (Hz)
)
    # Return an initialization function with these parameters built in.
    return (rng, plant) -> (;
        constants = (;
            K           = K,
            sample_rate = sample_rate
        ),
        xd = (;
            f = K ⋅ [plant.r, plant.v],
        ),
        t_next = 1/sample_rate, # Make sure Overdot stops at the next relevant time.
    )
end

# Btw, (; x) means exactly the same thing as (; x = x), so this is a helpful pattern for
# this kind of thing. Let's re-write the above even more briefly.

function make_controller(;
    K           = [-2., -4.], # Gain matrix
    sample_rate = 10//1,      # Sample rate (Hz)
)
    # Return an initializer function with these parameters built in.
    return (rng, plant) -> (;
        constants = (; K, sample_rate),
        xd        = (; f = K ⋅ [plant.r, plant.v]),
        t_next    = 1/sample_rate, # Make sure Overdot stops at the next relevant time.
    )
end

# It's easy to see how we're "just describing" the model.

# Now we can create a controller where everything is at its default, but the sample rate is
# custom.
my_controller_init = make_controller(; sample_rate = 5//1)

# From here on, we'll use this pattern regularly instead of making a "hard-coded"
# initialization function.

####################
# Random Variables #
####################

# It's easy to add random variables to a model. Just like you specify which fields are
# continuous or discrete in the output of the initialization function, you also specify
# which fields are random inputs. These can be continuous or discrete. All that's required
# is adding 'wc' or 'wd' to the model definition (what's returned by the initialization
# function) -- named tuples mapping variable name to functions that take random draws. Then,
# the named variables will show up in the model, just like the state variables do.

σ_d   = 2.         # Standard deviation of each sample
t_end = 10//1      # How long to run the sim
Δt    = 1//10      # Step size
n     = t_end / Δt # Number of steps from the initial condition

create_discrete_random_walk(rng) = (;
    constants = (; σ = σ_d),
    xd = (; x = 0.),
    wd = (;
        # Let's create a random variable called ξ with the given standard deviation. Here,
        # we define a function that takes in the current time and the model and returns
        # the value for the random variable at that time. This creates a closure around rng.
        ξ = (t, model) -> randn(rng) * model.σ
    ),
    dt_max = Δt,
)

# For discrete steps, that function will be evaluated, and the result will be part of our
# model under the `ξ` field.
discrete_random_walk_step(t, model) = (;
    update = (;
        x = model.x + model.ξ, # Here, we just use our random variable.
    ),
)

# After n steps, we expect the standard deviation of the random walk to be sqrt(n)*σ.
σ_expected = sqrt(n) * σ_d

# Let's run a bunch of simulations and find the standard deviation of the final state.
# That should approach our expected value for a large number of runs. (Note the use of the
# `seed` input to `simulate` here.)
var_empirical = 0.
n_runs        = 1000
for k = 1:n_runs
    _, _, xf = simulate(
        create_discrete_random_walk, nothing, discrete_random_walk_step, t_end;
        seed   = k,
        record = false,
    )
    global var_empirical += xf.x^2 / n_runs
end
σ_empirical = sqrt(var_empirical)

@show σ_expected
@show σ_empirical

# Discrete random variables are drawn prior to running the discrete update function. They
# are stateful, and will be held constant through the subsequent continuous-time update.
# E.g., random variables drawn at t^- will be given to the discrete process that updates
# from t^- to t^+. The discrete random variables will be held constant through the
# continuous-time process from t^+ to (t+Δt)^-.

# Continuous random variables are very similar. The primary differences are that the
# function the draws continuous-time random variables is expected to take in t_{k-1} _and_
# t_{k}. This is because the correct numerical integration of continuous-time random
# variables in fact draws a single random value for the entire step. It is scaled by the
# step size so that the integral of the result acts like an integral over continuous-time
# noise. See "Wiener process" for more.
#
# For instance, here's a continuous-time version of the above -- an approximation of a
# Wiener process.

# Getting the equivalent continuous-time white noise standard deviation from the discrete-
# time value requires a conversion that goes with 1/sqrt(Δt).
σ_c = σ_d / sqrt(Δt)

create_continuous_random_walk(rng) = (;
    constants = (; σ = σ_c),
    xc = (; x = 0.),
    wc = (;
        # For continuous-time white noise, the draw is _smaller_ for larger steps.
        ξ = (t_km1, t_k, model_km1) -> randn(rng) * model_km1.σ / sqrt(t_k - t_km1)
    ),
    dt_max = 1//5, # It doesn't matter what step size we use now.
)

continuous_random_walk_ode(t, model) = (;
    overdot = (;
        x = model.ξ # We can use this to drive our derivatives.
    ),
)

# We now have a model of x where dx/dt is driven by continuous-time white noise.

# Let's run the same number of sims to see if this gives the same spread.
var_empirical = 0.
for k = 1:n_runs
    _, _, xf = simulate(
        create_continuous_random_walk, continuous_random_walk_ode, nothing, t_end;
        seed   = k,
        record = false,
    )
    global var_empirical += xf.x^2 / n_runs
end
σ_empirical = sqrt(var_empirical)

@show σ_empirical

# Note that the continuous-time random variables will not be present outside of the
# continuous-time process. They have no valid value for the discrete-time process and simply
# will not be present in the model.

# Random variables should _always_ use the provided random number generator (RNG). This is
# seeded appropriately and managed by Overdot. If you use the global random number
# generator, you may end up with statistical problems that show up at odd times.

# That's it for random variables.

###########
# Outputs #
###########

# It's common to have byproducts of the continuous- and discrete-time functions that are
# useful to other models or that are also desired in the time history. To specify these,
# include `yc` or `yd` in the model description and in the appropriate output from the
# continuous or discrete function. Here's an example.

# Let's revisit the controller-plant problem again. This time, let's break it into four
# systems: plant, sensor, controller, actuator. The sensor will produce a discrete, noisy
# measurement of the plant. The actuator will listen to the controller force (which we'll
# now treat as a command to the actuator) and which will quickly achieve the commanded
# force.

# The sensor's measurement function. We'll use this in sensor_step but also in the sensor's
# initialization function to get the initial measurement.
function measure(plant, noise)
    return (;
        r = plant.r + 0.1  * noise[1],
        v = plant.v + 0.05 * noise[2],
    )
end

# Returns a sensor initialization function.
function make_sensor(;
    sample_rate = 10//1, # Hz
)
    return function (rng, plant)
        noise = (t, sensor) -> randn(rng, 2) # A function to draw the random variables
        draws = noise(0., nothing) # The initial draws (neither input is actually used)
        meas  = measure(plant, draws) # Initial measurement
        return (; # Here's the model description
            constants = (; sample_rate),
            xd        = (; meas),  # The measurement is _state_ (because we need to sample-
            yd        = meas,      # and-hold it), but it's also a discrete output.
            wd        = (; noise),
            t_next    = 1/sample_rate,
        )
    end
end

# Updates the measurement. The new measurement will be both the state update and the
# discrete output.
function sensor_step(t, sensor, plant)

    # Here, we need to update both xd and yd. We'll use zohxy (note the y on the end).
    # This function lets us specify the xd fields and the yd field, and it assumes that
    # the yd field is part of the state (because it must be part of the state because
    # the sample-and-hold behavior means it's stateful, not an ephemeral output), so the
    # yd field name (here, `:meas`) doesn't have to be listed in both places.
    return zohxy(t, sensor.sample_rate, sensor, (), :meas) do
        (; meas = measure(plant, sensor.noise),)
    end

end

# We'll make an actuator model that responds smoothly to the input command. We'll also have
# a continuous output for the difference between the commanded force and achieved force (for
# the logs).
function make_actuator(;
    τ = 0.1, # Time constant of the first-order response (s)
    f = 0.,  # Initial actuator force
)
    return (rng, f_cmd) -> (;
        constants = (; τ),
        xc        = (; f),     # The actuator's applied foce
        yc        = 0.,        # The difference between commanded and achieved force
        xd        = (; f_cmd), # The actuator stores its command
        dt_max    = τ, # Things will get weird if we step too much past the time constant.
    )
end

# For its ODE, the actuator just rises to the command with the given time constant. We'll
# also make a continuous-time output for the difference between the target force and the
# actuator force.
function actuator_ode(t, actuator)
    Δf = actuator.f_cmd - actuator.f
    return (;
        overdot = (;
            f = Δf / actuator.τ,
        ),
        yc = Δf, # Continuous-time output, available at all continuous-time steps
    )
end

# This actuator's discrete process always runs; it's not regularly sampled (it's much faster
# than anything we care to model).
actuator_step(t, actuator, f_cmd) = (; update = (; f_cmd),)

# The overall system just contains all of the subsystems we've modeled. We'll make
# parameters for the initialization functions that by default just call the default
# constructor for each. This way, the user can pass in the results of their own call to
# make_controller(...) with custom parameters.
function make_system(;
    sensor_init     = make_sensor(),
    controller_init = make_controller(),
    actuator_init   = make_actuator(),
)
    # Return an initialization function that initializes the subsystems.
    return function (rng)
        plant_description      = plant_init(rng)
        sensor_description     = sensor_init(rng, plant_description.xc)
        controller_description = controller_init(rng, sensor_description.yd)
        actuator_description   = actuator_init(rng, controller_description.xd.f)
        return (;
            models = (;
                plant      = plant_description,
                sensor     = sensor_description,
                controller = controller_description,
                actuator   = actuator_description,
            ),
        )
    end
end

# Only the plant and actuator have continuous-time dynamics.
function system_ode(t, system)
    return (;
        models = (;
            plant    = plant_ode(t, system.plant, system.actuator.f),
            actuator = actuator_ode(t, system.actuator)
        ),
    )
end

# The discrete dynamics imply an order for what happens first or what happens in parallel.
# Here, we'll say the sensor is sampled and its measurement is immediately provided to the
# controller. The controller is sampled, and its command is immediately provided to the
# actuator. But any order could be modeled here; it's up to the user.
function system_step(t, system)

    # Get the updates for each system. Note how we can use system.sensor for the state at
    # the current time or sensor_updates to get the instantaneous updates that it's
    # requesting.
    sensor_updates     = sensor_step(t, system.sensor, system.plant)
    controller_updates = controller_step(t, system.controller, sensor_updates.yd)
    actuator_updates   = actuator_step(t, system.actuator, controller_updates.update.f)

    # Now we output all of those updates.
    return (;
        models = (;
            sensor     = sensor_updates,
            controller = controller_updates,
            actuator   = actuator_updates,
        ),
    )

end

# Let's run and see what we get.
history, t_f, x_f = simulate(make_system(), system_ode, system_step, 10//1)

# We can now see our "outputs" in history.models.actuator.yc and history.models.sensor.yd.

###################
# Swapping Models #
###################

# It's common to have multiple models of different fidelity for the same phenomenon. For
# instance, we could easily imagine different controllers. It would be nice to be able to
# swap a new controller in for the old without replacing any of the old code. To demonstrate
# this, we'll also show the last feature of Overdot: the "type" description.

# First, let's make a PIDController type. We'll use Base.@kwdef to give it a constructor
# with keyword arguments, like PIDController(; i = 1., f = 2.).
Base.@kwdef struct PIDController
    sample_rate::Rational{Int64}
    Kp::Float64
    Ki::Float64
    Kd::Float64
    i::Float64 # Integral value
    f::Float64 # Commanded force
end

# Let's make a constructor for a PID controller. As long as the resulting initialization
# function and controller_step function have the same interface, we ought to be able to
# swap this system in for the old controller. Note that in the model description, we're
# declaring the type of the thing we're describing to be a PIDController.
function make_pid_controller(;
    sample_rate = 10//1, # Hz
    Kp          = 2.,    # Proportional gain
    Ki          = 1.,    # Integral gain
    Kd          = 4.,    # Derivative gain
    i0          = 0.,    # Initial value for the integral
)
    return (rng, plant) -> (;
        type = PIDController,
        constants = (; sample_rate, Kp, Ki, Kd,),
        xd = (;
            i = i0,
            f = -Kp * plant.r - Ki * i0 - Kp * plant.v,
        ),
        t_next = 1/sample_rate,
    )
end

# This has the same signature as the last controller_step, but this one type specifically
# with PIDController, so when that's what the controller type is, this ought to just work.
function controller_step(t, pid::PIDController, plant)

    # We want a ZOH on both discrete states.
    return zohx(t, pid.sample_rate, pid, (:i,:f,)) do

        # Update the integral and then calculate the force.
        i = pid.i + plant.r / pid.sample_rate
        f = -pid.Kp * plant.r - pid.Ki * i - pid.Kp * plant.v

        # Return the updates.
        return (; i, f,)

    end

end

# Let's construct a system with this controller now.
system = make_system(; controller_init = make_pid_controller(; Ki = 0.5))
history, t_f, x_f = simulate(system, system_ode, system_step, 10//1)

@time history, t_f, x_f = simulate(system, system_ode, system_step, 10//1)

# We can see very clearly that this new controller was running in the sim.
dump(x_f)

# Using Julia types for modularity via multiple dispatch is a powerful pattern, but it's
# certainly not the only one. Anything that works in Julia in general will work here;
# Overdot does not know or care how the outputs from the three top-level functions are
# created, so it's up to users to figure out what works best for what they're modeling.
# As an example of an alternative for this particular case, we might have made a constant
# in the model called `step` that indicates which step function to use. A user could pass in
# `pid_step`, for instance.

# Aside from modularity, using `type` in a model description is useful for other reasons.
# It prints more easily at the console (rather than a large NamedTuple type). It also
# helps give hints to the editor. To construct the type, Overdot builds the "named tuple
# form" as usual and then it splats that into the type constructor as named arguments.
# Therefore, it requires a constructor that accepts named arguments. Also note that Overdot
# does not make all aspects of the model available at all times. The random draws may or may
# not be present, depending on context. Therefore, models with random variables will need
# constructors with default values for these fields (such as `nothing`).

# That's it for this demo of Overdot. Just with these basic patterns, I believe this would
# be sufficient for many modeling needs, including building large, modular
# systems-of-systems simulations. The architecture is also easily amenable to new features.
# It is not hard to imagine doing zero-crossing detection in this context, for instance:
# just let the continuos-time process output a block of zero-crossing variables (just like
# it has a block of overdot variables). When the updated state's zero-crossing variables are
# on the other side of 0 from the last state's zero-crossing variables, it's time to iterate
# towards the time at which the crossing occurred. I'm hoping this can continue to grow to
# add these kinds of new, general features without slowing down any applications that don't
# need them.



# Footnotes

# ** The reason we take the convention that the initialization corresponds to t=0^+ (and
# hence the discrete update doesn't run at that point) instead of t=0^- (where we'd need to
# run the discrete process to get from t=0^- to t=0^+) is that many discrete processes are
# supposed to represent discrete propagation from the last sample to the current sample. If
# they ran at t=0^-, then they'd be propagating from some _prior_ time, and that contradicts
# the idea that initialization is supposed to correspond to t=0. The discrete processes
# would have to have some kind of check, like "At t=0, don't propagate," and that's cheesy.
# Rather, just let the initialization function do whatever's necessary to set things up
# at t=0^+, including the discrete state at that point.
