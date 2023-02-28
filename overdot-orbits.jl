# This implements the equations of motion for a bunch of little masses in circular orbits
# around the earth. It's useful for examining timing of large numbers of numbers of models.
# E.g., this is 4x slower with a tuple of orbiters than an array of orbiters.

include("overdot.jl")
using .Overdot
using LinearAlgebra
using StaticArrays

Re = 6378137.      # Earth radius (m)
h  = 550000.       # Orbital height (m)
μ  = 398600.4415e9 # Gravitational parameter (m^3/s^2)
i  = 50 * π/180    # Inclination (rad)
n  = 50            # Number of satellites to run
v  = √(μ/(Re+h))   # Orbital velocity (circular) (m/s)
T  = 2π*(Re+h) / v # Period (s)

# Returns a direction rotation matrix for a rotation of θ about z (frame rotation)
Rz(θ) = SA[cos(θ) sin(θ) 0.; -sin(θ) cos(θ) 0.; 0. 0. 1.]

# Simple point-mass gravity function
gravity(r) = -μ/norm(r)^3 * r

# Initializes a bunch of orbiters.
function f0(rng)
    return (;
        # Our top-level model only has sub-models.
        models = (;
            # We'll store everything in "orbits".
            orbiters = [ # We use an array; 50 is too big for tuples.
                (;
                    # Set up the continuous state spaced around the equator.
                    xc = (;
                        r = (Re+h) * SA[cos((k-1)/n * 2π), sin((k-1)/n * 2π), 0.],
                        v = Rz((k-1)/n * 2π)' * SA[0., v * cos(i), v * sin(i)],
                    ),
                ) for k in 1:n
            ]
        )
    )
end

# Returns the derivatives for all of the orbiters.
function fc(t, model)
    return (;
        models = (;
            orbiters = [
                (;
                    overdot = (;
                        r = o.v,
                        v = gravity(o.r)
                    ),
                ) for o in model.orbiters
            ]
        ),
    )
end

# Run it for a few seconds to get stuff started.
history, t_f, x_f = simulate(f0, fc, nothing, 3//1, dt_max = 1//1)

# Now time it.
@time history, t_f, x_f = simulate(f0, fc, nothing, 50*T, dt_max = 1//1)

# One run's timing was ~350,000x faster than real time on my 2016 MBP (2.3GHz i9).
