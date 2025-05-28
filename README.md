# FluidDynamicsSimulation

A 3D computational fluid dynamics (CFD) simulation framework for modeling fluid flow, and resulting concentration transport including diffusion with support for oscillating antennas (moving boundaries).

## Overview

Solves coupled equations using 4th-order Runge-Kutta time integration:
- **Conservation of Mass** - density evolution
- **Navier-Stokes equation** - velocity field computation  
- **Advection-Diffusion equation** - concentration transport

## Features

- 3D compressible fluid flow simulation
- Point source concentration tracking and diffusion
- Oscillating antenna modeling (point, rod, 2D circular motion)
- 2D velocity vector field and concentration animations
- Mass conservation analysis and stability monitoring
- Barometric density correction for atmospheric conditions

## Dependencies
- `numpy` - Numerical computations
- `matplotlib` - Plotting and animation
- `seaborn` - Enhanced visualizations
- `numba` - JIT compilation for performance
- `tqdm` - Progress bars

## Usage

### Basic Simulation Setup (Example)

```python
from cfd_simulation import *

# Initialize domain
height, length, width = 50, 50, 50  # grid cells
dx = 0.01  # spatial discretization [m]
dt = 0.00002  # time step [s]

# Initialize fields
density = init_density(DENSITY_AIR, height, length, width, dx)
velocity_x, velocity_y, velocity_z = init_velocity(height, length, width)
source = [25, 25, 25, 1.0]  # [x, y, z, initial_concentration]
concentration = init_concentration(source, height, length, width)

# Run simulation
simulation_time = 1.0  # [s]
animation_step = 0.01  # [s]
emission_rate = 0.1
diffusion_const = 1e-5

results = main_runge_kutta(
    simulation_time, animation_step, density, concentration, 
    source, emission_rate, diffusion_const,
    velocity_x, velocity_y, velocity_z, 
    (False, []), dx, dt
)

density_history, conc_history, vx_history, vy_history, vz_history = results
```
### Adding Oscillating Antennas

```python
# Create antenna objects
antenna1 = OscillatingPoint2d(
    base=np.array([25, 24, 25]), 
    frequency=2,  # Hz
    amplitude=0.01,  # m
    oscillation_direction_inphase='x',
    oscillation_direction_outphase='y',
    clockwise=True
)

antenna2 = OscillatingPoint2d(
    base=np.array([25, 26, 25]), 
    frequency=2, 
    amplitude=0.01,
    oscillation_direction_inphase='x',
    oscillation_direction_outphase='y', 
    clockwise=False
)

# Enable antennas in simulation
antenna_config = (True, [antenna1, antenna2])

# Run with antennas
results = main_runge_kutta(
    simulation_time, animation_step, density, concentration,
    source, emission_rate, diffusion_const,
    velocity_x, velocity_y, velocity_z,
    antenna_config, dx, dt
)
```

### Generating Visualizations

```python
# Create velocity vector field animation
vectorfield_2d(vx_history, vy_history, vz_history, 
               dt=animation_step, dx=dx, path="./output", 
               plane="xy", slice_pos=25)

# Create concentration animation
animate_concentration(conc_history, source, 'x', 'y', 
                     dt=animation_step, dx=dx, path="./output")

# Analyze mass conservation
mass_history = analyze_mass_conservation(density_history, dx, 
                                       animation_step, "./output")

# Calculate dispersion metrics
second_moments = second_moment(conc_history, source, dx)
center_of_mass_coords = center_of_mass(conc_history, dx)
```

## Physical Parameters

The simulation uses realistic atmospheric conditions defined in `constants.py`:

- **Temperature**: 298.15 K (25°C)
- **Air density**: 1.184 kg/m³  
- **Dynamic viscosity**: 18.37×10⁻⁶ Pa·s
- **Specific gas constant**: 287.05 J/(kg·K)
- **Gravitational acceleration**: 9.807 m/s²
- **Barometric scale height**: 8726.823 m

## Numerical Methods

### Time Integration
- 4th-order Runge-Kutta method for temporal discretization

### Spatial Discretization  
- Staggered grid arrangement for velocity components
- 2nd-order upwind scheme for convective terms
- Centered differences for diffusive terms
- No-slip boundary conditions at walls

### Stability Criteria
- **CFL condition**: Checks numerical stability for advection
- **Péclet number**: Validates diffusion-advection balance


## File Structure

```
├── __init__.py          # Package initialization
├── calculations.py      # Core simulation algorithms & antenna classes
├── analysis.py         # Visualization and analysis tools  
├── constants.py        # Physical constants (air properties, gravity, etc.)
└── README.md
```
