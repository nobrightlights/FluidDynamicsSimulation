import numpy as np
from math import ceil
from .constants import *
from numba import jit, prange
from abc import ABC, abstractmethod
from tqdm import tqdm


def main_runge_kutta(simulation_time, animation_step, density, concentration, source, emission_rate, diffusion_const, velocity_x, velocity_y, velocity_z, Antenna, dx, dt):
    """ Simulates fluid dynamics by calculating the temporal evolution of the density adn velocity field of the fluid using the 4th order Runge-Kutta method to integrate the governing equations with respect to time.
    The equations are the conservation of mass (to calculate density) and the Navier-Stokes equation (to calculate velocity). 
    With these solutions the temporal evolution of a carried quantity is calculated using the advection-diffusion equation, also through the 4th order Runge-Kutta method.
    The temporal rates of change of the density, velocity and concentration are calculated using finite difference methods on a staggered grid.
    Before the simulation starts, the initial conditions for density, velocity, concentration and the antenna must be set.
    Parameters:
        simulation_time (float):    Total time for the simulation.
        animation_step (float):     Time step between animation frames. 
                                    This is the time interval at which the simulation data is saved for visualization and analysis.
        density (np.ndarray):       Initial density distribution.
        concentration (np.ndarray): Initial concentration distribution.
        source (np.ndarray):        Source term of the concentrated quantity.
        diffusion_const (float):    Diffusion coefficient of the concentrated quantity.
        velocity_x (np.ndarray):    Initial x-compenent of the velocity field.
        velocity_y (np.ndarray):    Initial y-component of the velocity field.
        velocity_z (np.ndarray):    Initial z-component of the velocity field.
        Antenna (tuple):            Tuple containing a boolean and a list of antenna objects. 
                                    The boolean indicates whether the antenna is active, and the list contains the antenna objects that will be updated during the simulation.
        dx (float):                 Spatial discretization.
        dt (float):                 Temporal discretization or time step size.
    Returns:
        tuple: Contains the following arrays:
            - density_of_time (np.ndarray):         Density distribution over time.
            - concentration_of_time (np.ndarray):   Concentration distribution over time.
            - vx_of_time (np.ndarray):              x-component of the velocity-field over time. 
            - vy_of_time (np.ndarray):              y-component of the velocity-field over time.
            - vz_of_time (np.ndarray):              z-component of the velocity-field over time.
    """ 
    simulation_points = ceil(simulation_time/animation_step) + 1
    
    density = np.copy(density)
    concentration = np.copy(concentration)
    velocity_x = np.copy(velocity_x)
    velocity_y = np.copy(velocity_y)
    velocity_z = np.copy(velocity_z)

    # returned arrays
    density_of_time = np.zeros((simulation_points, density.shape[0], density.shape[1], density.shape[2]))
    density_of_time[0] = density

    concentration_of_time = np.zeros((simulation_points, density.shape[0], density.shape[1], density.shape[2]))
    concentration_of_time[0] = np.copy(concentration)

    vx_of_time = np.zeros((simulation_points, velocity_x.shape[0], velocity_x.shape[1], velocity_x.shape[2]))
    vx_of_time[0] = velocity_x

    vy_of_time = np.zeros((simulation_points, velocity_y.shape[0], velocity_y.shape[1], velocity_y.shape[2])) 
    vy_of_time[0] = velocity_y

    vz_of_time =  np.zeros((simulation_points, velocity_z.shape[0], velocity_z.shape[1], velocity_z.shape[2]))
    vz_of_time[0] = velocity_z

    k = 1

    for n in tqdm(range(1, int(simulation_time/dt + 1)), dynamic_ncols=True):
        time = n * dt

        # calculate updates for each property 
        # k1 update: slope at the beginning of the timestep
        density_diff_k1 = update_density(density, velocity_x, velocity_y, velocity_z, source, dx)
        vx_diff_k1, vy_diff_k1, vz_diff_k1 = update_velocity(density, velocity_x, velocity_y, velocity_z, dx)
        concentration_diff_k1 = update_concentration(concentration, source, emission_rate, diffusion_const, velocity_x, velocity_y, velocity_z, dx)
        
        # k2 input: estimate of the midpoint of the timestep using k1
        density_temp = density + 0.5*dt * density_diff_k1
        vx_temp, vy_temp, vz_temp = (velocity_x + 0.5*dt * vx_diff_k1, 
                                     velocity_y + 0.5*dt * vy_diff_k1, 
                                     velocity_z + 0.5*dt * vz_diff_k1)
        concentration_temp = concentration + 0.5*dt * concentration_diff_k1
        vx_temp[:,:,:], vy_temp[:,:,:], vz_temp[:,:,:] = no_slip_condition(vx_temp, vy_temp, vz_temp)
        concentration_temp = np.maximum(concentration_temp, 0.0)

        # k2 update: slope at the midpoint of the timestep
        density_diff_k2 = update_density(density_temp, vx_temp, vy_temp, vz_temp, source, dx)
        vx_diff_k2, vy_diff_k2, vz_diff_k2 = update_velocity(density_temp, vx_temp, vy_temp, vz_temp, dx)
        concentration_diff_k2 = update_concentration(concentration_temp, source, emission_rate, diffusion_const, vx_temp, vy_temp, vz_temp, dx)
        
        # k3 input: estimate of the midpoint of the timestep using k2
        density_temp = density + 0.5*dt * density_diff_k2
        vx_temp, vy_temp, vz_temp = (velocity_x + 0.5*dt * vx_diff_k2,
                                     velocity_y + 0.5*dt * vy_diff_k2, 
                                     velocity_z + 0.5*dt * vz_diff_k2)
        concentration_temp = concentration + 0.5*dt * concentration_diff_k2
        vx_temp[:,:,:], vy_temp[:,:,:], vz_temp[:,:,:] = no_slip_condition(vx_temp, vy_temp, vz_temp)
        concentration_temp = np.maximum(concentration_temp, 0.0)

        # k3 update: slope at the midpoint of the timestep
        density_diff_k3 = update_density(density_temp, vx_temp, vy_temp, vz_temp, source, dx)
        vx_diff_k3, vy_diff_k3, vz_diff_k3 = update_velocity(density_temp, vx_temp, vy_temp, vz_temp, dx)
        concentration_diff_k3 = update_concentration(concentration_temp, source, emission_rate, diffusion_const, vx_temp, vy_temp, vz_temp, dx)
       
        # k4 input: estimate of the end of the timestep using k3
        density_temp = density + dt * density_diff_k3
        vx_temp, vy_temp, vz_temp = (velocity_x + dt * vx_diff_k3,
                                     velocity_y + dt * vy_diff_k3,
                                     velocity_z + dt * vz_diff_k3)
        concentration_temp = concentration + dt * concentration_diff_k3
        vx_temp[:,:,:], vy_temp[:,:,:], vz_temp[:,:,:] = no_slip_condition(vx_temp, vy_temp, vz_temp)
        concentration_temp = np.maximum(concentration_temp, 0.0)

        # k4 update: slope at the end of the timestep
        density_diff_k4 = update_density(density_temp, vx_temp, vy_temp, vz_temp, source, dx)
        vx_diff_k4, vy_diff_k4, vz_diff_k4 = update_velocity(density_temp, vx_temp, vy_temp, vz_temp, dx)
        concentration_diff_k4 = update_concentration(concentration_temp, source, emission_rate, diffusion_const, vx_temp, vy_temp, vz_temp, dx)
        
        # update all properties
        density += (dt/6) * (density_diff_k1 + 2*density_diff_k2 + 2*density_diff_k3 + density_diff_k4)
        concentration += (dt/6) * (concentration_diff_k1 + 2*concentration_diff_k2 + 2*concentration_diff_k3 + concentration_diff_k4)
        velocity_x += (dt/6) * (vx_diff_k1 + 2*vx_diff_k2 + 2*vx_diff_k3 + vx_diff_k4)
        velocity_y += (dt/6) * (vy_diff_k1 + 2*vy_diff_k2 + 2*vy_diff_k3 + vy_diff_k4)
        velocity_z += (dt/6) * (vz_diff_k1 + 2*vz_diff_k2 + 2*vz_diff_k3 + vz_diff_k4)
        
         # enforce boundary conditions
        velocity_x[:,:,:], velocity_y[:,:,:], velocity_z[:,:,:] = no_slip_condition(velocity_x, velocity_y, velocity_z)
        concentration = np.maximum(concentration, 0.0)

        # update antenna and enforce oscillation
        if Antenna[0]:
            for a in Antenna[1]:
                update_antenna(a, velocity_x, velocity_y, velocity_z, dx, time)

        # save data at each animation step
        if n % (animation_step/dt) == 0:
            density_of_time[k] = density
            concentration_of_time[k] = concentration
            vx_of_time[k] = velocity_x
            vy_of_time[k] = velocity_y
            vz_of_time[k] = velocity_z
            k += 1
        
        # checks for simulation stability and validity
        if (density.max() > 2):
            print("abgebrochen")
            break
        
        if (cfl(velocity_x, velocity_y, velocity_z, dt, dx) > 1.0):
            print("CFL number too high")

        if (peclet(velocity_x, velocity_y, velocity_z, diffusion_const, dx) < 1.0):
            print("Peclet number too low")
            
    return density_of_time[:k], concentration_of_time[:k], vx_of_time[:k], vy_of_time[:k], vz_of_time[:k]

def cfl(velocity_x, velocity_y, velocity_z, dt, dx):
    """ Calculates the CFL number for the simulation.
    Returns:
        float: CFL number.
    """
    return (dt / dx) * np.sqrt(np.abs(velocity_x).max()**2 + np.abs(velocity_y).max()**2 + np.abs(velocity_z).max()**2)
 
def peclet(velocity_x, velocity_y, velocity_z, diffusion_const, L):
    """ Calculates the Peclet number for the simulation.
    Returns:
        float: Peclet number.
    """
    return (L * np.sqrt(np.abs(velocity_x).max()**2 + np.abs(velocity_y).max()**2 + np.abs(velocity_z).max()**2) ) / diffusion_const

def calc_max_speed(velocity_x, velocity_y, velocity_z):
    """ Calculates the maximum speed of the given velocity field stored in three separate arrays for the x, y, and z components on a staggered grid.
    Parameters:
        velocity_x (np.ndarray):    x-component of the velocity field.
        velocity_y (np.ndarray):    y-component of the velocity field.
        velocity_z (np.ndarray):    z-component of the velocity field.
    Returns:
        float: Maximum speed of the velocity field.
    """
    # calculate velocity at cell centers
    vx_centered = (velocity_x[1:,:,:] + velocity_x[:-1,:,:])/2
    vy_centered = (velocity_y[:,1:,:] + velocity_y[:,:-1,:])/2
    vz_centered = (velocity_z[:,:,1:] + velocity_z[:,:,:-1])/2
    # calculate speeds at cell centers
    speeds = np.sqrt(vx_centered**2 + vy_centered**2 + vz_centered**2)
    return np.max(speeds)


"""--------------------------------initialization functions----------------------------------"""
def init_density(density0, height, length, width, dx):
    """ Initializes a homogeneous density of dimension (height, length, width) with the value density0 and 
        corrects this density according to the barometric formula (for a constant temperature).  
    Parameters:
        density0 (float):   Value of homogeneous density.
        height (int):       Height of simulated space in number of grid cells.
        length (int):       Length      -------------"-----------
        width (int):        Width       -------------"-----------
        dx (float):         Spatial discretization of the simulated space.
    Returns:
        density (np.ndarray):   Density of simulated fluid with dimensions (height, length, width),
                                corrected according to barometric formula.
    """
    # initialize homogeneous density
    density = np.ones((height, length, width), dtype=np.float64) * density0

    # rescale according to barometric formula
    x_values = np.arange(0, height)
    barometric_scale = np.exp( - (np.tile(x_values[:, None, None], (1,length,width)) * dx) / BAROMETRIC_SCALE_HEIGHT)
    density[:,:,:] *= barometric_scale

    return density

def init_density_gaussian(height, length, width, point, amplitude, std_dev, dx):
    """ Initializes the density using init_density and adds a gaussian distribution around a specified point.
    Parameters:
        height (int):       Height of simulated space in number of grid cells. 
        length (int):       Length      -------------"-----------
        width (int):        Width       -------------"-----------
        point (list):       Position of the density pertubation (gaussian distribution).
        amplitude (float):  Amplitude of the density pertubation (gaussian distribution).
        std_dev (float):    Standard deviation of the gaussian distribution.
        dx (float):         Spatial discretization of the simulated space.
    Returns:
        density (np.ndarray):   Density of simulated fluid with dimensions (height, length, width),
                                corrected according to barometric formula and with a gaussian distribution around specified point.
    """
    density = init_density(DENSITY_AIR, height, length, width, dx)
    x, y, z = np.arange(0, height*dx, dx), np.arange(0, length*dx, dx), np.arange(0, width*dx, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    density += (amplitude * np.copy(density)[point[0], point[1], point[2]]) * np.exp(- ((X - point[0])**2 + (Y - point[1])**2 + (Z - point[2])**2) / (2 * std_dev**2))
     
    return density

def init_velocity(height, length, width):
    """ Initializes the velocity field components with zero velocity in all directions.
    Parameters:
        height (int):       Height of simulated space in number of grid cells.
        length (int):       Length      -------------"-----------
        width (int):        Width       -------------"-----------
    Returns:
        tuple: Contains the following arrays:
            - velocity_x (np.ndarray):    x-component of the velocity field with dimensions (height+1, length, width).
            - velocity_y (np.ndarray):    y-component of the velocity field with dimensions (height, length+1, width).
            - velocity_z (np.ndarray):    z-component of the velocity field with dimensions (height, length, width+1).
    """
    return np.zeros((height+1, length, width), dtype=np.float64), np.zeros((height, length+1, width), dtype=np.float64), np.zeros((height, length, width+1), dtype=np.float64)

def init_concentration(source, height, length, width):
    """ Initializes the concentration profile with zero concentration everywhere except at the position of the source.
    Parameters:
        source (list):      Position of the concentration source and amount of concentration at that position.
                            The source is a list with the following elements:
                            [x, y, z, amount], where x, y, z are the coordinates of the source and amount is the initial concentration at the source.
        height (int):       Height of simulated space in number of grid cells.
        length (int):       Length      -------------"-----------
        width (int):        Width       -------------"-----------
    Returns:
        concentration (np.ndarray):   Concentration of simulated fluid with dimensions (height, length, width),
                                      with a concentration at the position of the source.
    """
    # initialize concentration with zeros
    concentration = np.zeros((height, length, width), dtype=np.float64)
    # add concentration at the source position
    concentration[int(source[0]), int(source[1]), int(source[2])] = source[3]
    return concentration



"""--------------------------------update functions necessary for the runge-kutta method----------------------------------"""
@jit(nopython=True, nogil=True, cache=False, parallel=True)
def update_density(density, velocity_x, velocity_y, velocity_z, source, dx):
    """ Calculates the temporal rate of change of the density based on the current density distribution and velocity field according to the conservation of mass equation.
    The finite difference method in a staggered grid is used to calculate spatial derivatives.
    Parameters:
        density (np.ndarray):       Density of the fluid at the given times step.
        velocity_x (np.ndarray):    x-component of the velocity field at the given time step.
        velocity_y (np.ndarray):    y-component of the velocity field at the given time step.
        velocity_z (np.ndarray):    z-component of the velocity field at the given time step.
        dx (float):                 Spatial discretization. 
    Returns:
        density_diff (np.ndarray):  Change of density based on the current density distribution and velocity field.
    """
    # return array
    density_diff = np.zeros_like(density)
    
    # calculate density update at each cell
    for x in prange(0, density.shape[0]):
        for y in prange(0, density.shape[1]):
            for z in prange (0, density.shape[2]):

                # mean velocities at cell centers 
                mean_vx = (velocity_x[x+1,y,z] + velocity_x[x,y,z])/2
                mean_vy = (velocity_y[x,y+1,z] + velocity_y[x,y,z])/2
                mean_vz = (velocity_z[x,y,z+1] + velocity_z[x,y,z])/2

                # use upstream scheme for gradient of density:
                # use second-order backward difference for mean_v>0, second-order forward difference for mean_v<0
                # reduce order of accuracy at boundaries

                # boundaries
                if x==0:                    advection_x = (1/dx) * (density[x+1,y,z] - density[x,y,z])
                elif x==density.shape[0]-1: advection_x = (1/dx) * (density[x,y,z] - density[x-1,y,z])
                elif y==0:                  advection_y = (1/dx) * (density[x,y+1,z] - density[x,y,z])
                elif y==density.shape[1]-1: advection_y = (1/dx) * (density[x,y,z] - density[x,y-1,z])
                elif z==0:                  advection_z = (1/dx) * (density[x,y,z+1] - density[x,y,z])
                elif z==density.shape[2]-1: advection_z = (1/dx) * (density[x,y,z] - density[x,y,z-1])
                
                # interior cells
                else:
                    # x-direction
                    if mean_vx > 0: # backward difference
                        if x==1:    advection_x = (1/dx) * (density[x,y,z] - density[x-1,y,z])  # reduce order of accuracy to first-order at boundary
                        else:       advection_x = (1/(2*dx)) * (3*density[x,y,z] - 4*density[x-1,y,z] + density[x-2,y,z]) # use second-order in interior cells
                    else: # forward difference
                        if x==density.shape[0]-2: advection_x = (1/dx) * (density[x+1,y,z] - density[x,y,z]) # reduce order of accuracy to first-order at boundary
                        else:                     advection_x = (1/(2*dx)) * (-3*density[x,y,z] + 4*density[x+1,y,z] - density[x+2,y,z]) # use second-order in interior cells

                    # y-direction
                    if mean_vy > 0:
                        if y==1:    advection_y = (1/dx) * (density[x,y,z] - density[x,y-1,z])
                        else:       advection_y = (1/(2*dx)) * (3*density[x,y,z] - 4*density[x,y-1,z] + density[x,y-2,z])
                    else:
                        if y==density.shape[1]-2: advection_y = (1/dx) * (density[x,y+1,z] - density[x,y,z])
                        else:                     advection_y = (1/(2*dx)) * (-3*density[x,y,z] + 4*density[x,y+1,z] - density[x,y+2,z])

                    # z-direction
                    if mean_vz > 0:
                        if z==1:    advection_z = (1/dx) * (density[x,y,z] - density[x,y,z-1])
                        else:       advection_z = (1/(2*dx)) * (3*density[x,y,z] - 4*density[x,y,z-1] + density[x,y,z-2])
                    else:
                        if z==density.shape[2]-2: advection_z = (1/dx) * (density[x,y,z+1] - density[x,y,z])
                        else:                     advection_z = (1/(2*dx)) * (-3*density[x,y,z] + 4*density[x,y,z+1] - density[x,y,z+2])

                # combined advection term
                density_diff[x,y,z] -= (advection_x * mean_vx + advection_y * mean_vy + advection_z * mean_vz)

                # use centered difference scheme for the divergence of velocity  
                density_diff[x,y,z] -= (density[x,y,z]/dx) * (velocity_x[x+1,y,z] - velocity_x[x,y,z] +
                                                              velocity_y[x,y+1,z] - velocity_y[x,y,z] +
                                                              velocity_z[x,y,z+1] - velocity_z[x,y,z])                  
    return density_diff





@jit(nopython=True, nogil=True, cache=False, parallel=True)
def update_velocity(density, velocity_x, velocity_y, velocity_z, dx):
    """ Calculates the temporal rate of change of the velocity for each component based on the current density distribution and velocity field according to the Navier-Stokes equation for a compressible fluid.
    The finite difference method in a staggered grid is used to calculate spatial derivatives.  
    Parameters:
        density (np.ndarray):       Density of the fluid at the given time step.
        velocity_x (np.ndarray):    x-component of the velocity field at the given time step.
        velocity_y (np.ndarray):    y-component of the velocity field at the given time step.
        velocity_z (np.ndarray):    z-component of the velocity field at the given time step.
        dx (float):                 Spatial discretization.
    Returns:
        tuple: Contains the following arrays:
            - velocity_x_diff (np.ndarray):  Change of x-component of the velocity field based on the current density distribution and velocity field.
            - velocity_y_diff (np.ndarray):            y-component                            --------"-----------	 
            - velocity_z_diff (np.ndarray):            z-component                            --------"-----------
    """
    # return arrays
    velocity_x_diff, velocity_y_diff, velocity_z_diff = np.zeros_like(velocity_x), np.zeros_like(velocity_y), np.zeros_like(velocity_z)

    # factors for each term in the Navier-Stokes equations
    convective_factor = 1/(2*dx)
    pressure_factor = (SPECIFIC_GAS_CONST_AIR * TEMP / dx)
    kinvisc_factor = DYNAMIC_VISC_AIR/(dx**2)
    bulkvisc_factor = (4/3) * DYNAMIC_VISC_AIR/(dx**2)
    gravity_factor = GRAVITATIONAL_ACC

    # calculate velocity updates for each cell
    for x in prange(1, density.shape[0]):
        for y in prange(1, density.shape[1]):
            for z in prange(1, density.shape[2]):                
                
                # calculate velocity_x update
                if y<density.shape[1]-1 and z<density.shape[2]-1:
                    # mean properties at the position of vx
                    mean_density = (density[x,y,z] + density[x-1,y,z])/2
                    mean_vy_at_vx = (velocity_y[x-1,y,z] + velocity_y[x,y,z] + velocity_y[x-1,y+1,z] + velocity_y[x,y+1,z]) / 4
                    mean_vz_at_vx = (velocity_z[x-1,y,z] + velocity_z[x,y,z] + velocity_z[x-1,y,z+1] + velocity_z[x,y,z+1]) / 4
                    
                    # convection term
                    # use upstream scheme for gradient of velocity:
                    # gradient of vx in x-direction
                    if velocity_x[x,y,z] > 0: # backward difference
                        if x==1:    convection_x = 2 * (velocity_x[x,y,z] - velocity_x[x-1,y,z]) # reduce order of accuracy to first-order at boundary
                        else:       convection_x = 3*velocity_x[x,y,z] - 4*velocity_x[x-1,y,z] + velocity_x[x-2,y,z] # use second-order in interior cells
                    else: # forward difference
                        if x==density.shape[0]-1:   convection_x = 2 * (velocity_x[x+1,y,z] - velocity_x[x,y,z]) # reduce order of accuracy to first-order at boundary
                        else:                       convection_x = -3*velocity_x[x,y,z] + 4*velocity_x[x+1,y,z] - velocity_x[x+2,y,z] # use second-order in interior cells
                    # gradient of vx in y-direction
                    if mean_vy_at_vx > 0:
                        if y==1:    convection_y = 2 * (velocity_x[x,y,z] - velocity_x[x,y-1,z])
                        else:       convection_y = 3*velocity_x[x,y,z] - 4*velocity_x[x,y-1,z] + velocity_x[x,y-2,z]
                    else:
                        if y==density.shape[1]-2:   convection_y = 2 * (velocity_x[x,y+1,z] - velocity_x[x,y,z])
                        else:                       convection_y = -3*velocity_x[x,y,z] + 4*velocity_x[x,y+1,z] - velocity_x[x,y+2,z]
                    # gradient of vx in z-direction
                    if mean_vz_at_vx > 0:
                        if z==1:    convection_z = 2 * (velocity_x[x,y,z] - velocity_x[x,y,z-1])
                        else:       convection_z = 3*velocity_x[x,y,z] - 4*velocity_x[x,y,z-1] + velocity_x[x,y,z-2]
                    else:
                        if z==density.shape[2]-2:   convection_z = 2 * (velocity_x[x,y,z+1] - velocity_x[x,y,z])
                        else:                       convection_z = -3*velocity_x[x,y,z] + 4*velocity_x[x,y,z+1] - velocity_x[x,y,z+2]
                    
                    convective = - convective_factor * (convection_x * velocity_x[x,y,z] + convection_y * mean_vy_at_vx + convection_z * mean_vz_at_vx)

                    # pressure term  
                    pressure = - pressure_factor * (density[x,y,z] - density[x-1,y,z]) # centered difference

                    # viscosity terms
                    kin_visc = kinvisc_factor * ( velocity_x[x+1,y,z] + velocity_x[x-1,y,z] 
                                                + velocity_x[x,y+1,z] + velocity_x[x,y-1,z]
                                                + velocity_x[x,y,z+1] + velocity_x[x,y,z-1]
                                                - 6 * velocity_x[x,y,z] ) # centered difference
                    
                    bulk_visc = bulkvisc_factor * ( velocity_x[x+1,y,z] + velocity_x[x-1,y,z] - 2 * velocity_x[x,y,z] 
                                                + velocity_y[x,y+1,z] - velocity_y[x-1,y+1,z] - velocity_y[x,y,z] + velocity_y[x-1,y,z]
                                                + velocity_z[x,y,z+1] - velocity_z[x-1,y,z+1] - velocity_z[x,y,z] + velocity_z[x-1,y,z] ) # centered difference
                    # gravity term
                    gravity = - gravity_factor

                    # all terms combined
                    velocity_x_diff[x,y,z] = convective + gravity + (1/mean_density) * (pressure + kin_visc + bulk_visc)
                
                # calculate velocity_y updates
                if x<density.shape[0]-1 and z<density.shape[2]-1:
                    # mean properties at the position of vy
                    mean_density = (density[x,y,z] + density[x,y-1,z])/2
                    mean_vx_at_vy = (velocity_x[x,y-1,z] + velocity_x[x,y,z] + velocity_x[x+1,y-1,z] + velocity_x[x+1,y,z]) / 4
                    mean_vz_at_vy = (velocity_z[x,y-1,z] + velocity_z[x,y,z] + velocity_z[x,y-1,z+1] + velocity_z[x,y,z+1]) / 4

                    # convection term
                    # gradient of vy in x-direction
                    if mean_vx_at_vy > 0:
                        if x==1:    convection_x = 2 * (velocity_y[x,y,z] - velocity_y[x-1,y,z])
                        else:       convection_x = 3*velocity_y[x,y,z] - 4*velocity_y[x-1,y,z] + velocity_y[x-2,y,z]
                    else:   
                        if x==density.shape[0]-2:   convection_x = 2 * (velocity_y[x+1,y,z] - velocity_y[x,y,z])
                        else:                       convection_x = -3*velocity_y[x,y,z] + 4*velocity_y[x+1,y,z] - velocity_y[x+2,y,z]
                    # gradient of vy in y-direction
                    if velocity_y[x,y,z] > 0:
                        if y==1:    convection_y = 2 * (velocity_y[x,y,z] - velocity_y[x,y-1,z])
                        else:       convection_y = 3*velocity_y[x,y,z] - 4*velocity_y[x,y-1,z] + velocity_y[x,y-2,z]
                    else:
                        if y==density.shape[1]-1:   convection_y = 2 * (velocity_y[x,y+1,z] - velocity_y[x,y,z])
                        else:                       convection_y = -3*velocity_y[x,y,z] + 4*velocity_y[x,y+1,z] - velocity_y[x,y+2,z]
                    # gradient of vy in z-direction
                    if mean_vz_at_vy > 0:
                        if z==1:    convection_z = 2 * (velocity_y[x,y,z] - velocity_y[x,y,z-1])
                        else:       convection_z = 3*velocity_y[x,y,z] - 4*velocity_y[x,y,z-1] + velocity_y[x,y,z-2]
                    else:
                        if z==density.shape[2]-2:   convection_z = 2 * (velocity_y[x,y,z+1] - velocity_y[x,y,z])
                        else:                       convection_z = -3*velocity_y[x,y,z] + 4*velocity_y[x,y,z+1] - velocity_y[x,y,z+2]
                    
                    convective = - convective_factor * (convection_x * mean_vx_at_vy + convection_y * velocity_y[x,y,z] + convection_z * mean_vz_at_vy)
                    
                    # pressure term
                    pressure = - pressure_factor * (density[x,y,z] - density[x,y-1,z])

                    # viscosity terms
                    kin_visc = kinvisc_factor * ( velocity_y[x+1,y,z] + velocity_y[x-1,y,z]
                                                + velocity_y[x,y+1,z] + velocity_y[x,y-1,z] 
                                                + velocity_y[x,y,z+1] + velocity_y[x,y,z-1]
                                                - 6 * velocity_y[x,y,z] )
                    # kin_visc = 0
                    bulk_visc = bulkvisc_factor * ( velocity_y[x,y+1,z] + velocity_y[x,y-1,z] - 2 * velocity_y[x,y,z] 
                                                + velocity_x[x+1,y,z] - velocity_x[x+1,y-1,z] - velocity_x[x,y,z] + velocity_x[x,y-1,z]
                                                + velocity_z[x,y,z+1] - velocity_z[x,y-1,z+1] - velocity_z[x,y,z] + velocity_z[x,y-1,z] )
                    
                    # all terms combined
                    velocity_y_diff[x,y,z] = convective + (1/mean_density) * (pressure + kin_visc + bulk_visc)
                
                # calculate velocity_z update
                if x<density.shape[0]-1 and y<density.shape[1]-1:
                    # mean properties at the position of vz
                    mean_density = (density[x,y,z] + density[x,y,z-1])/2
                    mean_vx_at_vz = (velocity_x[x,y,z-1] + velocity_x[x,y,z] + velocity_x[x+1,y,z-1] + velocity_x[x+1,y,z]) / 4 
                    mean_vy_at_vz = (velocity_y[x,y,z-1] + velocity_y[x,y,z] + velocity_y[x,y+1,z-1] + velocity_y[x,y+1,z]) / 4
                    
                    # convection term
                    # gradient of vz in x-direction
                    if mean_vx_at_vz > 0:
                        if x==1:    convection_x = 2 * (velocity_z[x,y,z] - velocity_z[x-1,y,z])
                        else:       convection_x = 3*velocity_z[x,y,z] - 4*velocity_z[x-1,y,z] + velocity_z[x-2,y,z]
                    else:
                        if x==density.shape[0]-2:   convection_x = 2 * (velocity_z[x+1,y,z] - velocity_z[x,y,z])
                        else:                       convection_x = -3*velocity_z[x,y,z] + 4*velocity_z[x+1,y,z] - velocity_z[x+2,y,z]
                    # gradient of vz in y-direction
                    if mean_vy_at_vz > 0:
                        if y==1:    convection_y = 2 * (velocity_z[x,y,z] - velocity_z[x,y-1,z])
                        else:       convection_y = 3*velocity_z[x,y,z] - 4*velocity_z[x,y-1,z] + velocity_z[x,y-2,z]
                    else:
                        if y==density.shape[1]-2:   convection_y = 2 * (velocity_z[x,y+1,z] - velocity_z[x,y,z])
                        else:                       convection_y = -3*velocity_z[x,y,z] + 4*velocity_z[x,y+1,z] - velocity_z[x,y+2,z]
                    # gradient of vz in z-direction
                    if velocity_z[x,y,z] > 0:
                        if z==1:    convection_z = 2 * (velocity_z[x,y,z] - velocity_z[x,y,z-1])
                        else:       convection_z = 3*velocity_z[x,y,z] - 4*velocity_z[x,y,z-1] + velocity_z[x,y,z-2]
                    else:
                        if z==density.shape[2]-1:   convection_z = 2 * (velocity_z[x,y,z+1] - velocity_z[x,y,z])
                        else:                       convection_z = -3*velocity_z[x,y,z] + 4*velocity_z[x,y,z+1] - velocity_z[x,y,z+2]
                    
                    convective = - convective_factor * (convection_x * mean_vx_at_vz + convection_y * mean_vy_at_vz + convection_z * velocity_z[x,y,z])
                    
                    # pressure term
                    pressure = - pressure_factor * (density[x,y,z] - density[x,y,z-1]) 

                    # viscosity terms
                    kin_visc = kinvisc_factor * ( velocity_z[x+1,y,z] + velocity_z[x-1,y,z] 
                                                + velocity_z[x,y+1,z] + velocity_z[x,y-1,z]
                                                + velocity_z[x,y,z+1] + velocity_z[x,y,z-1]
                                                - 6 * velocity_z[x,y,z] )
                    
                    bulk_visc = bulkvisc_factor * ( velocity_z[x,y,z+1] + velocity_z[x,y,z-1] - 2 * velocity_z[x,y,z] 
                                                + velocity_x[x+1,y,z] - velocity_x[x+1,y,z-1] - velocity_x[x,y,z] + velocity_x[x,y,z-1]
                                                + velocity_y[x,y+1,z] - velocity_y[x,y+1,z-1] - velocity_y[x,y,z] + velocity_y[x,y,z-1] ) 
                    
                    # all terms combined
                    velocity_z_diff[x,y,z] = convective + (1/mean_density) * (pressure + kin_visc + bulk_visc)

    return velocity_x_diff, velocity_y_diff, velocity_z_diff


@jit(nopython=True, nogil=True, cache=False, parallel=True)
def update_concentration(concentration, source, emission_rate, diffusion_const, velocity_x, velocity_y, velocity_z, dx):
    """ Calculates the temporal rate of change of the concentration based on the current concentration and velocity field according to the advection-diffusion equation.
    The finite difference method in a staggered grid is used to calculate spatial derivatives.
    Parameters:
        concentration (np.ndarray):     Concentration profile of the diffusing and advected quantity at the given time step.
        source (np.ndarray):            Source of the diffusing and advected quantity.
                                        The source is a list with the following elements: 
                                        [x, y, z, amount], where x, y, z are the coordinates of the source and amount is the initial concentration at the source.
        emission_rate (float):          Emission rate of the diffusing and advected quantity at the source. 
        diffusion_const (float):        Diffusion coefficient of the diffusing and advected quantity.
        velocity_x (np.ndarray):        x-component of the velocity field at the given time step.
        velocity_y (np.ndarray):        y-component of the velocity field at the given time step.
        velocity_z (np.ndarray):        z-component of the velocity field at the given time step.
        dx (float):                     Spatial discretization.
    Returns:
        concentration_diff (np.ndarray): Change of concentration based on the current concentration and velocity field.
    """
    # return array
    concentration_diff = np.zeros_like(concentration)

    # calculate concentration update at each cell
    for x in prange(0, concentration.shape[0]):
        for y in prange(0, concentration.shape[1]):
            for z in prange(0, concentration.shape[2]):
                
                # divergence of velocity
                concentration_diff[x,y,z] -= (concentration[x,y,z]/dx) * (  velocity_x[x+1,y,z] - velocity_x[x,y,z]
                                                                          + velocity_y[x,y+1,z] - velocity_y[x,y,z]
                                                                          + velocity_z[x,y,z+1] - velocity_z[x,y,z] ) # centered difference
                
                # gradient of concentration and diffusion need special treatment at boudaries
                # mean velocities at the position of concentration
                mean_vx = (velocity_x[x+1,y,z] + velocity_x[x,y,z]) / 2
                mean_vy = (velocity_y[x,y+1,z] + velocity_y[x,y,z]) / 2
                mean_vz = (velocity_z[x,y,z+1] + velocity_z[x,y,z]) / 2

                # at boundaries
                # use backward/forward difference for gradient of concentration
                # consider no-flux condition for diffusion
                if x==0:
                    grad_x = (1/dx) * (concentration[x+1,y,z] - concentration[x,y,z])
                    diffussion_x = 2 * (diffusion_const/dx**2) * (concentration[x+1,y,z] - concentration[x,y,z])

                elif x==concentration.shape[0]-1:
                    grad_x = (1/dx) * (concentration[x,y,z] - concentration[x-1,y,z])
                    diffussion_x = 2 * (diffusion_const/dx**2) * (concentration[x-1,y,z] - concentration[x,y,z])

                # if not at boundaries
                # use second-order upstream scheme for gradient of concentration
                # reduce order of accuracy near boundaries
                else:
                    if mean_vx > 0: # backward difference
                        if x==1:    grad_x = (1/dx) * (concentration[x,y,z] - concentration[x-1,y,z])  
                        else:       grad_x = (1/(2*dx)) * (3*concentration[x,y,z] - 4*concentration[x-1,y,z] + concentration[x-2,y,z])
                    else: # forward difference
                        if x==concentration.shape[0]-2: grad_x = (1/dx) * (concentration[x+1,y,z] - concentration[x,y,z])
                        else:                           grad_x = (1/(2*dx)) * (-3*concentration[x,y,z] + 4*concentration[x+1,y,z] - concentration[x+2,y,z])
                    diffussion_x = (diffusion_const/dx**2) * (concentration[x+1,y,z] + concentration[x-1,y,z] - 2 * concentration[x,y,z])

                if y==0:
                    grad_y = (1/dx) * (concentration[x,y+1,z] - concentration[x,y,z])
                    diffussion_y = 2 * (diffusion_const/dx**2) * (concentration[x,y+1,z] - concentration[x,y,z])

                elif y==concentration.shape[1]-1:
                    grad_y = (1/dx) * (concentration[x,y,z] - concentration[x,y-1,z])
                    diffussion_y = 2 * (diffusion_const/dx**2) * (concentration[x,y-1,z] - concentration[x,y,z])
                else:
                    if mean_vy > 0:
                        if y==1:    grad_y = (1/dx) * (concentration[x,y,z] - concentration[x,y-1,z])
                        else:       grad_y = (1/(2*dx)) * (3*concentration[x,y,z] - 4*concentration[x,y-1,z] + concentration[x,y-2,z])
                    else:
                        if y==concentration.shape[1]-2: grad_y = (1/dx) * (concentration[x,y+1,z] - concentration[x,y,z])
                        else:                           grad_y = (1/(2*dx)) * (-3*concentration[x,y,z] + 4*concentration[x,y+1,z] - concentration[x,y+2,z])
                    diffussion_y = (diffusion_const/dx**2) * (concentration[x,y+1,z] + concentration[x,y-1,z] - 2 * concentration[x,y,z])
                
                if z==0:
                    grad_z = (1/dx) * (concentration[x,y,z+1] - concentration[x,y,z])
                    diffussion_z = 2 * (diffusion_const/dx**2) * (concentration[x,y,z+1] - concentration[x,y,z])

                elif z==concentration.shape[2]-1:  
                    grad_z = (1/dx) * (concentration[x,y,z] - concentration[x,y,z-1])
                    diffussion_z = 2 * (diffusion_const/dx**2) * (concentration[x,y,z-1] - concentration[x,y,z])
                else:
                    if mean_vz > 0:
                        if z==1:    grad_z = (1/dx) * (concentration[x,y,z] - concentration[x,y,z-1])
                        else:       grad_z = (1/(2*dx)) * (3*concentration[x,y,z] - 4*concentration[x,y,z-1] + concentration[x,y,z-2])
                    else:
                        if z==concentration.shape[2]-2: grad_z = (1/dx) * (concentration[x,y,z+1] - concentration[x,y,z])
                        else:                           grad_z = (1/(2*dx)) * (-3*concentration[x,y,z] + 4*concentration[x,y,z+1] - concentration[x,y,z+2])
                    diffussion_z = (diffusion_const/dx**2) * (concentration[x,y,z+1] + concentration[x,y,z-1] - 2 * concentration[x,y,z])

                concentration_diff[x,y,z] += (diffussion_x + diffussion_y + diffussion_z)
                concentration_diff[x,y,z] -= (grad_x * mean_vx + grad_y * mean_vy + grad_z * mean_vz)

    # add concentration from source
    concentration_diff[int(source[0]), int(source[1]), int(source[2])] += emission_rate

    return concentration_diff 

"""---------------------------------boundary conditions and oscillation functions----------------------------------"""
def no_slip_condition(velocity_x, velocity_y, velocity_z):
    """
    Ensure no-slip boundary conditions for the velocity field at a solid stationary wall bounding the simulated fluid.
    This function sets the velocity to zero at the walls and linearly interpolates the tangential velocities to ensure a no-slip condition.
    """
    # x-boundaries
    # normal velocity (directly at boundary): set to 0
    velocity_x[0,:,:] = 0
    velocity_x[-1,:,:] = 0
    # tangential velocities (half a grid cell away from the boundary): linearly interpolate from the adjacent cells to ensure 0 velocity at the walls
    velocity_y[0,:,:] = velocity_y[1,:,:] * (1/3)
    velocity_y[-1,:,:] = velocity_y[-2,:,:] * (1/3)
    velocity_z[0,:,:] = velocity_z[1,:,:] * (1/3)
    velocity_z[-1,:,:] = velocity_z[-2,:,:] * (1/3)

    # y-boundaries
    velocity_y[:,0,:] = 0
    velocity_y[:,-1,:] = 0
    velocity_x[:,0,:] = velocity_x[:,1,:] * (1/3)
    velocity_x[:,-1,:] = velocity_x[:,-2,:] * (1/3)
    velocity_z[:,0,:] = velocity_z[:,1,:] * (1/3)
    velocity_z[:,-1,:] = velocity_z[:,-2,:] * (1/3)

    # z-boundaries
    velocity_z[:,:,0] = 0
    velocity_z[:,:,-1] = 0
    velocity_x[:,:,0] = velocity_x[:,:,1] * (1/3)
    velocity_x[:,:,-1] = velocity_x[:,:,-2] * (1/3)
    velocity_y[:,:,0] = velocity_y[:,:,1] * (1/3)
    velocity_y[:,:,-1] = velocity_y[:,:,-2] * (1/3)

    return velocity_x, velocity_y, velocity_z


def update_antenna(Antenna, velocity_x, velocity_y, velocity_z, dx, time):
    """
    Update velocity fields based on antenna motion, accounting for the staggered grid structure.
    The function calculates the position and velocity of the antenna at a given time and updates the velocity fields accordingly.
    Parameters:
        Antenna (Antenna):           Antenna object containing position and velocity information.
        velocity_x (np.ndarray):     x-component of the velocity field to be updated.
        velocity_y (np.ndarray):     y-component of the velocity field to be updated.
        velocity_z (np.ndarray):     z-component of the velocity field to be updated.
        dx (float):                  Spatial discretization.
        time (float):                Current time in the simulation.
    """
    # obtain continuous (floating point) positions and velocities of the antenna
    positions = Antenna.get_positions(time)/dx  # converted to grid coordinates
    velocities = Antenna.get_velocity(time)
    

    for i in range(len(positions)):
        # get the position and velocity of the antenna
        pos = positions[i]
        vel = velocities[i]
        
        # get the cell indices where the antenna is located
        x_cell = int(np.floor(pos[0]))
        y_cell = int(np.floor(pos[1]))
        z_cell = int(np.floor(pos[2]))
        
        # calculate fractional position of the antenna within the cell
        x_frac = pos[0] - x_cell  # 0.0 to 1.0
        y_frac = pos[1] - y_cell  # 0.0 to 1.0
        z_frac = pos[2] - z_cell  # 0.0 to 1.0
        
        # update the velocity field at the neares cell faces
        velocity_x[x_cell, y_cell, z_cell] = vel[0] * (1.0 - x_frac)
        velocity_x[x_cell+1, y_cell, z_cell] = vel[0] * x_frac
        
        velocity_y[x_cell, y_cell, z_cell] = vel[1] * (1.0 - y_frac)
        velocity_y[x_cell, y_cell+1, z_cell] = vel[1] * y_frac
        
        velocity_z[x_cell, y_cell, z_cell] = vel[2] * (1.0 - z_frac)
        velocity_z[x_cell, y_cell, z_cell+1] = vel[2] * z_frac


class Antenna(ABC):
    """
    Abstract base class for antennas.
    This class defines the basic properties and methods for antennas, including their base position, frequency, and amplitude.
    Subclasses should implement the methods for calculating positions and velocities.
    """
    def __init__(self, base, frequency, amplitude):
        self.base = base
        self.frequency = frequency
        self.amplitude = amplitude

    @abstractmethod
    def get_positions(self, time):
        pass

    @abstractmethod
    def get_velocity(self, time):
        pass


class ImmobilePoint(Antenna):
    """
    Class representing an immobile point antenna.
    This antenna does not move and has a fixed position in space.
    The velocity however is a sinusoidal function of time, representing oscillation without movement.
    The velocity is calculated based on the oscillation parameters frequency and amplitude.
    The base position, frequency, amplitude and oscillation direction must be specified.
    The oscillation direction can be 'x', 'y' or 'z'.
    """
    def __init__(self, base, frequency, amplitude, oscillation_direction):
        super().__init__(base, frequency, amplitude)
        mapping = {'x':0, 'y':1, 'z':2}
        self.oscillation_direction = mapping.get(oscillation_direction)
    
    def get_positions(self, time):
        return self.base[None,:]
    
    def get_velocity(self, time):
        new = np.zeros(3)
        new[self.oscillation_direction] = self.amplitude * np.sin(2*np.pi * self.frequency * time)
        return new[None,:] 


class OscillatingPoint(Antenna):
    """
    Class representing an oscillating point antenna.
    This antenna moves in a sinusoidal pattern along a specified direction.
    The velocity is calculated based on the oscillation parameters frequency and amplitude.
    The base position, frequency, amplitude and oscillation direction must be specified.
    The oscillation direction can be 'x', 'y' or 'z'.
    """
    def __init__(self, base, frequency, amplitude, oscillation_direction):
        mapping = {'x':0, 'y':1, 'z':2}
        self.oscillation_direction = mapping.get(oscillation_direction)
        super().__init__(base, frequency, amplitude)
    
    def get_positions(self, time):
        new_pos = self.base.copy()
        new_pos[self.oscillation_direction] += self.amplitude * np.sin(2*np.pi * self.frequency * time)
        return new_pos[None,:]
    
    def get_velocity(self, time):
        new_vel = np.zeros(3)
        new_vel[self.oscillation_direction] = 2*np.pi * self.frequency * self.amplitude * np.cos(2*np.pi * self.frequency * time)
        return new_vel[None,:] 


class OscillatingPoint2d(Antenna):
    """
    Class representing a 2D oscillating point antenna, trying to represent bumblebee antennal motion in a simplified model.
    The antenna moves in a sinusoidal pattern along one direction, while moving in a cosine pattern along another direction.
    This results in a circulat motion along a circular path within the plane defined by the two oscillation directions.
    The velocity is calculated numerically.
    The base position, frequency, amplitude, the two oscillation directions and the initial direction of rotation (clockwise or anticlockwise) must be specified.
    The oscillation directions can be 'x', 'y' or 'z'.
    To model bumblebee antennal motion, two OscillatingPoint2d antennas are needed, one for each antenna, where one antenna initially rotates clockwise and the other anticlockwise.
    They should be initialized with the same oscillation directions.
    """
    def __init__(self, base, frequency, amplitude, oscillation_direction_inphase, oscillation_direction_outphase, clockwise):
        mapping = {'x':0, 'y':1, 'z':2}
        self.oscillation_direction_inphase = mapping.get(oscillation_direction_inphase)
        self.oscillation_direction_outphase = mapping.get(oscillation_direction_outphase)
        self.clockwise = clockwise
        super().__init__(base, frequency, amplitude)

    def phi_anticlockwise(self, time):
        return (np.pi/4) - (np.pi/4.5)*np.cos(2*np.pi * self.frequency * time)
    
    def phi_clockwise(self, time):
        return 2*np.pi - self.phi_anticlockwise(time)
    
    def get_positions(self, time):
        new_pos = self.base.copy()
        if self.clockwise:
            new_pos[self.oscillation_direction_outphase] += self.amplitude * np.sin(self.phi_clockwise(time))
            new_pos[self.oscillation_direction_inphase] += self.amplitude * np.cos(self.phi_clockwise(time))
        else:
            new_pos[self.oscillation_direction_outphase] += self.amplitude * np.sin(self.phi_anticlockwise(time))
            new_pos[self.oscillation_direction_inphase] += self.amplitude * np.cos(self.phi_anticlockwise(time))
        return new_pos[None,:]
    
    def get_velocity(self, time):
        # calculate numerical derivative for velocity
        delta_t = 0.001  # small time step for derivative
        pos_t = self.get_positions(time) # current position
        pos_t_plus_delta = self.get_positions(time + delta_t) # position at time + delta_t
        
        # velocity approximation using a finite difference
        velocity = (pos_t_plus_delta - pos_t) / delta_t
        return velocity

class OscillatingRod(Antenna):
    """
    Class representing an oscillating rod antenna.
    This antenna moves in a sinusoidal pattern along a specified direction.
    The rod is discretized into points, and the velocity is calculated based on the oscillation parameters.
    The base position, frequency, amplitude, length, discretization step (dx), orientation and oscillation direction must be specified.
    """
    def __init__(self, base, frequency, amplitude, length, dx, orientation, oscillation_direction):
        self.length = length
        self.discretization = dx
        mapping = {'x':0, 'y':1, 'z':2}
        self.orientation = mapping.get(orientation)
        self.oscillation_direction = mapping.get(oscillation_direction)
        super().__init__(base, frequency, amplitude)
        self.initial_pos = self.initial_positions()
        
    def initial_positions(self):
        line = self.base[self.orientation] + np.linspace(0, self.length, int(self.length/self.discretization + 1))
        
        indices = [0,1,2]
        indices.remove(self.orientation)
        axis2 = indices.pop(0)
        axis3 = indices.pop(0)

        initial_pos = np.empty((int(self.length/self.discretization + 1), 3))
        initial_pos[:, self.orientation] = line
        initial_pos[:, axis2] = self.base[axis2]
        initial_pos[:, axis3] = self.base[axis3]
        return initial_pos

    def get_positions(self, time):
        new_pos = self.initial_pos.copy()
        new_pos[:, self.oscillation_direction] += self.amplitude * np.sin(2*np.pi * self.frequency * time)
        return new_pos
        
    def get_velocity(self, time):
        new_vel = np.zeros((int(self.length/self.discretization + 1), 3))
        new_vel[:, self.oscillation_direction] = 2*np.pi * self.frequency * self.amplitude * np.cos(2*np.pi * self.frequency * time)
        return new_vel


