import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter
import matplotlib.colors as mcolors
import seaborn as sn
from .constants import *


def vectorfield_2d(vx, vy, vz, dt, dx, path, plane="xy", slice_pos=None):
    """
    Creates a 2D animation of velocity vectors in a specified plane.
    
    Parameters:
        vx, vy, vz (np.ndarray):    Time series of the velocity components (time, x, y, z).
        path (str):                 Path to save the animation.
        plane (str):                Plane to visualize ('xy', 'xz', or 'yz').
        slice_pos (int, optional):  Position of the slice in the third dimension.
                                    If None, the middle of the domain is used.
        dt (float):                 Time step between frames.
    """
    # get shape information
    animation_steps = vx.shape[0]
    
    # set default slice position to middle of domain if not specified
    if slice_pos is None:
        if plane == "xy":
            slice_pos = vz.shape[3] // 2
        elif plane == "xz":
            slice_pos = vy.shape[2] // 2
        else:  # yz plane
            slice_pos = vx.shape[1] // 2
    
    # prepare arrays based on the chosen plane
    if plane == "xy":
        # for xy-plane: use vx and vy at the specified z position
        # interpolate vx to cell centers (average in x direction)
        vx_centered = (vx[:, :-1, :, slice_pos] + vx[:, 1:, :, slice_pos]) / 2
        # interpolate vy to cell centers (average in y direction)
        vy_centered = (vy[:, :, :-1, slice_pos] + vy[:, :, 1:, slice_pos]) / 2
        
        size_x = vx_centered.shape[1]
        size_y = vy_centered.shape[2]
        title = 'XY-Plane (z={})'.format(slice_pos)
        
    elif plane == "xz":
        # for xz-plane: use vx and vz at the specified y position
        vx_centered = (vx[:, :-1, slice_pos, :] + vx[:, 1:, slice_pos, :]) / 2
        vz_centered = (vz[:, :, slice_pos, :-1] + vz[:, :, slice_pos, 1:]) / 2
        
        size_x = vx_centered.shape[1]
        size_z = vz_centered.shape[2]
        title = 'XZ-Plane (y={})'.format(slice_pos)
        
    else:  # yz plane
        # for yz-plane: use vy and vz at the specified x position
        vy_centered = (vy[:, slice_pos, :-1, :] + vy[:, slice_pos, 1:, :]) / 2
        vz_centered = (vz[:, slice_pos, :, :-1] + vz[:, slice_pos, :, 1:]) / 2
        
        size_y = vy_centered.shape[1]
        size_z = vz_centered.shape[2]
        title = 'YZ-Plane (x={})'.format(slice_pos)
    
    # create grid for visualization
    if plane == "xy":
        # create a structured grid of points
        x = np.arange(0, size_x)*dx + dx/2
        y = np.arange(0, size_y)*dx + dx/2
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # prepare for quiver plot - swap X and Y to make x vertical
        positions = np.column_stack((Y.flatten(), X.flatten()))  # Y horizontal, X vertical
        u_data = vy_centered  # horizontal component (displayed on y-axis)
        v_data = vx_centered  # vertical component (displayed on x-axis)
        horizontal_label = 'y'
        vertical_label = 'x'
        
    elif plane == "xz":
        x = np.arange(0, size_x)*dx + dx/2
        z = np.arange(0, size_z)*dx + dx/2
        X, Z = np.meshgrid(x, z, indexing='ij')
        
        # swap X and Z to make x vertical
        positions = np.column_stack((Z.flatten(), X.flatten()))  # Z horizontal, X vertical
        u_data = vz_centered  # horizontal component (displayed on z-axis)
        v_data = vx_centered  # vertical component (displayed on x-axis)
        horizontal_label = 'z'
        vertical_label = 'x'
        
    else:  # yz plane
        y = np.arange(0, size_y)*dx + dx/2
        z = np.arange(0, size_z)*dx + dx/2
        Y, Z = np.meshgrid(y, z, indexing='ij')
        
        # keep as is since x is not in this plane
        positions = np.column_stack((Y.flatten(), Z.flatten()))
        u_data = vy_centered
        v_data = vz_centered
        horizontal_label = 'y'
        vertical_label = 'z'
    
    # calculate max speed for color normalization 
    max_speed = np.sqrt(np.max(u_data**2 + v_data**2))
    color_norm = mcolors.Normalize(vmin=0, vmax=max_speed)
    
    cmap = sn.cubehelix_palette(start=0.5, rot=-.75, as_cmap=True, dark=.25, light=.75)
    
    # setup plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    # set axis labels according to the plane
    ax.set_xlabel(f'{horizontal_label}-axis')
    ax.set_ylabel(f'{vertical_label}-axis')
    
    # initial quiver plot
    Q = ax.quiver(positions[:, 0], positions[:, 1], 
                  np.zeros(positions.shape[0]), np.zeros(positions.shape[0]),
                  scale=max_speed*3, scale_units='inches')
    
    # add title and time text
    # update title to reflect new orientation
    if plane == "xy":
        title = f'XY-Plane (z={slice_pos}, x vertical)'
    elif plane == "xz":
        title = f'XZ-Plane (y={slice_pos}, x vertical)'
    else:  # yz plane
        title = f'YZ-Plane (x={slice_pos})'
        
    plt.figtext(0.15, 0.92, title, fontsize=12, ha='center', va='center')
    time_text = plt.figtext(0.80, 0.92, '0s', fontsize=12, ha='center', va='center')
    
    # define update function for animation
    def update(frame):
        # get the vector components for the current frame
        if plane == "xy":
            u = u_data[frame]
            v = v_data[frame]
        elif plane == "xz":
            u = u_data[frame]
            v = v_data[frame]
        else:  # yz plane
            u = u_data[frame]
            v = v_data[frame]
            
        # calculate speeds for coloring
        speeds = np.sqrt(u**2 + v**2)
        
        # apply color normalization
        colors = cmap(color_norm(speeds.flatten()))
        
        # update quiver plot
        Q.set_UVC(u.flatten(), v.flatten())
        Q.set_color(colors)
        
        time_text.set_text(f"{frame * dt:.3f}s")
        return Q,
    
    # create animation
    anim = FuncAnimation(fig, update, frames=animation_steps, interval=50, blit=True)
    
    # save animation
    writer = FFMpegWriter(fps=20)
    anim.save(path + "/velocity_" + plane + ".mp4", writer=writer, dpi=250)
    
    plt.close()
    


def animate_concentration(input, source, i, j, dt, dx, path):
    """
    Creates a 2D animation of concentration profile with a point source over time in a specified plane.
    Parameters:
        input (np.ndarray):     4D array of concentration data (time, x, y, z).
        source (list):          Coordinates of the point source in the grid.
        i (str):                First axis to plot ('x', 'y', or 'z').
        j (str):                Second axis to plot ('x', 'y', or 'z').
        dt (float):             Time step between frames.
        dx (float):             Spatial discretization length.
        path (str):             Path to save the animation.
    """
    # get the axis mapping
    mapping = {'x':1, 'y':2, 'z':3}
    axis_i = mapping.pop(i)
    axis_j = mapping.pop(j)
    axis_3 = mapping.popitem()[1]

    # get the data for the specified plane
    data = input.take(int(source[int(axis_3-1)]), axis=axis_3)

    # figure setup
    fig, ax = plt.subplots()

    def animate_heatmap(frame, input, dt, dx, i, j):
        """
        Animate a heatmap of concentration in specified 2D plane ij.
        Parameters:
            frame (int):            Current frame number.
            input (np.ndarray):     3D array of concentration data (time, i, j).
            dt (float):             Time step between frames.
            dx (float):             Spatial discretization length.
            i (str):                First axis to plot ('x', 'y', or 'z').
            j (str):                Second axis to plot ('x', 'y', or 'z').
        """
        # clear the current figure
        plt.clf()

        # plot the heatmap
        data = input[frame,:,:,]
        sn.heatmap(data/data.max(), cmap=sn.color_palette("crest", as_cmap=True), linewidths=0)
        plt.gca().invert_yaxis()

        # configure the figure annotation
        plt.xlabel(f"{j} [m]")
        plt.ylabel(f"{i} [m]")
        tick_positions = np.linspace(0, data.shape[0], num=6, dtype=int)  
        tick_labels = [f"{pos * dx:.2f}" for pos in tick_positions]
        plt.xticks(tick_positions, tick_labels)
        plt.yticks(tick_positions, tick_labels)
        plt.figtext(0.7, 0.95, str(round(frame * dt, 3)) + "s", fontsize=12, ha='center', va='center')

    
    # create animation
    anim = FuncAnimation(fig, animate_heatmap, frames=input.shape[0], fargs=(data, dt, dx, i, j))

    # save animation
    writer = FFMpegWriter(fps=20)
    plane = f"{i}{j}"
    anim.save(path + "/concentration_" + plane + ".mp4", writer=writer, dpi=300)

    

def animate_density(input, i, j, k, dt, path, dx):
    """
    Creates a 2D animation of the density over time in a specified plane ij.
    Each frame shows a scatter plot where the color represents the density.
    Parameters:
        input (np.ndarray):     4D array of density data (time, x, y, z).
        source (list):          Coordinates of the point source in the grid
        i (str):                First axis to plot ('x', 'y', or 'z').
        j (str):                Second axis to plot ('x', 'y', or 'z').
        k (int):                Index along the third axis to take the slice from the input data.
        dt (float):             Time step size between data points.
        path (str):             Path to save the animation.
        dx (float):             Spatial discretization length.
    """
    # get the axis mapping
    mapping = {'x':1, 'y':2, 'z':3}
    axis_i = mapping.pop(i)
    axis_j = mapping.pop(j)
    axis_3 = mapping.popitem()[1]

    # get the data for the specified plane
    data = input.take(k, axis=axis_3)
    X, Y = np.meshgrid(np.arange(data.shape[axis_i]), np.arange(data.shape[axis_j]))

    # setup plot
    fig, ax = plt.subplots()
    background = 1
    fig.set_facecolor((background, background, background))
    ax.set_facecolor((background, background, background))

    def animate_scatter(frame, X, Y, data, dt, dx, i, j):
        """ Animate a scatter plot of density in 2D where the color represents the density value.
        Parameters:
            frame (int):        Current frame number
            X (np.ndarray):     X-coordinates for the scatter plot.
            Y (np.ndarray):     Y-coordinates for the scatter plot.
            data (np.ndarray):  3D array of density data (t, i, j).
            dt (float):         Time step between frames.
            dx (float):         Spatial discretization length.
            i (str):            First axis to plot ('x', 'y', or 'z').
            j (str):            Second axis to plot ('x', 'y', or 'z').
        """
        # clear current figure
        plt.clf()

        # color map and normalization
        cmap = sn.diverging_palette(220, 20, as_cmap=True)
        divnorm = mpl.colors.Normalize(vmin=data[:,:,:].min(), vmax=data[:, :, :].max())

        # scatter plot
        plt.scatter(X, Y, c=data[frame, :, :], cmap=cmap, norm=divnorm, s=2)

        # configure figure annotation
        tick_positions = np.arange(0, data.shape[1]+2, step=data.shape[1]/4)  
        tick_labels = [f"{pos * dx}" for pos in tick_positions]
        plt.xticks(tick_positions, tick_labels)
        plt.yticks(tick_positions, tick_labels)
        plt.xlabel(f"{j} [m]")
        plt.ylabel(f"{i} [m]")
        plt.figtext(0.7, 0.95, str(round(frame * dt, 3)) + "s", fontsize=12, ha='center', va='center')
        plt.gca().set_facecolor((1, 1, 1))

    # create animation
    anim = FuncAnimation(fig, animate_scatter, frames=input.shape[0], fargs=(X, Y, data[:,:,:], dt, dx, i, j))

    # save animation
    writer = FFMpegWriter(fps=20)
    plane = f"{i}{j}"
    anim.save(path + "/density_" + plane + ".mp4", writer=writer, dpi=300)


def plot_density_point(input, point, dt, path):
    """
    Plot the density at a specific point over time.
    Parameters:
        input (np.ndarray):     4D array of density data (time, x, y, z).
        point (tuple):          Coordinates of the point in the grid (x, y, z) where the density is plotted.
        dt (float):             Time step size between data points.
        path (str):             Path to save the plot.
    """
    # setup plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(which='major', linestyle='--', linewidth=0.8)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(which='minor', length=2)
    ax.set_xlim(0, input.shape[0])

    # plot the density at the specified point
    ax.plot(input[:, point[0], point[1], point[2]], color='teal') 

    # configure figure annotation
    tick_locations = np.arange(0, input.shape[0], 10)
    tick_labels = [str(x) for x in np.arange(0, input.shape[0]*dt*1000, 10*dt*1000)][:len(tick_locations)]
    ax.set_xticks(tick_locations, labels=tick_labels)
    ax.set_xlabel(r'time [$ms$]')
    ax.set_ylabel(r'density [$kg/m^3$]')
    plt.savefig(path, dpi=300)


def second_moment(concentration, source, discretization_length):
    """
    Calculate the total second moment and the directional second moments of a 3D concentration profile containing a point source.
    Parameters:   
        concentration (np.ndarray):     4D array of concentration data (time, x, y, z).
        source (list):                  Coordinates of the point source in the grid.
        discretization_length (float):  Spatial discretization length of the grid.
    Returns:
        np.ndarray:     Array of second moments (total, along x, along y, along z).
    """
    # initialize second moment arrays
    simulation_time = concentration.shape[0]
    second_moment_total = np.zeros(simulation_time)
    second_moment_x = np.zeros(simulation_time)
    second_moment_y = np.zeros(simulation_time)
    second_moment_z = np.zeros(simulation_time)

    # calculate grid and source positions
    grid_positions = (np.indices((concentration.shape[1], concentration.shape[2], concentration.shape[3])).transpose(1, 2, 3, 0) * discretization_length)
    source_position = np.array(source[:-1]) * discretization_length

    # calculate total squared difference
    squared_diff = np.sum((grid_positions - source_position)**2, axis=3)

    # calculate squared difference for each axis
    squared_diff_x = (grid_positions[:,:,:,0] - source_position[0])**2
    squared_diff_y = (grid_positions[:,:,:,1] - source_position[1])**2
    squared_diff_z = (grid_positions[:,:,:,2] - source_position[2])**2

    # calculate total second moment
    second_moment_total[:] = np.sum(np.multiply(concentration, squared_diff), axis=(1,2,3)) / np.sum(concentration, axis=(1,2,3))
    
    # calculate directional second moments
    second_moment_x[:] = np.sum(np.multiply(concentration, squared_diff_x), axis=(1,2,3)) / np.sum(concentration, axis=(1,2,3))
    second_moment_y[:] = np.sum(np.multiply(concentration, squared_diff_y), axis=(1,2,3)) / np.sum(concentration, axis=(1,2,3))
    second_moment_z[:] = np.sum(np.multiply(concentration, squared_diff_z), axis=(1,2,3)) / np.sum(concentration, axis=(1,2,3))

    return np.array([second_moment_total, second_moment_x, second_moment_y, second_moment_z])

def center_of_mass(concentration, discretization_length):
    """
     Calculate the center of mass of a 3D concentration profile.
     Parameters:
        concentration (np.ndarray):     4D array of concentration data (time, x, y, z).
        discretization_length (float):  Spatial discretization length of the grid.
    Returns:
        np.ndarray: Array of center of mass coordinates.
    """
    grid_positions = (np.indices((concentration.shape[1], concentration.shape[2], concentration.shape[3])).transpose(1, 2, 3, 0) * discretization_length)
    total_mass = np.sum(concentration, axis=(1,2,3))

    # calculate center of mass for each axis
    center_of_mass_x = np.sum(np.multiply(concentration, grid_positions[:,:,:,0]), axis=(1,2,3)) / total_mass
    center_of_mass_y = np.sum(np.multiply(concentration, grid_positions[:,:,:,1]), axis=(1,2,3)) / total_mass
    center_of_mass_z = np.sum(np.multiply(concentration, grid_positions[:,:,:,2]), axis=(1,2,3)) / total_mass
    
    return np.array([center_of_mass_x, center_of_mass_y, center_of_mass_z])


def analyze_mass_conservation(density_of_time, dx, animation_step, path):
    """
    Analyze mass conservation in simulation data density_of_time.
    Parameters:
        density_of_time (np.ndarray):   Time series of density field from simulation.
        dx (float):                     Spatial discretization.
        animation_step (float):         Time between data points in seconds.
        path (str):                     Path to save the analysis result plot.
    Returns:
        mass_history (np.ndarray): History of total mass at each timestep.
    """
    # calculate mass at each timestep
    time_points = density_of_time.shape[0]
    mass_history = np.zeros(time_points)
    time = np.arange(0, time_points) * animation_step
    
    print("Analyzing mass conservation in existing data...")
    for i in range(time_points):
        mass_history[i] = density_of_time[i].sum() * dx**3
    
    # calculate statistics
    initial_mass = mass_history[0]
    final_mass = mass_history[-1]
    abs_change = final_mass - initial_mass
    rel_change = (abs_change / initial_mass) * 100
    
    print(f"Initial mass: {initial_mass:.10f} kg")
    print(f"Final mass: {final_mass:.10f} kg")
    print(f"Absolute change: {abs_change:.10f} kg")
    print(f"Relative change: {rel_change:.10f}%")
    
    # plot results
    plt.figure(figsize=(12, 8))
    
    # plot total mass
    plt.subplot(2, 1, 1)
    plt.plot(time, mass_history, color='teal')
    plt.xlabel(r'time [$s$]')
    plt.ylabel(r'mass [$kg$]')
    plt.grid(True)
    
    # plot relative change
    plt.subplot(2, 1, 2)
    relative_change = [(m - initial_mass) / initial_mass * 100 for m in mass_history]
    plt.plot(time, relative_change, color='teal')
    plt.xlabel(r'time [$s$]')
    plt.ylabel(r'relative change [$\%$]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path + '/mass_conservation_analysis.png', dpi=300)
    plt.show()
    
    return mass_history
