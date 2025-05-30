"""
Numerically solve and animate the trajectory of a point on a sphere under gravity.

Equations of motion are (30) and (31) [as of 24/05/2025] in the project pdf.
They are transformed into 4 first order differential equations so that
x' = z
z' = equation (30)
y' = w
w' = euqation (31)
( ' - denotes time derivative) 

To solve the system Runge - Kutta 4th order method is used.
See more: https://math.stackexchange.com/questions/721076/help-with-using-the-runge-kutta-4th-order-method-on-a-system-of-2-first-order-od 

The solution is plotted using Matplotlib.

24/05/2025
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def coord_xy_thetaphi(x, y):
    """ Coordinate transformation from the projection plane to spherical coordinates

    Args:
        x, y: coordinates in the xy projection plane

    Returns:
        theta, phi: spherical coordinates, where phi is the azimuthal angle
    
    """
    theta = [np.arctan(yv / xv) if xv > 0 else (np.pi + np.arctan(yv / xv)) for xv, yv in zip(x, y)]
    phi = [np.arctan(np.sqrt(xv**2 + yv**2) / R) for xv, yv in zip(x, y)]
    return (theta, phi)


def coord_thetaphi_3D(theta, phi):
    """ Coordinate transformation from spherical to Cartesian coordinates
    
    Args:
        theta, phi: spherical coordinates
    
    Returns:
        x, y, z: Cartesian coordinates
    """
    x = [R * np.cos(thetav) * np.sin(phiv) for thetav, phiv in zip(theta, phi)]
    y = [R * np.sin(thetav) * np.sin(phiv) for thetav, phiv in zip(theta, phi)]
    z = [R * (np.cos(phiv) + 1) for phiv in phi]
    return (x, y, z)


##### Equations for the respective time derivatives #####

def f_x(t, x, z):
    return z

def f_y(t, y, w):
    return w

def f_z(t, x, y):
    return (-1) * (G * m1 / R) * x * np.sqrt(R**2 + x**2 + y**2) / (x**2 + y**2)

def f_w(t, x, y):
    return (-1) * (G * m1 / R) * y * np.sqrt(R**2 + x**2 + y**2) / (x**2 + y**2)

#########################################################


def runge_kutta(f_x, f_z, f_y, f_w, t_0, x_0, z_0, y_0, w_0, h, num_steps):
    """
    Runge - Kutta 4th order method for solving ODEs
    
    Args:
        f_x: x'
        f_z: x''
        f_y: y'
        f_w: y''
        t_0, x_0, z_0. y_0, w_0: respective initial conditions
        h: step size
        num_steps: max amount of steps taken

    Returns:
        The solution array [t,x,y,z,w] to the ODE.

    """

    solution = [(t_0, x_0, y_0, z_0, w_0)] 

    t, x, y, z, w = t_0, x_0, y_0, z_0, w_0 # The initial position
   
    for _ in range(num_steps):
        k0 = h * f_x(t, x, z)
        l0 = h * f_z(t, x, y)
        m0 = h * f_y(t, y, w)
        n0 = h * f_w(t, x, y)
 
        k1 = h * f_x(t + 0.5 * h, x + 0.5 * k0, z + 0.5 * l0)
        l1 = h * f_z(t + 0.5 * h, x + 0.5 * k0, y + 0.5 * m0)
        m1 = h * f_y(t + 0.5 * h, y + 0.5 * m0, w + 0.5 * n0)
        n1 = h * f_w(t + 0.5 * h, x + 0.5 * k0, y + 0.5 * m0)

        k2 = h * f_x(t + 0.5 * h, x + 0.5 * k1, z + 0.5 * l1)
        l2 = h * f_z(t + 0.5 * h, x + 0.5 * k1, y + 0.5 * m1)
        m2 = h * f_y(t + 0.5 * h, y + 0.5 * m1, w + 0.5 * n1)
        n2 = h * f_w(t + 0.5 * h, x + 0.5 * k1, y + 0.5 * m1)

        k3 = h * f_x(t + h, x + k2, z + l2)
        l3 = h * f_z(t + h, x + k2, y + m2)
        m3 = h * f_y(t + h, y + m2, w + n2)
        n3 = h * f_w(t + h, x + k2, y + m2)

        x = x + (k0 + 2*k1 + 2*k2 + k3) / 6
        z = z + (l0 + 2*l1 + 2*l2 + l3) / 6
        y = y + (m0 + 2*m1 + 2*m2 + m3) / 6
        w = w + (n0 + 2*n1 + 2*n2 + n3) / 6

        t = t + h

        solution.append((t, x, y, z, w))

    return solution


############# Initial Conditions ################

R = 10 # Radius of the sphere
G = 1 # Gravitational constant
m1 = 1000 # Heavy mass
m2 = 1 # Light mass

t_0 = 0 
x_0 = 10 # Starting x position on the xy plane 
y_0 = -10 # Starting y position on the xy plane
z_0 = 10 # Starting x velocity on the xy plane
w_0 = 10 # Starting y velocity on the xy plane

h = 0.03 # R-K step size
num_steps = 500 # max step count

#################################################

solution = runge_kutta(f_x, f_z, f_y, f_w, t_0, x_0, z_0, y_0, w_0, h, num_steps)

t_values = [t for t, *_ in solution]
x_values = [x for _, x,  *_ in solution]
y_values = [y for _, _, y,  *_ in solution]


##### Plotting in 3D #####
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

## Sphere ##
theta_sphere = np.linspace(0, 2 * np.pi, 200)
phi_sphere = np.linspace(0, np.pi, 100)
theta_sphere, phi_sphere = np.meshgrid(theta_sphere, phi_sphere)

x_sphere = R * np.sin(phi_sphere) * np.cos(theta_sphere)
y_sphere = R * np.sin(phi_sphere) * np.sin(theta_sphere)
z_sphere = R * (np.cos(phi_sphere) + 1)

ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1)
############

theta, phi = coord_xy_thetaphi(x_values, y_values) # projection -> spherical coord 
x, y, z = coord_thetaphi_3D(theta, phi) # spherical coord -> cartesian 3D coord

## Plotting both masses and the trajectory ##
data = np.vstack((x, y, z))

mass1 = ax.scatter(0, 0, 2*R, color='k')
line, = ax.plot(data[0, 0], data[1, 0], data[2, 0], color='r')
mass2, = ax.plot(data[0,0], data[1,0], data[2,0], 'o', color='r')

def update(frame, data, line, mass2):
    line.set_data(data[:2, :frame])
    line.set_3d_properties(data[2, :frame])
    mass2.set_data(data[:2, (frame-1):frame])
    mass2.set_3d_properties(data[2, frame])

ax.set_aspect('equal')
plt.axis('off')
ax.grid(False)
plt.tight_layout()

ax.view_init(elev=60, azim=0) # Set camera rotation

anim = animation.FuncAnimation(fig=fig, func=update, frames=len(x), fargs=[data, line, mass2])

## If you want to save a video uncomment the following lines
# writervideo = animation.FFMpegWriter(fps=60)
# anim.save('animation.mp4', dpi=80, writer=writervideo)
# plt.close()

plt.show()
