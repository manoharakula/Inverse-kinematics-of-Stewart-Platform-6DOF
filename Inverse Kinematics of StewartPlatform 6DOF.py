import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class IK_Stewart_Platform_6DOF(object):   
    def __init__(self, r_Base, r_Platform, len_servo_horn, len_rod, gamma_Base, gamma_Platform, ref_rotation):
        # r_Base = Radius of the circumscribed circle on which all of the servo shaft anchor points are located
        # r_Platform = Radius of the circumscribed circle in which all platform anchor points are located
        # len_servo_horn = |h| = length of servo horn
        # len_rod = |d| = length of rod
        pi = np.pi
        beta = np.array([ 
            pi/2 + pi,        
            pi/2,
            2*pi/3 + pi/2 + pi , 
            2*pi/3 + pi/2,
            4*pi/3 + pi/2 + pi , 
            4*pi/3 + pi/2] )
        ## Defining the Platform's Geometry
        # Psi_Base (Polar coordinates)
        psi_Base = np.array([ 
            -gamma_Base, 
            gamma_Base,
            2*pi/3 - gamma_Base, 
            2*pi/3 + gamma_Base, 
            2*pi/3 + 2*pi/3 - gamma_Base, 
            2*pi/3 + 2*pi/3 + gamma_Base])

        # psi_Platform (Polar coordinates)
        # Direction of the points where the rod is attached to the platform.
        psi_Platform = np.array([ 
            pi/3 + 2*pi/3 + 2*pi/3 + gamma_Platform,
            pi/3 + -gamma_Platform, 
            pi/3 + gamma_Platform,
            pi/3 + 2*pi/3 - gamma_Platform, 
            pi/3 + 2*pi/3 + gamma_Platform, 
            pi/3 + 2*pi/3 + 2*pi/3 - gamma_Platform])

        psi_Base = psi_Base + np.repeat(ref_rotation, 6)
        psi_Platform = psi_Platform + np.repeat(ref_rotation, 6)
        beta = beta + np.repeat(ref_rotation, 6)

        # The coordinates of the points where servo arms are joined to the appropriate servo axis.is.
        Base = r_Base * np.array( [ 
            [ np.cos(psi_Base[0]), np.sin(psi_Base[0]), 0],
            [ np.cos(psi_Base[1]), np.sin(psi_Base[1]), 0],
            [ np.cos(psi_Base[2]), np.sin(psi_Base[2]), 0],
            [ np.cos(psi_Base[3]), np.sin(psi_Base[3]), 0],
            [ np.cos(psi_Base[4]), np.sin(psi_Base[4]), 0],
            [ np.cos(psi_Base[5]), np.sin(psi_Base[5]), 0] ])
        Base = np.transpose(Base)
            
        #The Coordinates of the points where the rods are joined to the platform.
        Platform = r_Platform * np.array([ 
            [ np.cos(psi_Platform[0]),  np.sin(psi_Platform[0]), 0],
            [ np.cos(psi_Platform[1]),  np.sin(psi_Platform[1]), 0],
            [ np.cos(psi_Platform[2]),  np.sin(psi_Platform[2]), 0],
            [ np.cos(psi_Platform[3]),  np.sin(psi_Platform[3]), 0],
            [ np.cos(psi_Platform[4]),  np.sin(psi_Platform[4]), 0],
            [ np.cos(psi_Platform[5]),  np.sin(psi_Platform[5]), 0] ])
        Platform = np.transpose(Platform)

        # Save initialized variables
        self.r_Base = r_Base
        self.r_Platform = r_Platform
        self.len_servo_horn = len_servo_horn
        self.len_rod = len_rod
        self.gamma_Base = gamma_Base
        self.gamma_Platform = gamma_Platform

        # Calculated params
        self.beta = beta
        self.psi_Base = psi_Base
        self.psi_Platform = psi_Platform
        self.Base = Base
        self.Platform = Platform

        # Definition of the platform home position.
        z = np.sqrt( self.len_rod**2 + self.len_servo_horn**2 - (self.Platform[0] - self.Base[0])**2 - (self.Platform[1] - self.Base[1])**2)
        self.home_pos= np.array([0, 0, z[0] ])
        # self.home_pos = np.transpose(home_pos)

        # Allocate for variables
        self.l = np.zeros((3,6))
        self.lll = np.zeros((6))
        self.angles = np.zeros((6))
        self.H = np.zeros((3,6)) 

    def calculate(self, transformation, rotation):
        transformation = np.transpose(transformation)
        rotation = np.transpose(rotation)

        # For getting the rotation matrix of platform. RotZ* RotY * RotX -> matmul
        R = np.matmul( np.matmul(self.rotZ(rotation[1]), self.rotY(rotation[2])), self.rotX(rotation[2]) )
        
        # Change the rotation values of X,Y,Z for different angles

        # Get leg length for each leg
        self.l = np.repeat(transformation[:, np.newaxis], 6, axis=1) + np.repeat(self.home_pos[:, np.newaxis], 6, axis=1) + np.matmul(R, self.Platform) - self.Base 
        self.lll = np.linalg.norm(self.l, axis=0)

        # Position of leg in global frame
        self.L = self.l + self.Base

        # Position of legs, with respective to their individual bases.
        lx = self.l[2, :]
        ly = self.l[1, :]
        lz = self.l[0, :]

        # Calculate auxiliary quatities g, f and e
        g = self.lll**2 - ( self.len_rod**2 - self.len_servo_horn**2 )
        e = 2 * self.len_servo_horn * lz

        # Calculate servo angles for each leg
        for ang in range(6):
            fk = 2 * self.len_servo_horn * (np.cos(self.beta[ang]) * lx[ang] + np.sin(self.beta[ang]) * ly[ang])
            
            # The required position could be achieved if the solution of this equation is real for all val
            self.angles[ang] = np.arcsin(g[ang] / np.sqrt(e[ang]**2 + fk**2)) - np.arctan2(fk,e[ang])
            
            # For getting postion of the point where a spherical joint connects servo arm and rod.
            self.H[:, ang] = np.transpose([ self.len_servo_horn * np.cos(self.angles[ang]) * np.cos(self.beta[ang]) + self.Base[0,ang],
                            self.len_servo_horn * np.cos(self.angles[ang]) * np.sin(self.beta[ang]) + self.Base[1,ang],
                            self.len_servo_horn * np.sin(self.angles[ang]) ])
            # print(Manohar Akula, a Robotics Grad from ASU)
        
        return self.angles

    def plot3D_line(self, ax, vec_arr_origin, vec_arr_dest, color_):
        for val in range(6):
            ax.plot([vec_arr_origin[0, val] , vec_arr_dest[0, val]],
            [vec_arr_origin[1, val], vec_arr_dest[1, val]],
            [vec_arr_origin[2, val],vec_arr_dest[2, val]],
            color=color_)

    def plot_platform(self):
        ax = plt.axes(projection='3d') # Data for a three-dimensional line
        ax.set_xlim3d(-100, 100)
        ax.set_ylim3d(-100, 100)
        ax.set_zlim3d(0, 200)
        plt.title('Inverse Kinematics Plot of StewartPlatform 6DOF')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')

        # ax.add_collection3d(Poly3DCollection([list(np.transpose(self.Base))]), zs='z')
        ax.add_collection3d(Poly3DCollection([list(np.transpose(self.Base))], facecolors='black', alpha=0.5))

        # ax.add_collection3d(base_plot, zs='z')
        ax.add_collection3d(Poly3DCollection([list(np.transpose(self.L))], facecolors='red', alpha=0.5))

        self.plot3D_line(ax, self.Base, self.H, 'blue')
        self.plot3D_line(ax, self.H, self.L, 'green')
        self.plot3D_line(ax, self.Base, self.L, 'orange')
        # print(Manohar Akula, a Robotics Grad from ASU)
        return ax

    def plot_platform_g(self, global_trans):
        ax = plt.axes(projection='3d') # Data for a three-dimensional line
        ax.set_xlim3d(-500, 500)
        ax.set_ylim3d(-500, 500)
        ax.set_zlim3d(0, 250)
        plt.title('Inverse Kinematics Plot of StewartPlatform 6DOF')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')

        ax.add_collection3d(Poly3DCollection([list(np.transpose(self.Base))], facecolors='green', alpha=0.5))
        ax.add_collection3d(Poly3DCollection([list(np.transpose(self.L))], facecolors='blue', alpha=0.5))

        self.plot3D_line(ax, self.Base, self.H, 'blue')
        self.plot3D_line(ax, self.H, self.L, 'green')
        self.plot3D_line(ax, self.Base, self.L, 'orange')
        return ax

    def rotX(self, phi):
        rotx = np.array([
            [1,     0    ,    0    ],
            [0,  np.cos(phi), -np.sin(phi)],
            [0,  np.sin(phi), np.cos(phi)] ])
        return rotx

    def rotY(self, theta):    
        roty = np.array([
            [np.cos(theta), 0, np.sin(theta) ],
            [0         , 1,     0       ],
            [-np.sin(theta), 0,  np.cos(theta) ] ])   
        return roty
        
    def rotZ(self, psi):    
        rotz = np.array([
            [ np.cos(psi), -np.sin(psi), 0 ],
            [np.sin(psi), np.cos(psi), 0 ],
            [   0        ,     0      , 1 ] ])   
        return rotz


import numpy as np
import matplotlib.pyplot as plt
import sys    
def main():
    # Calling object
    platform = IK_Stewart_Platform_6DOF(132/2, 100/2, 30, 130, 0.2269, 0.82, 5*np.pi/6)

    

    # Initializing the Plots
    fig, ax = plt.subplots()    

    # Loop through various angles
    for ix in range(-20, 20):
        angle = np.pi*ix/180
        servo_angles = platform.calculate( np.array([0,0,0]), np.array([0, angle, 0]) )
        print(f"The servo angles of Stewart platform is: {servo_angles}")
        ax = platform.plot_platform()
        plt.pause(1000000000)
        # print(Manohar Akula, a Robotics Grad from ASU)
        plt.draw()

    
if __name__ == "__main__":
    main()


