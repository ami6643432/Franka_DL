import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

def plot_ee_poses(ee1_current, ee2_current, ee1_target, ee2_target, fig, ax):
    """
    Plots the current and target end-effector poses for both robots.
    
    Args:
        ee1_current (list): [x, y, z, qx, qy, qz, qw] for robot 1's current pose.
        ee2_current (list): [x, y, z, qx, qy, qz, qw] for robot 2's current pose.
        ee1_target (list): [x, y, z, qx, qy, qz, qw] for robot 1's target pose.
        ee2_target (list): [x, y, z, qx, qy, qz, qw] for robot 2's target pose.
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes object.
    """
    ax.clear()

    # Plotting the current end-effector positions as red dots
    ax.scatter(ee1_current[0], ee1_current[1], ee1_current[2], color='red', marker='o', label='Left EE Current', s=100)
    ax.scatter(ee2_current[0], ee2_current[1], ee2_current[2], color='red', marker='o', label='Right EE Current', s=100)

    # Plotting the target end-effector positions as green stars
    ax.scatter(ee1_target[0], ee1_target[1], ee1_target[2], color='green', marker='*', label='Left EE Target', s=150)
    ax.scatter(ee2_target[0], ee2_target[1], ee2_target[2], color='green', marker='*', label='Right EE Target', s=150)

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Robot End-Effector Poses')

    # Set axis limits for a better view
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([0.0, 3.0])

    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()

def main():
    """
    Main loop for the visualizer.
    This simulates receiving data from Isaac Lab and updating the plot.
    In a real-world scenario, the data would come from a network connection.
    """
    plt.ion()  # Turn on interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Example mock data (this would be replaced by data from your Isaac Lab script)
    ee1_current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ee2_current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    
    # Target poses from the FrankaDlEnv logs you provided
    ee1_target = [1.6000, -0.2500, 2.1480, 0.0, 1.0, 0.0, 0.0]
    ee2_target = [1.4000, -0.2500, 2.1480, 0.0, 1.0, 0.0, 0.0]

    # Main loop
    for i in range(100):
        # In a real-world application, this is where you'd receive data.
        # For this example, we will just move the current poses towards the target.
        ee1_current[0] += (ee1_target[0] - ee1_current[0]) * 0.05
        ee1_current[1] += (ee1_target[1] - ee1_current[1]) * 0.05
        ee1_current[2] += (ee1_target[2] - ee1_current[2]) * 0.05

        ee2_current[0] += (ee2_target[0] - ee2_current[0]) * 0.05
        ee2_current[1] += (ee2_target[1] - ee2_current[1]) * 0.05
        ee2_current[2] += (ee2_target[2] - ee2_current[2]) * 0.05
        
        plot_ee_poses(ee1_current, ee2_current, ee1_target, ee2_target, fig, ax)
        time.sleep(0.1)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
