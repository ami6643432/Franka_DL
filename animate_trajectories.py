#!/usr/bin/env python3
import argparse
from typing import Tuple, List

import numpy as np
from franka_traj import Trajectory_generator, create_quaternion_from_euler


def quat_to_rotmat(q: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to 3x3 rotation matrix."""
    x, y, z, w = q
    # Normalize to be safe
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    x /= np.sqrt(n)
    y /= np.sqrt(n)
    z /= np.sqrt(n)
    w /= np.sqrt(n)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])
    return R


def set_axes_equal(ax, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, margin: float = 0.05):
    # Make axes of 3D plot have equal scale
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    # Add margin
    pad = max_range * margin
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    cz = (z_min + z_max) / 2.0
    ax.set_xlim(cx - max_range / 2 - pad, cx + max_range / 2 + pad)
    ax.set_ylim(cy - max_range / 2 - pad, cy + max_range / 2 + pad)
    ax.set_zlim(cz - max_range / 2 - pad, cz + max_range / 2 + pad)


def animate_single_arm(total_time: float = 4.0, max_velocity: float = 0.15, num_waypoints: int = 15, fps: int = 12, axis_len: float = 0.07, triad_every: int = 1):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 registers '3d' projection

    # Define right arm poses (matching the franka_traj demo setup)
    start_pose_right = {
        'position': [0.5, -0.3, 0.8],
        'orientation': [1, 0, 0, 1]
    }
    end_pose_right = {
        'position': [0.3, -0.1, 0.6],
        'orientation': create_quaternion_from_euler(0, 45, 0)
    }

    traj_right = Trajectory_generator(
        start_pose_right, end_pose_right,
        total_time=total_time, max_velocity=max_velocity, num_waypoints=num_waypoints
    )

    wps = traj_right.compute_waypoints()
    X = np.asarray([p[0] for p in wps['positions']])
    Y = np.asarray([p[1] for p in wps['positions']])
    Z = np.asarray([p[2] for p in wps['positions']])
    Q = np.asarray(wps['orientations'])  # shape (N,4) as [x,y,z,w]

    N = len(X)
    idxs = np.arange(N)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Right Arm Trajectory with Orientation Frames')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    set_axes_equal(ax, X, Y, Z, margin=0.1)

    # Full path (faint)
    ax.plot(X, Y, Z, color="#4C78A8", lw=1.0, alpha=0.25)

    # Static triads at each waypoint (sparser if triad_every > 1)
    for i in range(0, N, max(1, int(triad_every))):
        Rm = quat_to_rotmat(tuple(Q[i]))
        p = np.array([X[i], Y[i], Z[i]])
        # faint triads
        ax.plot([p[0], p[0] + Rm[0, 0] * axis_len], [p[1], p[1] + Rm[1, 0] * axis_len], [p[2], p[2] + Rm[2, 0] * axis_len], color='red',   lw=0.8, alpha=0.25, zorder=2)
        ax.plot([p[0], p[0] + Rm[0, 1] * axis_len], [p[1], p[1] + Rm[1, 1] * axis_len], [p[2], p[2] + Rm[2, 1] * axis_len], color='green', lw=0.8, alpha=0.25, zorder=2)
        ax.plot([p[0], p[0] + Rm[0, 2] * axis_len], [p[1], p[1] + Rm[1, 2] * axis_len], [p[2], p[2] + Rm[2, 2] * axis_len], color='blue',  lw=0.8, alpha=0.25, zorder=2)

    # Highlight start and end orientation frames using 3D arrows (quiver)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    p0 = np.array([X[0], Y[0], Z[0]])
    pN = np.array([X[-1], Y[-1], Z[-1]])
    R0 = quat_to_rotmat(tuple(Q[0]))
    RN = quat_to_rotmat(tuple(Q[-1]))
    # Start frame (solid arrows)
    ax.quiver([p0[0]], [p0[1]], [p0[2]], [R0[0, 0]], [R0[1, 0]], [R0[2, 0]], length=axis_len, color='red',   lw=2.5, arrow_length_ratio=0.2, normalize=False, zorder=6)
    ax.quiver([p0[0]], [p0[1]], [p0[2]], [R0[0, 1]], [R0[1, 1]], [R0[2, 1]], length=axis_len, color='green', lw=2.5, arrow_length_ratio=0.2, normalize=False, zorder=6)
    ax.quiver([p0[0]], [p0[1]], [p0[2]], [R0[0, 2]], [R0[1, 2]], [R0[2, 2]], length=axis_len, color='blue',  lw=2.5, arrow_length_ratio=0.2, normalize=False, zorder=6)
    # End frame (thicker arrows)
    ax.quiver([pN[0]], [pN[1]], [pN[2]], [RN[0, 0]], [RN[1, 0]], [RN[2, 0]], length=axis_len, color='red',   lw=3.0, arrow_length_ratio=0.25, normalize=False, zorder=6)
    ax.quiver([pN[0]], [pN[1]], [pN[2]], [RN[0, 1]], [RN[1, 1]], [RN[2, 1]], length=axis_len, color='green', lw=3.0, arrow_length_ratio=0.25, normalize=False, zorder=6)
    ax.quiver([pN[0]], [pN[1]], [pN[2]], [RN[0, 2]], [RN[1, 2]], [RN[2, 2]], length=axis_len, color='blue',  lw=3.0, arrow_length_ratio=0.25, normalize=False, zorder=6)

    # Highlight start and end points
    start_scatter = ax.scatter([X[0]], [Y[0]], [Z[0]], s=70, c='green', marker='o', edgecolors='k', label='Start', zorder=5)
    end_scatter   = ax.scatter([X[-1]], [Y[-1]], [Z[-1]], s=90, c='red',   marker='^', edgecolors='k', label='End',   zorder=5)
    # Optional text labels
    ax.text(X[0],  Y[0],  Z[0],  'Start', color='green', fontsize=9, zorder=6)
    ax.text(X[-1], Y[-1], Z[-1], 'End',   color='red',   fontsize=9, zorder=6)

    # Progressive path and current marker
    path_line, = ax.plot([], [], [], color="#4C78A8", lw=2.0, label='Right path')
    marker, = ax.plot([], [], [], marker='o', color="#4C78A8", markersize=6, linestyle='None')

    # Current triad (highlighted)
    axis_x, = ax.plot([], [], [], color='red',   lw=3, zorder=7)
    axis_y, = ax.plot([], [], [], color='green', lw=3, zorder=7)
    axis_z, = ax.plot([], [], [], color='blue',  lw=3, zorder=7)

    ax.legend(loc='best')

    def set_point(line, x, y, z):
        line.set_data([x], [y])
        line.set_3d_properties([z])

    def set_line(line, xs, ys, zs):
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    def update(i: int):
        # Path so far
        set_line(path_line, X[: i + 1], Y[: i + 1], Z[: i + 1])
        # Marker
        set_point(marker, X[i], Y[i], Z[i])
        # Triad at current point
        Rm = quat_to_rotmat(tuple(Q[i]))
        p = np.array([X[i], Y[i], Z[i]])
        px = p + Rm[:, 0] * axis_len
        py = p + Rm[:, 1] * axis_len
        pz = p + Rm[:, 2] * axis_len
        set_line(axis_x, [p[0], px[0]], [p[1], px[1]], [p[2], px[2]])
        set_line(axis_y, [p[0], py[0]], [p[1], py[1]], [p[2], py[2]])
        set_line(axis_z, [p[0], pz[0]], [p[1], pz[1]], [p[2], pz[2]])
        return path_line, marker, axis_x, axis_y, axis_z

    animation.FuncAnimation(
        fig,
        update,
        frames=idxs,
        interval=1000.0 / float(fps),
        blit=False,
        repeat=True,
    )

    # Show interactive window only; no saving
    import matplotlib.pyplot as plt  # re-import for clarity
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Animate RIGHT arm trajectory (computed via franka_traj.Trajectory_generator) with orientation frames. Displays window only.")
    parser.add_argument("--total-time", type=float, default=4.0, help="Total motion time [s]")
    parser.add_argument("--max-vel", type=float, default=0.15, help="Max linear velocity [m/s]")
    parser.add_argument("--points", type=int, default=15, help="Number of waypoints")
    parser.add_argument("--fps", type=int, default=12, help="Animation FPS")
    parser.add_argument("--axis-len", type=float, default=0.07, help="Triad axis length [m]")
    parser.add_argument("--triad-every", type=int, default=1, help="Draw static triad at every Nth waypoint")
    args = parser.parse_args()

    animate_single_arm(
        total_time=args.total_time,
        max_velocity=args.max_vel,
        num_waypoints=args.points,
        fps=args.fps,
        axis_len=args.axis_len,
        triad_every=args.triad_every,
    )


if __name__ == "__main__":
    main()
