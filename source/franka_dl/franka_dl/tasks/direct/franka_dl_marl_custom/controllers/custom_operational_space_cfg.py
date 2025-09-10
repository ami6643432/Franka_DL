from collections.abc import Sequence

from isaaclab.utils import configclass

# Import the base OSC config from Isaac Lab package
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from .custom_operational_space import CustomOperationalSpaceController


@configclass
class CustomOperationalSpaceControllerCfg(OperationalSpaceControllerCfg):
    """Configuration for custom operational-space controller with joint-space control modes."""

    class_type: type = CustomOperationalSpaceController
    """The associated controller class."""

    # Custom control mode configurations
    control_mode: str = "operational_space"
    """Control mode: "operational_space", "joint_pd", "joint_impedance", "custom"."""

    # Joint-space PD controller parameters (controller gains)
    pd_kp: float | Sequence[float] | None = None
    """Joint-space proportional gain Kp for the controller (not physical stiffness). If None, falls back to
    :obj:`joint_stiffness` for backward compatibility."""

    pd_kd: float | Sequence[float] | None = None
    """Joint-space derivative gain Kd for the controller (not physical damping). If None, computed from
    :obj:`joint_damping_ratio` as 2*sqrt(Kp)*zeta."""

    # Backward-compatibility aliases (treated as controller gains when pd_kp/pd_kd are not provided)
    joint_stiffness: float | Sequence[float] = 100.0
    """Alias used previously for controller Kp; kept for backward compatibility."""

    joint_damping_ratio: float | Sequence[float] = 1.0
    """Damping ratio used to compute controller Kd if :obj:`pd_kd` is None."""

    joint_stiffness_limits: tuple[float, float] = (0, 1000)
    """Minimum and maximum values for joint stiffness gains."""

    joint_damping_ratio_limits: tuple[float, float] = (0, 100)
    """Minimum and maximum values for joint damping ratios."""

    # Joint-space impedance controller parameters
    joint_impedance_mode: str = "fixed"
    """Joint impedance mode: "fixed", "variable", "variable_kp"."""

    joint_inertial_compensation: bool = False
    """Whether to perform inertial compensation in joint space."""

    joint_gravity_compensation: bool = False
    """Whether to perform gravity compensation in joint space."""

    # Custom control equation parameters
    enable_custom_equations: bool = False
    """Whether to enable custom control equations."""

    custom_equation_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    """Weights for combining different control components in custom mode."""
