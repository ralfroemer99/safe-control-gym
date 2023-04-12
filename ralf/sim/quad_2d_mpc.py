"""2D quadrotor with MPC.

Example:

    $ python3 quad_2d_mpc.py --overrides ../examples/3d_quad.yaml

"""
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from ralf.plot.animate_2d_quad import plot_robot


T_s = 0.1
T_hor = 1


def main():
    """ The main function creating, running, and closing an environment.

    """
    # Start a timer.
    START = time.time()

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    env = partial(make, 'quadrotor', **config.task_config)

    # Controller
    ctrl = make(config.algo,
                env,
                T_s=0.05,
                horizon=int(T_hor / T_s),
                terminate_run_on_done=False,
                **config.algo_config)

    # Reset the environment, obtain and print the initial observations.
    ctrl.reset()

    # Run experiment
    results = ctrl.run()

    # Close the environment and print timing statistics.
    ctrl.close()

    # Plot
    x_all, u_all = results['obs'][:-1], results['action']
    ref_all = ctrl.env.X_GOAL
    t_all = np.arange(0, env.keywords['episode_len_sec'], 1 / env.keywords['ctrl_freq'])

    fig, ax = plt.subplots(1, 3, figsize=(30, 15))
    # Show trajectory in plot
    plot_robot(ax[0], x_all, u_all, t_all)
    ax[0].plot(x_all[:, 0], x_all[:, 2], label='$(x,z)$')
    ax[0].plot(ref_all[:, 0], ref_all[:, 2], 'k--', label='$(x_\\mathrm{r},z_\\mathrm{r})$')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$z$')
    ax[0].set_xlim([-1.5, 1.5])
    ax[0].set_ylim([0, 3])
    ax[0].legend()
    ax[1].plot(t_all, x_all[:, 0], 'b', label='$x$')
    ax[1].plot(t_all, ref_all[:, 0], 'b:', label='$x_\\mathrm{r}$')
    ax[1].plot(t_all, x_all[:, 1], 'b--', label='$\\dot{x}$')
    ax[1].plot(t_all, x_all[:, 2], 'g', label='$z$')
    ax[1].plot(t_all, ref_all[:, 2], 'g:', label='$z_\\mathrm{r}$')
    ax[1].plot(t_all, x_all[:, 3], 'g--', label='$\\dot{z}$')
    ax[1].plot(t_all, x_all[:, 4], 'r', label='$\\theta$')
    ax[1].plot(t_all, x_all[:, 5], 'r--', label='$\\dot{\\theta}$')
    ax[1].set_xlabel('$t$')
    ax[1].legend()
    ax[2].plot(t_all, u_all[:, 0], 'b', label='$u_1$ (left)')
    ax[2].plot(t_all, u_all[:, 1], 'b--', label='$u_2$ (right)')
    ax[2].set_xlabel('$t$')
    ax[2].set_ylabel('$u$')
    ax[2].legend()

    plt.show()


if __name__ == "__main__":
    main()
