import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, LinearConstraint

from rosbags2optidata import getData

tensegrity_rod_length = 0.22 # [m]
r = tensegrity_rod_length * (np.array([
   [1.00, 0.50, 0.25],
   [1.00, 0.50, 0.75],
   [0.25, 0.00, 0.50],
   [0.25, 1.00, 0.50],
   [0.75, 0.00, 0.50],
   [0.75, 1.00, 0.50],
   [0.50, 0.25, 1.00],
   [0.50, 0.75, 1.00]]
   ) - 0.5)

def pred(v_prior: np.array,                       # Pre-collision velocity
         omega_prior: np.array,                   # Pre-collision angular velocity
         R: Rotation,                             # Pre-collision attitude
         contact_state: list[bool],               # Contact State
         e: float,                                # Coefficient of restitution
         mu: float,                               # Friction coefficient
         n = np.array([0, -1, 0], dtype=float),   # Wall normal
         m = 0.361,                               # Mass in [kg] 
         I = np.diag([6.21e-4, 9.62e-4, 9.22e-4]) # Inertia Matrix [kg*m^2]
         ):
    
    # Init the deltaV and deltaOmega
    deltaV, deltaOmega = np.zeros(3), np.zeros(3)

    # Init the counter
    contact_points = 0

    # Iterate over all contact state
    for i in range(len(contact_state)):
        # If we are in contact we compute the contribution
        if contact_state[i]:
            # Increment contact point counter
            contact_points += 1

            # Compute velocities and directions
            r_i = R.as_matrix() @ r[i]
            v_node = v_prior + np.cross(r_i, omega_prior)
            v_normal = np.dot(n, v_node) * n 
            v_tangential = v_node - v_normal

            if np.linalg.norm(v_tangential) > 1e-3:
                t = v_tangential / np.linalg.norm(v_tangential)
            else:
                t = np.array([1, 0, 0])
            # Compute Impulse
            j_i = ((- (1 + e) * np.linalg.norm(v_normal) /
                    (1.0 / m + np.dot(n, np.linalg.inv(I) 
                                    @ np.cross(np.cross(r_i, n),
                                               r_i
                                             )
                                )
                    )
                ) 
                * (n - mu * t)
            )

            deltaV += 1.0 / m * j_i
            deltaOmega += np.linalg.inv(I) @ np.cross(r_i, j_i)

        # Otherwise we simply continue
        else:
            continue

    # Normalize the deltas to the number of contacts
    if contact_points > 0:
        deltaV /= contact_points
        deltaOmega /= contact_points

    # Return the change in velocities
    return deltaV, deltaOmega

def run_sys_id(v_prior_meas, omega_prior_meas,
               v_post_meas, omega_post_meas,
               orientations_prior_meas,
               contact_state):
    
    # Define Cost Function
    def f(x, w_v = 1.0, w_omega = 0.0):
        n = len(contact_state)
        e = x[0]
        mu = x[1]
        v_pred = np.zeros_like(v_prior_meas)
        omega_pred = np.zeros_like(omega_prior_meas)
        for i in range(n):
        # True post collision vel
            v_pred[i, :], omega_pred[i, :] = pred(v_prior_meas[i, :],
                                                omega_prior_meas[i, :],
                                                orientations_prior_meas[i],
                                                contact_state[i],
                                                e=e, mu=mu)
        
        errors = (w_v * np.linalg.norm(v_post_meas - v_pred, axis=0) +
                w_omega * np.linalg.norm(omega_post_meas - omega_pred, axis=0)) 
        return np.sum(errors, axis=0) / n

    # Define and Run optimization
    const = LinearConstraint(A=np.eye(2), lb=np.zeros(2))
    res = minimize(fun=f, x0=[0.5, 0.5], constraints=(const,))

    return res


def dummyData():
    n = 20

    sigma_v = 0.25
    sigma_omega = 0.5
    sigma_R = np.deg2rad(1)

    e_true = 0.8
    mu_true = 0.1

    v_prior = np.concatenate([[np.linspace(start=0.1, stop=1, num=n)],
                              [np.linspace(start=0.1, stop=5, num=n)],
                               np.zeros([1, n])], axis=0)
    v_prior_noisy = v_prior + np.random.normal(0.0, sigma_v, size=(3, n))

    omega_prior = np.zeros([3, n])
    omega_prior_noisy = omega_prior + np.random.normal(0.0, sigma_omega, size=(3, n))

    orientations_prior = [Rotation.from_euler(seq="xyz",
                                            angles=np.deg2rad([
                                                5, 0, 0
                                            ])) for i in range(n)]
    orientations_prior_noisy = [orientations_prior[i] * Rotation.from_euler(seq="xyz",
                                    angles=np.random.normal(0.0, sigma_R, size=3)) for i in range(n)] 

    contact_state = [[False, False, False, True, False, True, False, False]
                    for i in range(n)]

    v_post = np.zeros_like(v_prior)
    omega_post = np.zeros_like(omega_prior)

    v_post_noisy = np.zeros_like(v_prior)
    omega_post_noisy = np.zeros_like(omega_prior)

    for i in range(n):
        # True post collision vel
        v_post[:, i], omega_post[:, i] = pred(v_prior[:, i],
                                            omega_prior[:, i],
                                            orientations_prior[i],
                                            contact_state[i],
                                            e=e_true, mu=mu_true)
        # Corrupt with noise
        v_post_noisy[:, i] = v_post[:, i] + np.random.normal(loc=0.0, scale=sigma_v, size=3) 
        omega_post_noisy[:, i] = omega_post[:, i] + np.random.normal(loc=0.0, scale=sigma_omega, size=3) 

                        
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].set_ylabel("Post-Collision Velocity" + r"[$m/s$]")
    axs[0].plot(v_prior[1, :], v_post[:, :].T, label=[r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"])
    axs[0].plot(v_prior[1, :], v_post_noisy[:, :].T, label=[r"$\dot{x}_{meas}$", r"$\dot{y}_{meas}$", r"$\dot{z}_{meas}$"])
    axs[0].legend()

    axs[1].set_ylabel("Post-Collision Rates" + r"[$^\circ/s$]")
    axs[1].set_xlabel("Pre-Collision Velocity Norm " + r"$||\mathbf{v}^{-}||$ [$m/s$]")
    axs[1].plot(v_prior[1, :], omega_post[:, :].T, label=[r"$\dot{\varphi}$", r"$\dot{\theta}$", r"$\dot{\psi}$"])
    axs[1].plot(v_prior[1, :], omega_post_noisy[:, :].T, label=[r"$\dot{\varphi}_{meas}$",
                                                                r"$\dot{\theta}_{meas}$",
                                                                r"$\dot{\psi}_{meas}$"])
    axs[1].legend()
    fig.tight_layout()

    if False:
        plt.show()

    return (v_prior_noisy, omega_prior_noisy,
            v_post_noisy, omega_post_noisy,
            orientations_prior_noisy,
            contact_state)


def run_sys_id_w_dummy_data():
    return run_sys_id(*dummyData())

def run_sys_id_w_data():
    return run_sys_id(*getData())

def main():
    #res = run_sys_id_w_dummy_data()
    res = run_sys_id_w_data()
    print(res)
    print(f"Error: {np.abs(res.x - [0.8, 0.1]) / np.array([0.8, 0.1])}%")

if __name__ == "__main__":
    main()