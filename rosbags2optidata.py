"""A script that takes a list of rosbag file names and outputs labeled data in the format
    - v_prior_meas 
    - omega_prior_meas
    - v_post_meas
    - omega_post_meas
    - orientations_prior_meas,
    - contact_state

    Each of which is an array of n data points that describes a collision.
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.spatial.transform import Rotation

from pathlib import Path
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.rosbag1 import Reader


def guess_msgtype(path: Path) -> str:
    """Guess message type name from path."""
    name = path.relative_to(path.parents[2]).with_suffix('')
    if 'msg' not in name.parts:
        name = name.parent / 'msg' / name.name
    return str(name)

def numeric_diff(values: np.array, times: np.array,
                 window_size: int) -> np.array:
    diff = np.zeros_like(values)

    for i in range(values.shape[0]):
        k = max([i - window_size, 0])  
        val = np.zeros_like(values[i, :])
        for j in range(k, k+window_size):
            val += (values[j + 1, :] - values[j,:]) / (times[j + 1] - times[j]) 
        
        diff[i, :] = val / window_size

    return diff    

def numeric_att_diff(values: np.array, times: np.array,
                     window_size: int) -> np.array:
    diff = np.zeros([len(values), 3])

    for i in range(values.shape[0]):
        k = max([i - window_size, 0])  
        val = np.zeros(3)
        for j in range(k, k+window_size):
            rot_vec = Rotation.from_matrix(
                (values[j + 1].as_matrix().T @ values[j].as_matrix()).T
            ).as_rotvec()
            val +=  rot_vec / (times[j + 1] - times[j]) 
        
        diff[i, :] = val / window_size

    return diff  

def plot_data(t_mocap, mocap_p, mocap_v, t_collision_start, t_collision_end,
              mocap_col_prior_sample_index, mocap_col_post_lin_sample_index,
              mocap_att, mocap_omega, mocap_col_post_rot_sample_index,
              t_motorForces, motorForces, t_contact, contact, collision_sampling_point):
    fig, axs = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [1.5, 1.5, 1, 0.5]})
    
    axs[0].set_prop_cycle(cycler('color', ["red", "green", "blue"]))
    axs[0].plot(t_mocap, mocap_p, label=[r"$x$",r"$y$",r"$z$"])
    axs[0].plot(t_mocap, mocap_v, label=[r"$\dot{x}$",r"$\dot{y}$",r"$\dot{z}$"], linestyle ="--")
    axs[0].plot(t_mocap[mocap_col_post_lin_sample_index],
                mocap_v[mocap_col_post_lin_sample_index, 2],
                   # Remove the effect of gravity
                   #- (t_mocap[mocap_col_post_lin_sample_index] - t_collision_start) * -9.81),
                marker="x", linestyle="", color="black")
    axs[0].axvspan(t_collision_start, t_collision_end, color="black", alpha=0.2, linestyle=":")
    axs[0].axvline(t_mocap[mocap_col_prior_sample_index], color="black", linestyle=":", label="Prior Sampling Point")
    axs[0].axvline(t_mocap[mocap_col_post_lin_sample_index], color="black", linestyle=":", label=r"Post Sampling Point $v$")
    axs[0].legend()

    axs[1].set_prop_cycle(cycler('color', ["#970808", "#055A5A", "#618D08"]))
    axs[1].step(t_mocap,
                [att.as_euler(seq="xyz") for att in mocap_att], label=[r"$\varphi$",r"$\theta$",r"$\psi$"], where='pre')
    axs[1].step(t_mocap, mocap_omega, label=[r"$\dot{\varphi}$",r"$\dot{\theta}$", r"$\dot{\psi}$"], linestyle="--", where='pre')
    axs[1].axvspan(t_collision_start, t_collision_end, color="black", alpha=0.2, linestyle=":")
    axs[1].axvline(t_mocap[mocap_col_post_rot_sample_index], color="black", linestyle=":", label=r"Post Sampling Point $\omega$")
    #axs[1].set_ylim([-1000, 1000])
    axs[1].legend()

    axs[2].plot(t_motorForces, motorForces, label=[r"$f_0$",r"$f_1$",
                                                   r"$f_2$",r"$f_3$"])
    axs[2].legend()

    axs[3].plot(t_contact, contact * np.array([1,2,3,4,5,6,7,8]),
                marker="o", linestyle="", color="grey")
    axs[3].axvline(t_contact[collision_sampling_point], color="black", linestyle=":", label="Collision Sampling Point")
    axs[3].set_ylim([0.75, 8.25])
    axs[3].grid()
    axs[3].legend()
    axs[3].set_yticks(np.array([1,2,3,4,5,6,7,8]))
    axs[3].set_yticklabels([r"$v_2$",  r"$v_4$", 
                            r"$v_5$",  r"$v_6$", 
                            r"$v_7$",  r"$v_8$", 
                            r"$v_{10}$", r"$v_{12}$"])
    axs[3].set_xlim([t_collision_start - 0.1, t_collision_end + 0.25])
    plt.show()        
    

def rosbag2data(path: str, plot: bool):

    ############## Register non-standard msg types ##############
    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types = {}

    for pathstr in [
        '/home/anton/projects/colliding-drone/LabCode/GeneralCode/ROS/hiperlab_hardware/msg/PoseEulerStamped.msg',
        '/home/anton/projects/colliding-drone/LabCode/GeneralCode/ROS/hiperlab_rostools/msg/estimator_output.msg',
        '/home/anton/projects/colliding-drone/LabCode/GeneralCode/ROS/hiperlab_rostools/msg/estimator_output.msg',
        '/home/anton/projects/colliding-drone/LabCode/GeneralCode/ROS/hiperlab_rostools/msg/mocap_output.msg',
        '/home/anton/projects/colliding-drone/LabCode/GeneralCode/ROS/hiperlab_rostools/msg/radio_command.msg',
        '/home/anton/projects/colliding-drone/LabCode/GeneralCode/ROS/hiperlab_rostools/msg/simulator_truth.msg',
        '/home/anton/projects/colliding-drone/LabCode/GeneralCode/ROS/hiperlab_rostools/msg/telemetry.msg',
    ]:
        msgpath = Path(pathstr)
        msgdef = msgpath.read_text(encoding='utf-8')
        add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))

    typestore.register(add_types)

    ##############################################################
    ############## Load all the data #############################
    ##############################################################

    t_mocap = []
    mocap_p = []
    mocap_att = []

    t_contact = []
    contact = []

    t_motorForces = []
    motorForces = []

    # Create reader instance and open for reading.
    with Reader(path) as reader:
        # Iterate over messages.
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/mocap_output25':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                t_mocap += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                mocap_p += [[msg.posx, msg.posy, msg.posz]]
                mocap_att += [Rotation.from_quat([
                    msg.attq0, msg.attq1, msg.attq2, msg.attq3 
                ], scalar_first=True)]

            if connection.topic == '/telemetry25':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                t_contact += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                contact += [[bool(msg.customPacket1[0] & (0b00000001 << i)) for i in range(8)]]  

                t_motorForces += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                motorForces += [[msg.motorForces[0], msg.motorForces[1],
                                 msg.motorForces[2], msg.motorForces[3]]]          
    
    t_start = min(np.concatenate((t_mocap, t_contact, t_motorForces)))
    t_mocap = np.array(t_mocap) - t_start
    t_contact = np.array(t_contact) - t_start
    t_motorForces = np.array(t_motorForces) - t_start

    linear_dead_time = 0.04
    rotational_dead_time = - 1/100.0

    mocap_p = np.array(mocap_p)
    mocap_att = np.array(mocap_att)
    contact = np.array(contact)
    motorForces = np.array(motorForces)

    mocap_v = numeric_diff(mocap_p, t_mocap, window_size=10)
    mocap_omega = numeric_att_diff(mocap_att, t_mocap, window_size=3)

    ############## Find the timestamp of collision #####################################
    index_collision_start = np.where(np.any(contact, axis=1))[0][0] - 1
    t_collision_start = t_contact[index_collision_start]
    index_collision_end = index_collision_start + np.where(np.all(contact[index_collision_start+1:] == False, axis=1))[0][0] + 1
    t_collision_end = t_contact[index_collision_end]

    ############# Find the data points ############################
    mocap_col_prior_sample_index = np.argmin((t_mocap - t_collision_start)**2)
    mocap_col_post_lin_sample_index = np.argmin((t_mocap - (t_collision_end + linear_dead_time))**2)
    mocap_col_post_rot_sample_index = np.argmin((t_mocap - (t_collision_end + rotational_dead_time))**2)
    collision_sampling_point = index_collision_start + 1

    orientation_prior_meas = mocap_att[mocap_col_prior_sample_index]
    v_prior_meas = mocap_v[mocap_col_prior_sample_index - 1, :]
    print(f"Prior v: {v_prior_meas}")
    omega_prior_meas = mocap_omega[mocap_col_prior_sample_index - 1, :]
    v_post_meas = (mocap_v[mocap_col_post_lin_sample_index, :] 
                   # Remove the effect of gravity
                   #- (t_mocap[mocap_col_post_lin_sample_index] - t_collision_start) * np.array([0, 0, -9.81])
                   ) 
    
    print(f"Post v: {v_post_meas}")
    omega_post_meas = mocap_omega[mocap_col_post_rot_sample_index, :]
    
    contact_state = contact[collision_sampling_point]

    if plot:
        plot_data(t_mocap, mocap_p, mocap_v, t_collision_start, t_collision_end,
              mocap_col_prior_sample_index, mocap_col_post_lin_sample_index,
              mocap_att, mocap_omega, mocap_col_post_rot_sample_index,
              t_motorForces, motorForces, t_contact, contact, collision_sampling_point)

    return (v_prior_meas, omega_prior_meas,
            v_post_meas, omega_post_meas,
            orientation_prior_meas,
            contact_state) 


def getData(plot=False):

    rosbag_paths = ["/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-21-50-25.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-21-58-51.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-21-59-19.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-21-59-39.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-22-00-03.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-22-00-29.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-22-00-56.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-22-01-27.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-22-01-55.bag",
                    "/home/anton/projects/colliding-drone/rosbags/sim_collisions/2024-08-02-22-02-52.bag"]

    v_prior_meas = []
    omega_prior_meas = []
    v_post_meas = []
    omega_post_meas = []
    orientations_prior_meas = []
    contact_state = []

    for i in range(len(rosbag_paths)):
        (v_prior_meas_i, omega_prior_meas_i,
         v_post_meas_i, omega_post_meas_i,
         orientation_prior_meas_i,
         contact_state_i) = rosbag2data(rosbag_paths[i], plot=plot)
        
        v_prior_meas.append(v_prior_meas_i)
        omega_prior_meas.append(omega_prior_meas_i)
        v_post_meas.append(v_post_meas_i)
        omega_post_meas.append(omega_post_meas_i)
        orientations_prior_meas.append(orientation_prior_meas_i)
        contact_state.append(contact_state_i)

    return (np.array(v_prior_meas),
            np.array(omega_prior_meas),
            np.array(v_post_meas),
            np.array(omega_post_meas),
            np.array(orientations_prior_meas),
            np.array(contact_state))