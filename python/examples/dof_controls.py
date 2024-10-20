# """
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# DOF control methods example
# ---------------------------
# An example that demonstrates various DOF control methods:
# - Load cartpole asset from an urdf
# - Get/set DOF properties
# - Set DOF position and velocity targets
# - Get DOF positions
# - Apply DOF efforts
# """

# import math
# from isaacgym import gymapi
# from isaacgym import gymutil

# # initialize gym
# gym = gymapi.acquire_gym()

# # parse arguments
# args = gymutil.parse_arguments(description="Joint control Methods Example")

# # create a simulator
# sim_params = gymapi.SimParams()
# sim_params.substeps = 2
# sim_params.dt = 1.0 / 60.0

# sim_params.physx.solver_type = 1
# sim_params.physx.num_position_iterations = 4
# sim_params.physx.num_velocity_iterations = 1

# sim_params.physx.num_threads = args.num_threads
# sim_params.physx.use_gpu = args.use_gpu

# sim_params.use_gpu_pipeline = False
# if args.use_gpu_pipeline:
#     print("WARNING: Forcing CPU pipeline.")

# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# if sim is None:
#     print("*** Failed to create sim")
#     quit()

# # create viewer using the default camera properties
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     raise ValueError('*** Failed to create viewer')

# # add ground plane
# plane_params = gymapi.PlaneParams()
# gym.add_ground(sim, gymapi.PlaneParams())

# # set up the env grid
# num_envs = 4
# spacing = 1.5
# env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
# env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# # add cartpole urdf asset
# asset_root = "../../assets"
# asset_file = "urdf/cartpole.urdf"

# # Load asset with default control type of position for all joints
# asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
# print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
# cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# # initial root pose for cartpole actors
# initial_pose = gymapi.Transform()
# initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
# initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# # Create environment 0
# # Cart held steady using position target mode.
# # Pole held at a 45 degree angle using position target mode.
# env0 = gym.create_env(sim, env_lower, env_upper, 2)
# cartpole0 = gym.create_actor(env0, cartpole_asset, initial_pose, 'cartpole', 0, 1)
# # Configure DOF properties
# props = gym.get_actor_dof_properties(env0, cartpole0)
# props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
# props["stiffness"] = (5000.0, 5000.0)
# props["damping"] = (100.0, 100.0)
# gym.set_actor_dof_properties(env0, cartpole0, props)
# # Set DOF drive targets
# cart_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'slider_to_cart')
# pole_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'cart_to_pole')
# gym.set_dof_target_position(env0, cart_dof_handle0, 0)
# gym.set_dof_target_position(env0, pole_dof_handle0, 0.25 * math.pi)

# # Create environment 1
# # Cart held steady using position target mode.
# # Pole rotating using velocity target mode.
# env1 = gym.create_env(sim, env_lower, env_upper, 2)
# cartpole1 = gym.create_actor(env1, cartpole_asset, initial_pose, 'cartpole', 1, 1)
# # Configure DOF properties
# props = gym.get_actor_dof_properties(env1, cartpole1)
# props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL)
# props["stiffness"] = (5000.0, 0.0)
# props["damping"] = (100.0, 200.0)
# gym.set_actor_dof_properties(env1, cartpole1, props)
# # Set DOF drive targets
# cart_dof_handle1 = gym.find_actor_dof_handle(env1, cartpole1, 'slider_to_cart')
# pole_dof_handle1 = gym.find_actor_dof_handle(env1, cartpole1, 'cart_to_pole')
# gym.set_dof_target_position(env1, cart_dof_handle1, 0)
# gym.set_dof_target_velocity(env1, pole_dof_handle1, -2.0 * math.pi)

# # Create environment 2
# # Cart moving side to side using velocity target mode.
# # Pole held steady using position target mode.
# env2 = gym.create_env(sim, env_lower, env_upper, 2)
# cartpole2 = gym.create_actor(env2, cartpole_asset, initial_pose, 'cartpole', 2, 1)
# # Configure DOF properties
# props = gym.get_actor_dof_properties(env2, cartpole2)
# props["driveMode"] = (gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_POS)
# props["stiffness"] = (0.0, 5000.0)
# props["damping"] = (200.0, 100.0)
# gym.set_actor_dof_properties(env2, cartpole2, props)
# # Set DOF drive targets
# cart_dof_handle2 = gym.find_actor_dof_handle(env2, cartpole2, 'slider_to_cart')
# pole_dof_handle2 = gym.find_actor_dof_handle(env2, cartpole2, 'cart_to_pole')
# gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)
# gym.set_dof_target_position(env2, pole_dof_handle2, 0.0)

# # Create environment 3
# # Cart has no drive mode, but will be pushed around using forces.
# # Pole held steady using position target mode.
# env3 = gym.create_env(sim, env_lower, env_upper, 2)
# cartpole3 = gym.create_actor(env3, cartpole_asset, initial_pose, 'cartpole', 3, 1)
# # Configure DOF properties
# props = gym.get_actor_dof_properties(env3, cartpole3)
# props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_EFFORT)
# props["stiffness"] = (5000.0, 0.0)
# props["damping"] = (100.0, 0.0)
# gym.set_actor_dof_properties(env3, cartpole3, props)
# # Set DOF drive targets
# cart_dof_handle3 = gym.find_actor_dof_handle(env3, cartpole3, 'slider_to_cart')
# pole_dof_handle3 = gym.find_actor_dof_handle(env3, cartpole3, 'cart_to_pole')
# gym.set_dof_target_position(env3, cart_dof_handle3, 0.0)
# gym.apply_dof_effort(env3, pole_dof_handle3, 200)

# # Look at the first env
# cam_pos = gymapi.Vec3(8, 4, 1.5)
# cam_target = gymapi.Vec3(0, 2, 1.5)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# # Simulate
# while not gym.query_viewer_has_closed(viewer):

#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)

#     # Nothing to be done for env 0

#     # Nothing to be done for env 1

#     # Update env 2: reverse cart target velocity when bounds reached
#     pos = gym.get_dof_position(env2, cart_dof_handle2)
#     if pos >= 0.5:
#         gym.set_dof_target_velocity(env2, cart_dof_handle2, -1.0)
#     elif pos <= -0.5:
#         gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)

#     # Update env 3: apply an effort to the pole to keep it upright
#     pos = gym.get_dof_position(env3, pole_dof_handle3)
#     gym.apply_dof_effort(env3, pole_dof_handle3, -pos * 50)

#     # Wait for dt to elapse in real time.
#     # This synchronizes the physics simulation with the rendering rate.
#     gym.sync_frame_time(sim)

# print('Done')

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from isaacgym import gymapi
# from isaacgym import gymutil

# # initialize gym
# gym = gymapi.acquire_gym()

# # parse arguments
# args = gymutil.parse_arguments(description="Joint control Methods Example")

# # create a simulator
# sim_params = gymapi.SimParams()
# sim_params.substeps = 2
# sim_params.dt = 1.0 / 60.0

# sim_params.physx.solver_type = 1
# sim_params.physx.num_position_iterations = 4
# sim_params.physx.num_velocity_iterations = 1

# sim_params.physx.num_threads = args.num_threads
# sim_params.physx.use_gpu = args.use_gpu

# sim_params.use_gpu_pipeline = False
# if args.use_gpu_pipeline:
#     print("WARNING: Forcing CPU pipeline.")

# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# if sim is None:
#     print("*** Failed to create sim")
#     quit()

# # create viewer using the default camera properties
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     raise ValueError('*** Failed to create viewer')

# # add ground plane
# plane_params = gymapi.PlaneParams()
# gym.add_ground(sim, gymapi.PlaneParams())

# # set up the env grid
# spacing = 1.5
# env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
# env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# # add cartpole urdf asset
# asset_root = "../../assets"
# asset_file = "urdf/upper_thormang_copy.urdf"

# # Load asset with default control type of position for all joints
# asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
# print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
# cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# # initial root pose for cartpole actors
# initial_pose = gymapi.Transform()
# initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
# initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# # Create environment 0
# env0 = gym.create_env(sim, env_lower, env_upper, 2)
# cartpole0 = gym.create_actor(env0, cartpole_asset, initial_pose, 'cartpole', 0, 1)
# # Configure DOF properties
# props = gym.get_actor_dof_properties(env0, cartpole0)
# props["driveMode"] = (gymapi.DOF_MODE_POS)
# props["stiffness"] = (100.0)
# props["damping"] = (20.0)
# gym.set_actor_dof_properties(env0, cartpole0, props)
# # Set DOF drive targets
# cart_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'slider_to_cart')

# # Chirp wave parameters
# chirp_start_freq = 0.1  # 起始頻率
# chirp_end_freq = 2.0    # 終止頻率
# chirp_duration = 10.0   # 總持續時間
# t = 0.0                 # 當前時間
# dt = sim_params.dt       # 模擬時間步長
# gym.set_dof_target_position(env0, cart_dof_handle0, 0.0)

# # 儲存控制命令和關節位置狀態
# control_commands = []
# joint_positions = []
# time_steps = []

# # Look at the first env
# cam_pos = gymapi.Vec3(8, 4, 1.5)
# cam_target = gymapi.Vec3(0, 2, 1.5)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# # Chirp wave function
# def chirp_wave(t, f0, f1, duration):
#     k = (f1 - f0) / duration
#     return math.sin(2 * math.pi * (f0 * t + 0.5 * k * t**2))

# # Simulate
# while not gym.query_viewer_has_closed(viewer):

#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     # Update chirp wave control signal
#     control_signal = chirp_wave(t, chirp_start_freq, chirp_end_freq, chirp_duration)
#     gym.set_dof_target_position(env0, cart_dof_handle0, control_signal)

#     # 觀察關節位置
#     dof_state = gym.get_actor_dof_states(env0, cartpole0, gymapi.STATE_POS)
#     joint_position = dof_state['pos'][cart_dof_handle0]

#     # 儲存控制信號和關節位置
#     control_commands.append(control_signal)
#     joint_positions.append(joint_position)
#     time_steps.append(t)

#     # 更新時間
#     t += dt

#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)

#     # Wait for dt to elapse in real time.
#     gym.sync_frame_time(sim)

# print('Simulation done')

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)

# # 將數據轉換為 numpy 陣列以便繪圖
# time_steps = np.array(time_steps)
# control_commands = np.array(control_commands)
# joint_positions = np.array(joint_positions)

# # 繪製控制命令和關節位置在同一張圖上
# plt.figure(figsize=(10, 5))
# plt.plot(time_steps, control_commands, label='Control Command (Chirp Wave)', color='blue')
# plt.plot(time_steps, joint_positions, label='Joint Position Observation', color='orange')
# plt.title(f"Control Command vs Joint Position")
# plt.xlabel("Time (s)")
# plt.ylabel("Value")
# plt.legend()

# plt.tight_layout()
# plt.show()
"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 4
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# add cartpole urdf asset
asset_root = "../../assets"
asset_file = "urdf/upper_thormang_copy.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole0 = gym.create_actor(env0, cartpole_asset, initial_pose, 'cartpole', 0, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cartpole0)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
props["stiffness"] = (5000.0, 5000.0)
props["damping"] = (100.0, 100.0)
gym.set_actor_dof_properties(env0, cartpole0, props)
# Set DOF drive targets
cart_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'slider_to_cart')
pole_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'cart_to_pole')
gym.set_dof_target_position(env0, cart_dof_handle0, 0)
gym.set_dof_target_position(env0, pole_dof_handle0, 0.25 * math.pi)



# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Nothing to be done for env 0

    # Nothing to be done for env 1

    # Update env 2: reverse cart target velocity when bounds reached
    pos = gym.get_dof_position(env2, cart_dof_handle2)
    if pos >= 0.5:
        gym.set_dof_target_velocity(env2, cart_dof_handle2, -1.0)
    elif pos <= -0.5:
        gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)

    # Update env 3: apply an effort to the pole to keep it upright
    pos = gym.get_dof_position(env3, pole_dof_handle3)
    gym.apply_dof_effort(env3, pole_dof_handle3, -pos * 50)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
