# """
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Joint Monkey
# ------------
# - Animates degree-of-freedom ranges for a given asset.
# - Demonstrates usage of DOF properties and states.
# - Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
# """

# import math
# import numpy as np
# from isaacgym import gymapi, gymutil


# def clamp(x, min_value, max_value):
#     return max(min(x, max_value), min_value)

# # simple asset descriptor for selecting from a list


# class AssetDesc:
#     def __init__(self, file_name, flip_visual_attachments=False):
#         self.file_name = file_name
#         self.flip_visual_attachments = flip_visual_attachments


# asset_descriptors = [
#      AssetDesc("urdf/upper_thormang_copy.urdf", False),
#     AssetDesc("mjcf/nv_ant.xml", False),
#     AssetDesc("urdf/cartpole.urdf", False),
#     AssetDesc("urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf", False),
#     AssetDesc("urdf/franka_description/robots/franka_panda.urdf", True),
#     AssetDesc("urdf/kinova_description/urdf/kinova.urdf", False),
#     AssetDesc("urdf/anymal_b_simple_description/urdf/anymal.urdf", True),
#     AssetDesc("urdf/upper_thormang_copy.urdf", False),
#     AssetDesc("urdf/gogoro_and_thormang3.urdf", False),
# ]


# # parse arguments
# args = gymutil.parse_arguments(
#     description="Joint monkey: Animate degree-of-freedom ranges",
#     custom_parameters=[
#         {"name": "--asset_id", "type": int, "default": 0, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
#         {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
#         {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

# if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
#     print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
#     quit()


# # initialize gym
# gym = gymapi.acquire_gym()

# # configure sim
# sim_params = gymapi.SimParams()
# sim_params.dt = dt = 1.0 / 60.0
# if args.physics_engine == gymapi.SIM_FLEX:
#     pass
# elif args.physics_engine == gymapi.SIM_PHYSX:
#     sim_params.physx.solver_type = 1
#     sim_params.physx.num_position_iterations = 6
#     sim_params.physx.num_velocity_iterations = 0
#     sim_params.physx.num_threads = args.num_threads
#     sim_params.physx.use_gpu = args.use_gpu

# sim_params.use_gpu_pipeline = False
# if args.use_gpu_pipeline:
#     print("WARNING: Forcing CPU pipeline.")

# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
# if sim is None:
#     print("*** Failed to create sim")
#     quit()

# # add ground plane
# plane_params = gymapi.PlaneParams()
# gym.add_ground(sim, plane_params)

# # create viewer
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     print("*** Failed to create viewer")
#     quit()

# # load asset
# asset_root = "../../assets"
# asset_file = asset_descriptors[args.asset_id].file_name

# asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
# asset_options.use_mesh_materials = True

# print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
# asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# # get array of DOF names
# dof_names = gym.get_asset_dof_names(asset)
# actuated_dofs = np.array([[0,1,2,3,4,5,6]])
# # get array of DOF properties
# dof_props = gym.get_asset_dof_properties(asset)
# dof_props['driveMode'][actuated_dofs] = gymapi.DOF_MODE_POS
# dof_props['stiffness'][actuated_dofs] = 100.0
# dof_props['damping'][actuated_dofs] = 20.0

# # create an array of DOF states that will be used to update the actors
# num_dofs = gym.get_asset_dof_count(asset)
# dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# # get list of DOF types
# dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# # get the position slice of the DOF state array
# dof_positions = dof_states['pos']

# # get the limit-related slices of the DOF properties array
# stiffnesses = dof_props['stiffness']
# dampings = dof_props['damping']
# armatures = dof_props['armature']
# has_limits = dof_props['hasLimits']
# lower_limits = dof_props['lower']
# upper_limits = dof_props['upper']

# # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
# defaults = np.zeros(num_dofs)
# speeds = np.zeros(num_dofs)
# for i in range(num_dofs):
#     if has_limits[i]:
#         if dof_types[i] == gymapi.DOF_ROTATION:
#             lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
#             upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
#         # make sure our default position is in range
#         if lower_limits[i] > 0.0:
#             defaults[i] = lower_limits[i]
#         elif upper_limits[i] < 0.0:
#             defaults[i] = upper_limits[i]
#     else:
#         # set reasonable animation limits for unlimited joints
#         if dof_types[i] == gymapi.DOF_ROTATION:
#             # unlimited revolute joint
#             lower_limits[i] = -math.pi
#             upper_limits[i] = math.pi
#         elif dof_types[i] == gymapi.DOF_TRANSLATION:
#             # unlimited prismatic joint
#             lower_limits[i] = -1.0
#             upper_limits[i] = 1.0
#     # set DOF position to default
#     dof_positions[i] = defaults[i]
#     # set speed depending on DOF type and range of motion
#     if dof_types[i] == gymapi.DOF_ROTATION:
#         speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
#     else:
#         speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

# # Print DOF properties
# for i in range(num_dofs):
#     print("DOF %d" % i)
#     print("  Name:     '%s'" % dof_names[i])
#     print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
#     print("  Stiffness:  %r" % stiffnesses[i])
#     print("  Damping:  %r" % dampings[i])
#     print("  Armature:  %r" % armatures[i])
#     print("  Limited?  %r" % has_limits[i])
#     if has_limits[i]:
#         print("    Lower   %f" % lower_limits[i])
#         print("    Upper   %f" % upper_limits[i])

# # set up the env grid
# num_envs = 36
# num_per_row = 6
# spacing = 2.5
# env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
# env_upper = gymapi.Vec3(spacing, spacing, spacing)

# # position the camera
# cam_pos = gymapi.Vec3(17.2, 2.0, 16)
# cam_target = gymapi.Vec3(5, -2.5, 13)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# # cache useful handles
# envs = []
# actor_handles = []

# print("Creating %d environments" % num_envs)
# for i in range(num_envs):
#     # create env
#     env = gym.create_env(sim, env_lower, env_upper, num_per_row)
#     envs.append(env)

#     # add actor
#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
#     pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

#     actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
#     actor_handles.append(actor_handle)
    


#     # add actor
#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
#     pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    
#     actuated_dofs = np.array([11,12,13,14,15,16,17])
#     dof_props['driveMode'][actuated_dofs] = gymapi.DOF_MODE_POS
#     dof_props['stiffness'][actuated_dofs] = 0.0
#     dof_props['damping'][actuated_dofs] = 0.0
#     dof_props['stiffness'].fill(0.0)
#     dof_props['damping'].fill(0.0)
#     # set default DOF positions
#     gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

#     # 設置剛度和阻尼屬性
#     gym.set_actor_dof_properties(env, actor_handle, dof_props)
#     # set default DOF positions
#     gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

# # joint animation states
# ANIM_SEEK_LOWER = 1
# ANIM_SEEK_UPPER = 2
# ANIM_SEEK_DEFAULT = 3
# ANIM_FINISHED = 4

# # initialize animation state
# anim_state = ANIM_SEEK_LOWER
# current_dof = 0
# print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

# while not gym.query_viewer_has_closed(viewer):

#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     speed = speeds[current_dof]

#     # animate the dofs
#     if anim_state == ANIM_SEEK_LOWER:
#         dof_positions[current_dof] -= speed * dt
#         if dof_positions[current_dof] <= lower_limits[current_dof]:
#             dof_positions[current_dof] = lower_limits[current_dof]
#             anim_state = ANIM_SEEK_UPPER
#     elif anim_state == ANIM_SEEK_UPPER:
#         dof_positions[current_dof] += speed * dt
#         if dof_positions[current_dof] >= upper_limits[current_dof]:
#             dof_positions[current_dof] = upper_limits[current_dof]
#             anim_state = ANIM_SEEK_DEFAULT
#     if anim_state == ANIM_SEEK_DEFAULT:
#         dof_positions[current_dof] -= speed * dt
#         if dof_positions[current_dof] <= defaults[current_dof]:
#             dof_positions[current_dof] = defaults[current_dof]
#             anim_state = ANIM_FINISHED
#     elif anim_state == ANIM_FINISHED:
#         dof_positions[current_dof] = defaults[current_dof]
#         current_dof = (current_dof + 1) % num_dofs
#         anim_state = ANIM_SEEK_LOWER
#         print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

#     if args.show_axis:
#         gym.clear_lines(viewer)

#     # clone actor state in all of the environments
#     for i in range(num_envs):
#         gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

#         if args.show_axis:
#             # get the DOF frame (origin and axis)
#             dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], current_dof)
#             frame = gym.get_dof_frame(envs[i], dof_handle)

#             # draw a line from DOF origin along the DOF axis
#             p1 = frame.origin
#             p2 = frame.origin + frame.axis * 0.7
#             color = gymapi.Vec3(1.0, 0.0, 0.0)
#             gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)

#     # Wait for dt to elapse in real time.
#     # This synchronizes the physics simulation with the rendering rate.
#     gym.sync_frame_time(sim)

# print("Done")

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)
import math
import numpy as np
import matplotlib.pyplot as plt
from isaacgym import gymapi, gymutil
from scipy.signal import chirp  # 导入chirp函数

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

asset_descriptors = [
    AssetDesc("urdf/upper_thormang_copy.urdf", False),
    # ... 其他资产
]

# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges with chirp wave",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
    quit()

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 0.05
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "../../assets"
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# get number of DOFs
num_dofs = gym.get_asset_dof_count(asset)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# 初始化 chirp 波的參數
chirp_duration = 30.0   # Chirp波的持续时间 (秒)
f0 = 0.005              # 起始频率 (Hz)
f1 = 2                  # 结束频率 (Hz)
chirp_time = np.linspace(0, chirp_duration, int(chirp_duration / dt))  # 時間序列

# 生成 chirp 波
chirp_wave = chirp(chirp_time, f0=f0, f1=f1, t1=chirp_duration, method='linear')

# 保存控制命令和关节位置状态
control_commands = []
joint_positions = []
time_steps = []

# 打印 DOF 属性和索引
print("DOF Indices and Names:")
for i in range(num_dofs):
    print(f"DOF {i}: Name = {dof_names[i]}")
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# 定义要驱动的 DOF（验证索引）
dof = 15
actuated_dofs = np.array([dof])  #elbow joint

# set up the env grid
num_envs = 36
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # 为每个 actor 创建 dof_props 的副本
    dof_props_actor = gym.get_asset_dof_properties(asset)
    # 设置驱动模式、刚度和阻尼
    for dof_index in range(num_dofs):
        if dof_index in actuated_dofs:
            dof_type = dof_types[dof_index]
            if dof_type == gymapi.DOF_ROTATION or dof_type == gymapi.DOF_TRANSLATION:
                dof_props_actor['driveMode'][dof_index] = gymapi.DOF_MODE_POS
                dof_props_actor['damping'][dof_index] = 100#8200.0
                dof_props_actor['stiffness'][dof_index] =2000#59500.0
                dof_props_actor['effort'][dof_index] = 10.7 #4407
                dof_props_actor['velocity'][dof_index] = 5.24 #3.03
            else:
                print(f"DOF {dof_index} does not support position control.")
        else:
            dof_props_actor['driveMode'][dof_index] = gymapi.DOF_MODE_NONE

    gym.set_actor_dof_properties(env, actor_handle, dof_props_actor)

# 初始化目标位置
dof_targets = np.zeros(num_dofs, dtype=np.float32)

# simulation loop
t = 0.0  # 重置時間
while not gym.query_viewer_has_closed(viewer):
    # 控制 DOF 的 chirp 波控制
    current_control = []
    current_positions = []
    dof_positions = []

    for dof_index in actuated_dofs:
        # 取得當前 chirp 波中的值
        chirp_index = int(t / dt)
        control_signal = chirp_wave[chirp_index] if chirp_index < len(chirp_wave) else chirp_wave[-1]

        target_position = clamp(control_signal, lower_limits[dof_index], upper_limits[dof_index])
        dof_pos = gym.get_dof_position(env, dof)  # 假設這裡是取得相應的關節位置
        dof_targets[dof_index] = target_position

        # 保存當前控制信號和關節位置
        current_control.append(control_signal)
        current_positions.append(target_position)
        dof_positions.append(dof_pos)
    
    control_commands.append(current_control)
    joint_positions.append(dof_positions)
    time_steps.append(t)
    
    # 增加時間計數器
    t += dt

    # 設定關節的目標位置
    for i in range(num_envs):
        gym.set_actor_dof_position_targets(envs[i], actor_handles[i], dof_targets)

    # 執行模擬
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新視覺化工具
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待實際的時間步長，這會同步物理模擬與渲染速率
    gym.sync_frame_time(sim)

print("Simulation done")

# 銷毀 viewer 和模擬
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


# 讀取從 ROS 紀錄的數據
commanded_positions_ros = np.load('commanded_positions.npy')
observed_positions_ros = np.load('dynamixel_positions.npy')
time_steps_ros = np.load('time_steps.npy')

# 假設模擬的時間步長和時間點
time_steps_simulation = np.array(time_steps)  # 模擬的時間點

# 插值 ROS 數據以對應模擬的時間點
commanded_positions_ros_interp = np.interp(time_steps_simulation, time_steps_ros, commanded_positions_ros)
observed_positions_ros_interp = np.interp(time_steps_simulation, time_steps_ros, observed_positions_ros)

# 確保 control_commands 和 joint_positions 是 numpy 陣列
control_commands = np.array(control_commands)
joint_positions = np.array(joint_positions)

# 繪製比較圖表
plt.figure(figsize=(10, 6))

# 模擬數據
plt.plot(time_steps_simulation, control_commands[:, 0], label='Simulated Control Command', color='blue')
plt.plot(time_steps_simulation, joint_positions[:, 0], label='Simulated Joint Position', color='orange')

# ROS 數據
plt.plot(time_steps_simulation, commanded_positions_ros_interp, label='ROS Commanded Positions', linestyle='--', color='green')
plt.plot(time_steps_simulation, observed_positions_ros_interp, label='ROS Observed Positions', linestyle='--', color='red')

# 標題和標籤
plt.title("Comparison of Simulated and ROS Data for DOF")
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.legend()
plt.grid(True)
plt.show()