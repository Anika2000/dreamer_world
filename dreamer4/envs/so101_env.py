import gymnasium as gym
import numpy as np
import mujoco
import cv2

#(below notes are from Feb 21, 2026)
#have to pip instal mujoco by yourself thats why its not in requirements.txt 


class SO101Env(gym.Env):
    def __init__(self, model_path="", image_size=64, render_on=True):
        super().__init__()
        self.render_on = render_on # Store the value
        model_path = "dreamer4/envs/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 20
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "topview")
        if self.camera_id!= -1:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.camera_id
        self.renderer = mujoco.Renderer(self.model)
        self.image_size = image_size
        self.n_joints = self.model.nv
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8
        )
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        obs = self.render()
        info = {}
        return obs, info

    #Actor output → scaled → data.ctrl → MuJoCo physics → joint motion
    # Position actuators work like this:
    #     - data.ctrl[i] = desired joint position
    #     - MuJoCo applies internal PD controller
    #     - Physics computes forces
    #     - Joint moves naturally
    def step(self, action):
        """
        Executes one simulation step using MuJoCo actuators correctly.

        IMPORTANT FIX:
        We no longer write to qvel directly.
        Instead, we send control signals to the actuators via data.ctrl.
        This allows MuJoCo to apply proper position control dynamics.
        """

        # 1 Clip action to [-1, 1]
        # Your actor outputs tanh, so this keeps safety in case of numerical drift.
        action = np.clip(action, -1.0, 1.0)

        # 2 Convert normalized action [-1,1] → actuator control range
        # MuJoCo actuators have specific ctrlrange values defined in XML.
        # Example from your XML:
        #   <position name="shoulder_pan" ctrlrange="-1.91986 1.91986"/>
        #
        # model.actuator_ctrlrange gives us:
        #   shape: (num_actuators, 2)
        #   [:, 0] = minimum
        #   [:, 1] = maximum

        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]

        # Scale action from [-1,1] → [ctrl_low, ctrl_high]
        scaled_action = ctrl_low + (action + 1.0) * 0.5 * (ctrl_high - ctrl_low)

        # 3 Send control signals to actuators
        # This is the CORRECT way to control MuJoCo actuators.
        # MuJoCo will now apply position control physics properly.
        self.data.ctrl[:] = scaled_action

        # 4 Step the physics simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 5 Get observation (rendered image)
        obs = self.render()

        # 6 Compute reward
        reward = self.compute_reward()

        # 7 Check termination
        terminated = self.is_done()

        # 8 No time truncation yet
        truncated = False

        info = {}

        return obs, reward, terminated, truncated, info
    

    #we give reward here whenever the pick and place works
    # need a reward that encourages the behavior you want. 
    # The world model and actor learn from images what sequence of actions produces high reward.
    def compute_reward(self):
        # get IDs
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        # Safety check
        if gripper_id == -1:
            raise ValueError("Body 'gripper' not found in model.")
        if cube_id == -1:
            raise ValueError("Body 'cube' not found in model.")
        # get positions
        gripper_pos = self.data.xpos[gripper_id]  # site positions are included in data.xpos
        cube_pos = self.data.xpos[cube_id]        # body positions also in data.xpos
            
        dist_horizontal = np.linalg.norm(gripper_pos[:2] - cube_pos[:2])  # xy-plane distance
        dist_vertical = cube_pos[2] - 0.05  # height above table (assuming table z=0.05)

        # 4. Reward shaping
        reach_reward = -dist_horizontal                  # closer in xy-plane → higher reward
        lift_reward = max(dist_vertical, 0.0)           # reward for lifting above table
        success_reward = 1.0 if dist_vertical > 0.15 else 0.0  # cube lifted above threshold

        # 5. Combine rewards
        total_reward = reach_reward + lift_reward + success_reward

        return total_reward

    def is_done(self):
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        cube_pos = self.data.xpos[cube_id]
        table_height = 0.05
        success_threshold = 0.15
        terminated = cube_pos[2] > table_height + success_threshold
        return terminated

    def render(self, mode='rgb_array'):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam)
        # Now render, the camera info is implicitly part of the scene
        img = self.renderer.render()
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img