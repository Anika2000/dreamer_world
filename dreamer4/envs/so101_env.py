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
        self.viewer = None
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
    
    def compute_reward(self):
        # get IDs
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'gripper')
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        # get positions
        gripper_pos = self.data.xpos[gripper_id]  # site positions are included in data.xpos
        cube_pos = self.data.xpos[cube_id]        # body positions also in data.xpos
        dist = np.linalg.norm(gripper_pos - cube_pos)
        return -dist

    def is_done(self):
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        cube_z = self.data.xpos[cube_id][2]
        return cube_z > 0.2

    def render(self, mode='rgb_array'):
        self.renderer.update_scene(self.data, self.cam)
        # Now render, the camera info is implicitly part of the scene
        img = self.renderer.render()
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img