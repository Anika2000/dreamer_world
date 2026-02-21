import gymnasium as gym
import numpy as np
import mujoco
from mujoco.glfw import glfw

class SO101Env(gym.Env):
    def __init__(self, model_path="so101.xml", image_size=64):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.image_size = image_size
        self.n_joints = self.model.nv
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8
        )

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Optional: randomize object positions
        mujoco.mj_forward(self.model, self.data)
        return self.render(mode='rgb_array')

    def step(self, action):
        # Clip action to joint limits
        action = np.clip(action, -1, 1)
        # Convert normalized action to joint velocities
        qvel_cmd = action * 0.5  # scale factor for speed
        self.data.qvel[:] = qvel_cmd
        mujoco.mj_step(self.model, self.data)

        obs = self.render(mode='rgb_array')
        reward = self.compute_reward()
        done = self.is_done()
        info = {}
        return obs, reward, done, info

    def compute_reward(self):
        # Example reward: distance of gripper to target object
        gripper_pos = self.data.site_xpos[self.model.site_name2id('gripper')]
        obj_pos = self.data.body_xpos[self.model.body_name2id('cube')]
        dist = np.linalg.norm(gripper_pos - obj_pos)
        return -dist  # smaller distance = higher reward

    def is_done(self):
        # Example: done if object lifted above threshold
        obj_z = self.data.body_xpos[self.model.body_name2id('cube')][2]
        return obj_z > 0.2

    def render(self, mode='rgb_array'):
        width, height = self.image_size, self.image_size
        camera_id = self.model.camera_name2id('topview')
        img = mujoco.mjr_render_to_pixels(self.model, self.data, camera_id, width, height)
        return img