import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

import h5py

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

import json

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    libero_raw_data_dir: str = "/home/miki/LIBERO/libero_dataset/datasets/libero_spatial"

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    

    # fmt: on



class RolloutWorkerOpenVLA():
    def __init__(self, cfg: GenerateConfig):
        self.cfg = cfg
        self.cfg.unnorm_key = cfg.task_suite_name
        self.model = get_model(cfg)
        self.model_family = cfg.model_family
        self.processor = get_processor(cfg)
        self.task_suite = self.get_task_suite()
        self.num_tasks_in_suite = self.task_suite.n_tasks
        self.resize_size = get_image_resize_size(cfg)
        self.envs, self.task_descriptions = self.make_all_envs()
        self.rollout_batch_size = self.num_tasks_in_suite
        self.initial_obs = [None] * self.rollout_batch_size
        self.initial_achieved_goal = [None] * self.rollout_batch_size
        self.initial_desired_goal = [None] * self.rollout_batch_size
        self.libero_raw_data_dir = cfg.libero_raw_data_dir
        self.task_desired_goals = self.get_task_desired_goals_json()
        self.task_suite_name = cfg.task_suite_name
        self.num_steps_wait = cfg.num_steps_wait
        self.reset_all_rollouts(0)
        
        if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
        elif cfg.task_suite_name == "libero_object":
            max_steps = 280  # longest training demo has 254 steps
        elif cfg.task_suite_name == "libero_goal":
            max_steps = 300  # longest training demo has 270 steps
        elif cfg.task_suite_name == "libero_10":
            max_steps = 520  # longest training demo has 505 steps
        elif cfg.task_suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps
        
        self.T = max_steps

    def get_task_desired_goals_json(self):
      with open("/home/miki/openvla/data/spatial_task_desired_goals.json", "r") as f:
        data = json.load(f)
        return data
        
    def get_task_suite(self):
      benchmark_dict = benchmark.get_benchmark_dict()
      return benchmark_dict[self.cfg.task_suite_name]()
    
    def make_all_envs(self):
      envs = []
      task_descriptions = []
      for task_id in range(self.num_tasks_in_suite):
        task = self.task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, self.model_family, resolution=256)
        envs.append(env)
        task_descriptions.append(task_description)
      
      return envs, task_descriptions
    
    def reset_rollout(self, task_id, episode_id):
      task = self.task_suite.get_task(task_id)
      task_description = "_".join(self.task_descriptions[task_id].split(" "))
      
      # ex {'akita_black_bowl_1_main': [0.061956970218480775, 0.19921577625065975, 0.9075433073452307]}
      desired_goal_dict = self.task_desired_goals[task_description][episode_id]
      
      # akita_black_bowl_1_main
      target_object = list(desired_goal_dict.keys())[0]
      
      # [0.061956970218480775, 0.19921577625065975, 0.9075433073452307]
      desired_goal = list(desired_goal_dict.values())[0]
      
      # target nameの取り出し方はlibero taskによって違う
      if self.task_suite_name == "libero_spatial":
        # akita_black_bowl_1_pos
        target_object_pos = target_object.replace("_main", "_pos")
        target_object_quat = target_object.replace("_main", "_quat")
                  
      self.initial_desired_goal[task_id] = desired_goal
      
      orig_data_path = os.path.join(self.libero_raw_data_dir, f"{task.name}_demo.hdf5")
      assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
      orig_data_file = h5py.File(orig_data_path, "r")
      orig_data = orig_data_file["data"]
      
      demo_data = orig_data[f"demo_{episode_id}"]
      orig_actions = demo_data["actions"][()]
      orig_states = demo_data["states"][()]
            
      self.envs[task_id].reset()
      initial_obs = self.envs[task_id].set_init_state(orig_states[0])
      initial_achieved_goal = np.concatenate([initial_obs[target_object_pos], initial_obs[target_object_quat]])
      self.initial_achieved_goal = initial_achieved_goal
      return initial_obs, desired_goal, initial_achieved_goal, target_object
      
    def reset_all_rollouts(self, episode_id):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for task_id in range(self.rollout_batch_size):
            self.reset_rollout(task_id, episode_id)
    
    def warm_up_env(self, task_id, episode_id):
      self.reset_rollout(task_id, episode_id)
      t = 0
      if t < self.num_steps_wait:
        obs, reward, done, info = self.envs[task_id].step(get_libero_dummy_action(self.model_family))
        t += 1
      
      return obs
            
    def get_action(self, task_id, episode_id, obs):
      
      img = get_libero_image(obs, self.resize_size)
      
      task_description = self.task_descriptions[task_id]
      
      observation = {
          "full_image": img,
          "state": np.concatenate(
              (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
          ),
      }
      action = get_action(
          self.cfg,
          self.model,
          observation,
          task_description,
          processor=self.processor,
      )
      return action
    
    def step(self, task_id, action):
      obs, reward, done, info = self.envs[task_id].step(action.tolist())
      return obs, reward, done, info
    
    def get_all_state(self):
        return self.__dict__
    
import numpy as np
from collections import OrderedDict
      

@draccus.wrap()
def example(cfg: GenerateConfig):
  task_id = 0
  episode_id = 0
  openvla_rollout_worker = RolloutWorkerOpenVLA(cfg)  
  obs = openvla_rollout_worker.warm_up_env(task_id, episode_id)
  action = openvla_rollout_worker.get_action(task_id, episode_id, obs)
  obs, reward, done, info = openvla_rollout_worker.step(task_id, action)
  print(openvla_rollout_worker.get_all_state())
  
@draccus.wrap()
def main(cfg: GenerateConfig):
  openvla_rollout_worker = RolloutWorkerOpenVLA(cfg)
  return openvla_rollout_worker
  

from fastapi import FastAPI, Request, Response
import uvicorn
from fastapi.encoders import jsonable_encoder
import pickle


app = FastAPI()

# environments を Python 内部に保持
envs = {}
openvla_worker = main()  # 既存の OpenVLA worker

@app.post("/reset")
async def reset(request: Request):
    body = await request.json()
    task_id = body["task_id"]
    episode_id = body["episode_id"]
    
    initial_obs, desired_goal, initial_achieved_goal, target_object = openvla_worker.reset_rollout(task_id, episode_id)  
    payload = {
        "initial_obs": initial_obs,
        "desired_goal": desired_goal,
        "initial_achieved_goal": initial_achieved_goal,
        "target_object": target_object,
    }

    return Response(
        content=pickle.dumps(payload),
        media_type="application/octet-stream",
    )
    
@app.post("/set_init_state")
def set_init_state(request: dict):
    env_id = request["env_id"]
    init_state = request["init_state"]
    obs = openvla_worker.envs[env_id].set_init_state(init_state)
    return {"obs": obs}

@app.post("/step")
async def step(request: Request):
    body = await request.body()
    payload_dict = pickle.loads(body)
    env_id = payload_dict["env_id"]
    action = payload_dict["action"]
    obs, reward, done, info = openvla_worker.envs[env_id].step(action)
    response_bytes = pickle.dumps({"obs": obs, "reward": reward, "done": done, "info": info})
    return Response(content=response_bytes, media_type="application/octet-stream")

@app.post("/get_base_action")
async def get_base_action(request: Request):
    body = await request.body()
    payload_dict = pickle.loads(body)
    task_id = payload_dict["task_id"]
    episode_id = payload_dict["episode_id"]
    obs = payload_dict["obs"]
    action = openvla_worker.get_action(task_id, episode_id, obs)
    response_bytes = pickle.dumps({"action": action})
    return Response(content=response_bytes, media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
