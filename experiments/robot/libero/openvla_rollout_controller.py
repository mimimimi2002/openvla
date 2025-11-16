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
        self.model = get_model(cfg)
        self.processor = get_processor(cfg)
        self.task_suite = self.get_task_suite(cfg)
        self.num_tasks_in_suite = self.task_suite.n_tasks
        self.resize_size = get_image_resize_size(cfg)
        self.envs, self.task_descriptions = self.make_all_envs(cfg)
        self.rollout_batch_size = self.num_tasks_in_suite
        self.initial_obs = [None] * self.rollout_batch_size
        self.initial_achieved_goal = [None] * self.rollout_batch_size
        self.initial_desired_goal = None
        self.libero_raw_data_dir = cfg.libero_raw_data_dir
        self.task_desired_goals = self.get_task_desired_goals_json()
        self.reset_all_rollouts(0)
    
    def get_task_desired_goals_json(self):
      with open("/home/miki/openvla/data/spatial_task_desired_goals.json", "r") as f:
        data = json.load(f)
        return data
        
    def get_task_suite(self, cfg: GenerateConfig):
      benchmark_dict = benchmark.get_benchmark_dict()
      return benchmark_dict[cfg.task_suite_name]()
    
    def make_all_envs(self, cfg: GenerateConfig):
      envs = []
      task_descriptions = []
      for task_id in range(self.num_tasks_in_suite):
        task = self.task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        envs.append(env)
        task_descriptions.append(task_description)
      
      return envs, task_descriptions
    
    def reset_rollout(self, task_idx, episode_idx):
      task = self.task_suite.get_task(task_idx)
      task_description = "_".join(self.task_descriptions[task_idx].split(" "))
      
      print("task_description", task_description)
      
      # need to be fixed
      desired_goal = self.task_desired_goals[task_description][0][episode_idx]
      
      orig_data_path = os.path.join(self.libero_raw_data_dir, f"{task.name}_demo.hdf5")
      assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
      orig_data_file = h5py.File(orig_data_path, "r")
      orig_data = orig_data_file["data"]
      
      demo_data = orig_data[f"demo_{episode_idx}"]
      orig_actions = demo_data["actions"][()]
      orig_states = demo_data["states"][()]
            
      self.envs[task_idx].reset()
      obs = self.envs[task_idx].set_init_state(orig_states[0])
      
      self.initial_obs[task_idx] = obs
      
      
    def reset_all_rollouts(self, episode_idx):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for task_idx in range(self.rollout_batch_size):
            self.reset_rollout(task_idx, episode_idx)



@draccus.wrap()
def main(cfg: GenerateConfig):
  openvla_rollout_worker = RolloutWorkerOpenVLA(cfg)  
  
  
if __name__ == "__main__":
  main()  