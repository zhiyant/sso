from typing import List, Union, Tuple
import os
import json

import numpy as np
from tqdm import tqdm
from copy import deepcopy

from sso.trajectory import Trajectory
from sso.memory import Memory

from sso.llm import query_llm
from sso.utils import get_state_similarity


class MemoryBasedonRewards(Memory):

    def __init__(self, 
                 max_traj_len: int = 4,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.max_traj_len = max_traj_len
        self.gamma = 0.9

        self.best_reward = 0
    
    def save(self, save_path: str) -> None:

        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, "trajectories.json"), "w") as f:
            json.dump(dict(
                trajectories=[s.to_dict() for s in self.trajectories],
            ), f, indent=4)

    def load_and_reboot(self, load_path:str, is_single:bool=True):

        if is_single:
            with open(os.path.join(load_path, "trajectory.json"), "r") as f:
                data = json.load(f)
            trajectory = Trajectory.from_dict(data)
            self.build(trajectory)
        
        else:
            for subfolder in os.listdir(load_path):
                subfolder_path = os.path.join(load_path, subfolder)
                if os.path.exists(os.path.join(subfolder_path, "trajectory.json")):
                    with open(os.path.join(subfolder_path, "trajectory.json"), "r") as f:
                        data = json.load(f)

                    trajectory = Trajectory.from_dict(data)
                    self.build(trajectory)
    
        

    def get_discounted_rewards(self, trajectory: Trajectory) -> List[float]:
        discounted_reward = []
        _add = 0
        for state in reversed(trajectory):
            if state.reward is not None:
                _add = state.reward + self.gamma * _add
                discounted_reward.insert(0, _add)
        return discounted_reward

    def insert(self, trajectory: Trajectory) -> None:
        # TODO: reduce overlap in sub trajectories

        # print the current state
        print("#### Current Trajectory ####")
        for state in trajectory:
            print(state.last_action, state.reward)

        # remove trailing zeros
        reward = [x.reward for x in trajectory]
        while reward and reward[-1] == 0:
            reward.pop()
        
        # caculate the discounted rewards of trajectory
        discounted_rewards = self.get_discounted_rewards(trajectory)

        discounted_rewards = discounted_rewards[:len(reward)]


        # fixed length of sub_traj
        length = self.max_traj_len

        while True:
            
            best_return = 0
            sidx = None

            for idx in range(1, len(discounted_rewards) - length + 1, 1):
                
                cumm_return = best_return - discounted_rewards[idx - 1] + discounted_rewards[idx + length - 1]
                if cumm_return > best_return:
                    best_return = cumm_return
                    sidx = idx
 
            
            
            if sidx is None or sidx + length + 1 > len(discounted_rewards):
                break
            else:
                sub_traj = trajectory.slice(sidx, sidx + length) # best_sub_traj
                self.trajectories.append(sub_traj)
                for idx in range(sidx, sidx + length + 1):
                    discounted_rewards[idx] = -np.inf


        print("#### Selected Subtrajectories in insert() ####")
        for i, subtraj in enumerate(self.trajectories):
            print(f"# subtrajectory {i} #")
            for x in subtraj:
                print(x.last_action, x.reward)

        return


    def score_traj(self):
        return

    def get_memories(self, trajectory: Trajectory = None, n: int = None) -> List[Trajectory]:

        print("#### Current Trajectory ####")
        for state in trajectory:
            print(state.last_action, state.reward)


        if n is None or n > len(self.trajectories):
            n = len(self.trajectories)
        

        alpha1 = 0.1
        alpha2 = 1.5
        alpha3 = 0.25

        combined_list = []

        for i in range(len(self.trajectories)):

            ret = sum(self.get_discounted_rewards(self.trajectories[i])) * alpha1
            # reward //= self.best_reward
            
            idx = (i // len(self.trajectories)) *alpha2

            state_similarities = get_state_similarity(self.trajectories[i][0], trajectory[-1]) * alpha3 if trajectory is not None else 0
            

            combined_list.append([ret, idx, state_similarities, self.trajectories[i]])


        all_mean = np.mean(combined_list,axis=0)
        all_std = np.std(combined_list, axis=0)




        sorted_combined_list = sorted(combined_list, 
                                      key=lambda x:(x[0],x[1],x[2]),
                                      reverse=True) # desc
        
        sorted_trajectories = []

        for i in range(n):
            sorted_trajectories.append(sorted_combined_list[i][3])

        print("#### Result for get_memories() ####")
        for i, sub_traj in enumerate(sorted_trajectories):
            print(f"# subtrajectory {i} #")
            for x in sub_traj:
                print(x.last_action, x.reward)


        return sorted_trajectories

    

