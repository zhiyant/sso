from typing import Tuple, Dict, List
from argparse import ArgumentParser
from tqdm import tqdm
import os
import json
import numpy as np

from sso.env import Env
from sso.env.scienceworld import ScienceWorld
from sso.env.nethack.base import NetHackTask
from sso.agent import Agent
from sso.agent.skills import SkillsAgent
from sso.agent.fewshot import FewshotAgent
from sso.agent.reflexion import ReflexionAgent
from sso.memory.skillset import SkillSetMemory
from sso.memory.examples import ExamplesMemory
from sso.memory.subtrajset import MemoryBasedonRewards

from sso.trajectory import Trajectory
from sso.llm import set_default_model



if __name__ == '__main__':

    parser = ArgumentParser()

    # Experiment params
    parser.add_argument("--output", type=str, default="results", help="output directory")
    parser.add_argument("--train_iters", type=int, default=3, help="number of iterations to run")
    parser.add_argument("--test_iters", type=int, default=1, help="number of test iterations to run")

    # Agent params
    parser.add_argument("--agent", type=str, default="fewshot", help="agent type")
    parser.add_argument("--load", type=str, default=None, help="directory to load agent from")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model name")
    parser.add_argument("--train_temp", type=float, default=0.7, help="Generation temperature for the llm during training")
    parser.add_argument("--test_temp", type=float, default=0, help="Generation temperature for the llm during testing")
    parser.add_argument("--max_history", type=int, default=10, help="number of past steps to keep in history")
    parser.add_argument("--full_states", action="store_true", help="do not trim states to only keep new information")
    parser.add_argument("--similarity_metric", type=str, default="text-embedding-ada-002", help="similarity metric to use, iou or model name")

    # Memory params
    parser.add_argument("--memory", type=str, default="skills", help="memory type")
    parser.add_argument("--reward_weight", type=float, default=0.1, help="weight for trajectory reward in skill score")
    parser.add_argument("--state_weight", type=float, default=1, help="weight for state similarity in skill score")
    parser.add_argument("--action_weight", type=float, default=1, help="weight for action similarity in skill score")
    parser.add_argument("--coverage_weight", type=float, default=0.01, help="weight for task coverage in skill score")

    args = parser.parse_args()

    # Set LLMs
    set_default_model(model=args.model, temp=args.train_temp,
                      embedding=None if args.similarity_metric == "iou" else args.similarity_metric)

    
    memory = MemoryBasedonRewards()

    memory.load_and_reboot("test_transfer", is_single = False)

    
    # test with 2-nd trajectory.json
    load_path = "test_transfer/2"
    with open(os.path.join(load_path, "trajectory.json"), "r") as f:
        data = json.load(f)
    trajectory = Trajectory.from_dict(data)
    
    # load every pieces as the current trajectory
    for eidx in range(1, len(trajectory)):
        sub_traj = trajectory.slice(0, eidx)

        # log the last 
        memory.get_memories(trajectory=sub_traj, n = 3)
    

def log_results(
    agent: Agent,
    trajectory: Trajectory,
    save_path: str,
    iteration: int,
    task_id: str,
    success: bool,
    score: float
):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "logs.json"), "w") as f:
        json.dump(agent.get_log(), f, indent=4)
    with open(os.path.join(save_path, "trajectory.json"), "w") as f:
        json.dump(trajectory.to_dict(), f, indent=4)
    agent.save(save_path)
    print("Iter: {}, task: {}, success: {}, score: {}".format(iteration, task_id, success, score))
