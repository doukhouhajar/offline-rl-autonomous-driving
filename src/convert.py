
import pickle
import numpy as np
from d3rlpy.dataset import MDPDataset

PATH = "/home/normal/Desktop/rl_project/aae-train-donkeycar/buffer.pkl"

with open(PATH, "rb") as f:
    data = pickle.load(f)

obs, actions, rewards, next_obs, dones = zip(*data)

dataset = MDPDataset(
    observations=np.array(obs),
    actions=np.array(actions),
    rewards=np.array(rewards),
    terminals=np.array(dones),
)

dataset.dump("dataset.pkl")
print("dataset.pkl created")
