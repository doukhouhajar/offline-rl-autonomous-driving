import gym
import gym_donkeycar
import time
from d3rlpy.algos import CQL

from aae_train_donkeycar.ae.wrapper import AutoencoderWrapper
ENV = "donkey-mountain-track-v0"
MODEL_PATH = "cql_donkey_policy"

print("Loading offline CQL model...")

# ---- Create env first ----
env = gym.make(ENV)
env=AutoencoderWrapper(env)
# ---- Rebuild model EXACTLY as training ----
model = CQL(
    batch_size=512,
    conservative_weight=10.0,
    n_action_samples=20,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    gamma=0.99,
    use_gpu=False
)

# ---- IMPORTANT ----
model.build_with_env(env)

# ---- Load trained weights ----
model.load_model(MODEL_PATH)

print("Model loaded successfully!")

obs = env.reset()
laps = 0

while True:
    action = model.predict(obs.reshape(1, -1))[0]
    obs, reward, done, info = env.step(action)

    if "lap" in info:
        laps += 1
        print(f"🏁 OFFLINE LAP {laps}")

    if done:
        obs = env.reset()

    time.sleep(0.05)
