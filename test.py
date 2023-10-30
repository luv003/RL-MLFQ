import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines")

import wandb
import numpy as np
from stable_baselines import PPO2,ACKTR
from agents import PPOAgent,ACKTRAgent
from env import MLFQEnv,Gui

np.random.seed(2)

nQueues = 3
boost = 0
agent = "PPO2"
file = "ppomodel"

env = MLFQEnv(boost, nQueues, False)
agent_dict = {'PPO2': PPO2,'ACKTR':ACKTR}
print("Agent:", agent)

model = agent_dict[agent].load(file)

wandb.init(project='process-scheduling01', config={'nQueues': nQueues, 'boost': boost, 'agent': agent, 'file': file})
wandb.run.name = agent
wandb.run.save()
wandb.config.update({'nQueues': nQueues, 'boost': boost, 'agent': agent, 'file': file})
np.random.seed(123)
for i in range(100):
    obs = env.reset()
    init_quantum = env.quantum_list.copy()
    r = 0

    for j in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        r += reward

        if done:
            break

    wandb.log({"Iteration": i, "Reward": r})
    print(f"Iteration {i}: reward =", r)

    env.print_stats()
    env.log_stats()
    #env.render()

wandb.run.finish()

