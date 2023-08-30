import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines")
import wandb
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback
from env import mEnv

nQueues = 3
boost = 0

wandb.init(project='process-scheduling1', config={"nQueues": nQueues, "boost": boost})
wandb.run.name = "Baseline"
wandb.config.update({"nQueues": nQueues, "boost": boost})

env= mEnv(boost,nQueues,False)
env=DummyVecEnv([lambda: env])

for i in range(100):
        obs = env.reset()
        init_quantum = env.quantum_list
        total_reward = 0

        for j in range(10000):
            obs, reward, done, info = env.step(init_quantum)
            total_reward += reward
            if done:
                break 

        print(f"Iteration {i}: reward =", total_reward)
        wandb.log({"Iteration": i, "Reward": total_reward})

        env.print_stats()
        env.render()
        env.log_stats()
wandb.run.finish()
