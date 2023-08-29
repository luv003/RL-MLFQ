import warnings
warnings.filterwarnings("ignore",catefgory=UserWarning)
warnings.filterwarnings("ignore",category=UserWarning,module="stable_baselines")
import wandb
from stable_baselines import PPO2, ACKTR
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback
from env imoprt mEnv

nQueues = 3
boost=0
agent = "PPO2"
#agent="ACKTR"

wandb.init(project='process-scheduling1', config={"nQueues": nQueues, "boost": boost, "agent": agent})
wandb.run.name = agent_name
wandb.config.update({"nQueues": nQueues, "boost": boost, "agent": agent})



env= mEnv(boost,nQueues,False)
env=DummyVecEnv([lambda: env])

hyperparams={'gamma':0.99,'n_steps':128,'learning_rate':0.00025}

model  = PPO2('MlpPolicy',env,verbose=1,**hyperparams)
callback =EvalCallback(env,best_model_save_path='./logs/',
                       log_path='./logs/',
                       eval_freq=5000,
                       deterministic=True,
                       render=False)

obs = env.reset()
r=0
for _ in range(20000):
  action,_state=model.predict(obs,deterministic=True)
  obs,reward,done,info=env.step(action)
  r+=reward
  if _ % 100==0:
    wandb.log({"reward":r})
    model.save(f"{agent}_{_}")
    
  
  



wandb.finish()

   

