import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



import json, math, sys, os, shutil, optparse, copy
import pickle as pkl
from tqdm import tqdm
from visualiser.custom_citylearn_env import CustomCityLearnEnv
import experiments.utils.setup as setup
from experiments.utils.setup import env_reset
from visualiser.frame_cache import FrameCache
from visualiser.train_progress_bar import TrainProgressBar

p = optparse.OptionParser()
p.add_option('--force', '-f', type=None, help='Force overwrite of experiment',action="store_true")
pth = os.getcwd()

def train(config, experiment_json_file_path='example_experiment.json', force_overwrite=True, progress_bar = None, frame_cache = None):
    raw = experiment_json_file_path
    os.chdir(pth)
    experiment =  json.load(open(raw,'r'))

    
    gid = 1
    while os.path.exists(os.path.join('experiments', str(gid))):
        gid += 1


    experiment['id'] = gid
    # 
    # experiment = json.load(open(raw,'r'))
    experiment_raw = copy.deepcopy(experiment)
    setup.deserialize_document(experiment)
    # print("========================================")
    # Check if experiment already run

    
    experiment_log_path = os.path.join('experiments',str(experiment['id']))
    if os.path.exists(experiment_log_path):
        # print("Warning: Duplicate experiment ID.")
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
        if not force_overwrite:
            quit()
        shutil.rmtree(experiment_log_path)
        # print("Continuing overwrite")
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))
    
    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))


    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))


    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))


    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    if os.path.exists(experiment_log_path):
        gid = 1
        while os.path.exists(os.path.join('experiments', str(gid))):
            gid += 1
        experiment['id'] = gid
        experiment_raw['id'] = gid
        experiment_log_path = os.path.join('experiments',str(experiment['id']))

    os.makedirs(experiment_log_path)
    
    json.dump(experiment_raw,open(os.path.join(experiment_log_path,'agent.json'),'w'))

    # Environment setup

    # print("Loading agent {}".format(experiment['agent']['type']))

    experiment['agent']['attributes']['gamma'] = config['gamma']
    experiment['agent']['attributes']['lr'] = config['lr']
    experiment['agent']['attributes']['tau'] = config['tau']
    experiment['agent']['attributes']['batch_size'] = config['batch_size']
    experiment['agent']['attributes']['memory_size'] = config['memory_size']
    experiment['agent']['attributes']['a_kwargs']['comm_steps'] = config['comm_steps']
    experiment['agent']['attributes']['a_kwargs']['comm_size'] = config['comm_size']
    experiment['agent']['attributes']['a_kwargs']['hidden_size'] = config['hidden_size_a']
    experiment['agent']['attributes']['c_kwargs']['hidden_size'] = config['hidden_size_c']
    
    agent = setup.setup_agent(type=experiment['agent']['type'],attributes=experiment['agent']['attributes'])
    # print("Setting reward {}".format(experiment['reward']))
    setup.setup_reward(experiment['reward'])
    # print("Setting environment {}".format(experiment['schema']))
    env = CustomCityLearnEnv(experiment['schema'])

    episodes_number = experiment['episodes']
    steps_per_frame_save = experiment['steps_per_frame_save']
    obs_dict = env_reset(env)

    # Run agent
    best_metrics = math.inf
    best_episode = 0

    # print('Start')

    if progress_bar == 'tqdm':
        pbar = tqdm(total=24*365)

    for episode in range(episodes_number):
        actions = agent.register_reset(obs_dict, training=True)
        epoch = 0

        while True:
            observations, _, done, _ = env.step(actions)
            
            if progress_bar is not None:
                if progress_bar != 'tqdm':
                    progress_bar.update_progress_by_one()
                else:
                    pbar.update(1)


            if done:
                # Metric calculation
                if progress_bar == 'tqdm':
                    pbar.reset()

                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
                
                # print(f"Episode complete: {episode} | Score: {sum(metrics_t)}", )
                if sum(metrics_t) < best_metrics:
                    best_metrics = sum(metrics_t)
                    best_episode = episode
                    
                # Save state
                checkpoint = os.path.join(experiment_log_path,str(episode))
                os.makedirs(checkpoint)
                agent.save(checkpoint)
                with open(os.path.join(checkpoint,'env.pkl'),'wb') as f:
                    pkl.dump(env,f)
                with open(os.path.join(checkpoint,'metrics.json'),'w') as f:
                    json.dump(metrics_t,f)
                tune.report(sum_of_metrics=sum(metrics_t))
    
                # Next
                obs_dict = env_reset(env)
                break
            
            actions = agent.compute_action(observations)

            epoch += 1
            if epoch % steps_per_frame_save == 0 and frame_cache is not None:
                frame_cache.append_one_frame(env.render())

    if progress_bar == 'tqdm':
        pbar.close()
    # print("Best episode {}".format(best_episode))

#############################################################

def main():
    config = {'gamma':0.99,
    'lr':3e-4,
    'tau':0.001,
    'batch_size':256,
    'memory_size':65536,
    'comm_steps':5,
    'comm_size':6,
    'hidden_size_a':64,
    'hidden_size_c':128
    }
    config = {'gamma':0.99,
    'lr':tune.grid_search([3e-4, 1e-5]),
    'tau':0.001,
    'batch_size':256,
    'memory_size':65536,
    'comm_steps':5,
    'comm_size':6,
    'hidden_size_a':64,
    'hidden_size_c':128
    }
    config = {'gamma':tune.loguniform(0.6, 0.999),
    'lr':tune.loguniform(1e-6, 1e-2),
    'tau':tune.loguniform(1e-4, 1e-2),
    'batch_size':tune.grid_search([64, 128, 256, 512]),
    'memory_size':65536,
    'comm_steps':tune.randint(2,8),
    'comm_size':tune.randint(2,8),
    'hidden_size_a':tune.grid_search([64, 128, 256, 512]),
    'hidden_size_c':tune.grid_search([64, 128, 256, 512])
    }
    analysis = tune.run(
        train,
        config=config,
        )
    print("Best config: ", analysis.get_best_config(metric="sum_of_metrics", mode="min"))
    df = analysis.results_df
    return df

if __name__=='__main__':
    # CREATE_SUSPENDED = 0x00000004  # from Windows headers
    # CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    df = main()