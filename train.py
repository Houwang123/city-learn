import json, math, sys, os, shutil, optparse, copy
from citylearn.citylearn import CityLearnEnv
import pickle as pkl
from tqdm import tqdm
import ipywidgets as widgets


import experiments.utils.setup as setup
from experiments.utils.setup import env_reset
from visualiser.frame_cache import append_one_frame
from visualiser.train_progress_bar import update_progress_by_one

# p = optparse.OptionParser()
# p.add_option('--force', '-f', type=None, help='Force overwrite of experiment',action="store_true")

def train(experiment_json_file_path, force_overwrite=False):
    raw = experiment_json_file_path
    experiment =  json.load(open(raw,'r'))
    # options, arguments = p.parse_args()
    # raw = sys.argv[1]
    # experiment = json.load(open(raw,'r'))
    experiment_raw = copy.deepcopy(experiment)
    setup.deserialize_document(experiment)
    print("========================================")
    # Check if experiment already run
    experiment_log_path = os.path.join('experiments',str(experiment['id']))
    if os.path.exists(experiment_log_path):
        print("Warning: Duplicate experiment ID.")
        if not force_overwrite:
            quit()
        shutil.rmtree(experiment_log_path)
        print("Continuing overwrite")
    
    os.makedirs(experiment_log_path)
    json.dump(experiment_raw,open(os.path.join(experiment_log_path,'agent.json'),'w'))

    # Environment setup

    print("Loading agent {}".format(experiment['agent']['type']))
    agent = setup.setup_agent(type=experiment['agent']['type'],attributes=experiment['agent']['attributes'])
    print("Setting reward {}".format(experiment['reward']))
    setup.setup_reward(experiment['reward'])
    print("Setting environment {}".format(experiment['schema']))
    env = CityLearnEnv(experiment['schema'])

    episodes_number = experiment['episodes']
    steps_per_frame_save = experiment['steps_per_frame_save']
    obs_dict = env_reset(env)

    # Run agent
    best_metrics = math.inf
    best_episode = 0

    for episode in range(episodes_number):
        actions = agent.register_reset(obs_dict, training=True)
        epoch = 0
        while True:
            observations, _, done, _ = env.step(actions)
            
            update_progress_by_one()

            if done:
                # Metric calculation
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
                
                print(f"Episode complete: {episode} | Score: {sum(metrics_t)}", )
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
    
                # Next
                obs_dict = env_reset(env)
                break
            
            actions = agent.compute_action(observations)

            epoch += 1
            if epoch % steps_per_frame_save == 0:
                append_one_frame(env.render())
    print("Best episode {}".format(best_episode))