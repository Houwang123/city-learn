import torch as t
import torch.nn as nn
from agents.actor_critic_agent.network import Policy

policy = Policy()

obs = t.tensor([ 0.6667,  0.1613,  0.5417,  0.7233,  0.6467,  0.6667,  0.7600,  0.7300,
         0.7900,  0.6800,  0.6100,  0.9600,  0.0220,  0.0000,  0.6770,  0.8190,
         0.1290,  0.0000,  0.2590,  0.2085,  2.8080,  2.8176,  0.0000, -2.2404,
         0.2200,  0.5400,  0.2200,  0.2200,  0.0000,  0.0000,  0.0000,  1.0000,
         0.0000], requires_grad=True)

for ep in range(1, 100):
    obs.grad = None
    print(f'ep: {ep}')
    policy.load_state_dict(t.load(f'./net_param/policy_{ep}.net')['weight'])
    action = policy(obs)
    print(action)
    action[13].backward()
    print(obs.grad)