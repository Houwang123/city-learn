import os
from matplotlib import pyplot as plt

os.chdir('../..')

with open('reward_function.txt', 'r') as f:
    lines = f.readlines()

while 'new episode\n' in lines:
    episode = lines[:lines.index('new episode\n')]
    lines = lines[lines.index('new episode\n')+1:]
    episode = [float(line) for line in episode]

    # print average of episode
    print(sum(episode) / len(episode))