import os
from matplotlib import pyplot as plt

os.chdir('../..')

with open('reward_function.txt', 'r') as f:
    lines = f.readlines()
    lines = lines[:lines.index('new episode\n')]
    lines = [float(line) for line in lines]

len_rewards = len(lines)

# 10 fold the data and see average
data = []
for i in range(10):
    data.append(lines[i * len_rewards // 10: (i + 1) * len_rewards // 10])

avgs = []
for i in range(10):
    avgs.append(sum(data[i]) / len(data[i]))

for avg in avgs:
    print(avg)
