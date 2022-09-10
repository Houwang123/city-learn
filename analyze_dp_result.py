import pickle
import numpy as np
import time

INTERVAL = 3
AGENT_ID = 0

with open('observation.pkl', 'rb') as f:
    memory = pickle.load(f)

agent_count = len(memory)
step_count = len(memory[AGENT_ID])
print(f'agent_count: {agent_count}, step_count: {step_count}')

carbon_baseline = 4627.487397661689
price_baseline = 8277.733108897906
print(f'carbon baseline: {carbon_baseline}, price baseline: {price_baseline}')

states = np.linspace(0, 6.4, INTERVAL)

price_contribution = 0
carbon_contribution = 0

# score, previous_charge_command, previous_objective, previous_capacity, previous soc
# previous_price_score, previous_carbon_score
with open(f'./actions/results_{INTERVAL}_approx_agent_{AGENT_ID}.pkl', 'rb') as file:
    f = pickle.load(file)

f_history = []
min_score_key = min(f[-1].keys(), key=lambda x: f[-1][x][0])
score, charge_command, last_objective, previous_capacity, previous_soc, price_score, carbon_score\
    = f[-1][min_score_key]
print(
    f'final score: {score}, carbon score: {carbon_score}, price score: {price_score}, final soc: {previous_soc}')


for step in range(step_count, -1, -1):
    # normalized_command_history.append(f[step][min_score_key][1] / f[step][min_score_key][3])
    # soc_history.append(f[step][min_score_key][4])
    # objective_history.append(f[step][min_score_key][2])
    f_history.append(f[step][min_score_key])
    min_score_key = f[step][min_score_key][2]

f_history.reverse()

for step in range(len(f_history)):
    unit_carbon_intensity = memory[AGENT_ID][step][19]
    non_shiftable_load = memory[AGENT_ID][step][20]
    solar_generation = memory[AGENT_ID][step][21]
    unit_price = memory[AGENT_ID][step][24]

    last_score, last_charge_command, last_objective, last_capacity, last_soc, last_price_score, last_carbon_score \
        = f_history[step]
    this_score, this_charge_command, this_objective, this_capacity, this_soc, this_price_score, this_carbon_score \
        = f_history[step + 1]

    if step == 12:
        print()
    net_load = non_shiftable_load + last_charge_command - solar_generation
    if net_load > 0:
        price_score = last_price_score + net_load * unit_price / price_baseline
        carbon_score = last_carbon_score + net_load * unit_carbon_intensity / carbon_baseline
    else:
        price_score = last_price_score
        carbon_score = last_carbon_score
    score = price_score + carbon_score


    print(f'{step}:\t{last_charge_command}\t{non_shiftable_load}\t{solar_generation}\t{this_carbon_score*carbon_baseline}\t{carbon_score*carbon_baseline}')





