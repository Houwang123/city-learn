import pickle
import numpy as np
import time
from find_charge import find_charge

INTERVAL = 101

with open('observation.pkl', 'rb') as f:
    memory = pickle.load(f)

agent_count = len(memory)
step_count = len(memory[0])
print(f'agent_count: {agent_count}, step_count: {step_count}')

carbon_baseline = 4627.487397661689
price_baseline = 8277.733108897906
print(f'carbon baseline: {carbon_baseline}, price baseline: {price_baseline}')

states = np.linspace(0, 6.4, INTERVAL)

def do_dp(agent_id):
    # f is objective : score, previous_charge_command, previous_objective, capacity, true soc,
    # price_score, carbon_score
    start_time = time.time()
    f = []
    unit_carbon_intensity = memory[agent_id][0][19]
    non_shiftable_load = memory[agent_id][0][20]
    solar_generation = memory[agent_id][0][21]
    unit_price = memory[agent_id][0][24]

    net_load = non_shiftable_load - solar_generation
    if net_load > 0:
        price_score = net_load * unit_price / price_baseline
        carbon_score = net_load * unit_carbon_intensity / carbon_baseline
    else:
        price_score = 0
        carbon_score = 0
    score = price_score + carbon_score

    f.append({0: (score, None, None, 6.4, 0, price_score, carbon_score)})  # initial score

    for step in range(1, step_count):
        f.append(dict())

        unit_carbon_intensity = memory[agent_id][step][19]
        non_shiftable_load = memory[agent_id][step][20]
        solar_generation = memory[agent_id][step][21]
        unit_price = memory[agent_id][step][24]

        for last_objective in f[step-1]:
            last_score, _, _, last_capacity, last_soc, last_price_score, last_carbon_score \
                = f[step-1][last_objective]
            for new_objective in states:
                if new_objective > last_capacity:
                    continue

                charge_command, new_capacity, new_soc = find_charge(last_soc, new_objective, last_capacity)
                if charge_command is not None:
                    net_load = non_shiftable_load + charge_command - solar_generation
                    if net_load > 0:
                        price_score = last_price_score + net_load * unit_price / price_baseline
                        carbon_score = last_carbon_score + net_load * unit_carbon_intensity / carbon_baseline
                    else:
                        price_score = last_price_score
                        carbon_score = last_carbon_score
                    score = price_score + carbon_score

                    if new_objective not in f[step]:
                        f[step][new_objective] = \
                            (score, charge_command, last_objective, new_capacity, new_soc, price_score, carbon_score)
                    else:
                        if score < f[step][new_objective][0]:
                            f[step][new_objective] = \
                                (score, charge_command, last_objective, new_capacity, new_soc, price_score, carbon_score)
        # if step % 50 == 0:
        #     print(f'step: {step}')

    with open(f'./actions/results_{INTERVAL}_approx_agent_{agent_id}.pkl', 'wb') as file:
        pickle.dump(f, file)

    normalized_command_history = []
    soc_history = []
    min_score_key = min(f[-1].keys(), key=lambda x: f[-1][x][0])
    final_score, charge_command, last_objective, previous_capacity, previous_soc, price_score, carbon_score\
        = f[-1][min_score_key]
    print(
        f'final score: {score}, carbon score: {carbon_score}, price score: {price_score}, final soc: {previous_soc}')
    print(f'time: {time.time() - start_time}')
    for step in range(step_count-1, 0, -1):
        normalized_command_history.append(f[step][min_score_key][1] / f[step][min_score_key][3])
        soc_history.append(f[step][min_score_key][4])
        min_score_key = f[step][min_score_key][2]

    normalized_commands = list(reversed(normalized_command_history))
    with open(f'./actions/{agent_id}_actions.pkl', 'wb') as file:
        pickle.dump(normalized_commands, file)

    # print('command history:', normalized_commands)

    # print('soc history:', list(reversed(soc_history)))
    return price_score, carbon_score

carbon_sum = 0
price_sum = 0
for a_id in range(5):
    p, c = do_dp(a_id)
    price_sum += p
    carbon_sum += c
print(f'price sum: {price_sum}, carbon sum: {carbon_sum}')