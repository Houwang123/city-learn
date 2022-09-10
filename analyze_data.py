import pickle

with open('observation.pkl', 'rb') as f:
    memory = pickle.load(f)

agent_count = len(memory)
step_count = len(memory[0])
print(f'agent_count: {agent_count}, step_count: {step_count}')

calc_carbon_baseline = 4625.736879225167  # not counting last day
calc_price_baseline = 8275.6322188979

true_carbon_baseline = 4627.487397661689
true_price_baseline = 8277.733108897906

def find_individual(agent_id):
    # find total carbon emission and total price
    carbon_baseline = 0.0
    price_baseline = 0.0

    for step in range(step_count):
        electricity_this_step = 0
        unit_carbon_intensity = memory[0][step][19]
        unit_price = memory[0][step][24]

        for agent_id in [agent_id]: #range(agent_count):
            non_shiftable_load = memory[agent_id][step][20]
            solar_generation = memory[agent_id][step][21]
            electricity_this_building = non_shiftable_load - solar_generation
            electricity_this_step += electricity_this_building
            carbon_baseline += max(0, unit_carbon_intensity * electricity_this_building)
        price_baseline += max(0, unit_price * electricity_this_step)
    print(f'carbon baseline: {carbon_baseline}, price baseline: {price_baseline}')
    print(f'prop - carbon: {carbon_baseline / calc_carbon_baseline}, '
          f'price: {price_baseline / calc_price_baseline}')
    return carbon_baseline / calc_carbon_baseline, price_baseline / calc_price_baseline

carbon_total = 0
price_total = 0

for agent_id in range(5):
    carbon, price = find_individual(agent_id)
    carbon_total += carbon
    price_total += price

print(f'carbon total: {carbon_total}, price total: {price_total}')