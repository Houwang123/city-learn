import pickle

with open('observation.pkl', 'rb') as f:
    memory = pickle.load(f)

agent_count = len(memory)
step_count = len(memory[0])
print(f'agent_count: {agent_count}, step_count: {step_count}')

# find total carbon emission and total price
carbon_baseline = 0.0
price_baseline = 0.0
for agent_id in range(agent_count):
    for step in range(step_count):
        unit_carbon_intensity = memory[agent_id][step][19]
        non_shiftable_load = memory[agent_id][step][20]
        unit_price = memory[agent_id][step][24]
        carbon_baseline += unit_carbon_intensity * non_shiftable_load
        price_baseline += unit_price * non_shiftable_load
print(f'carbon baseline: {carbon_baseline}, price baseline: {price_baseline}')

for agent_id in range(agent_count):
    agent_carbon = 0
    agent_price = 0
    for step in range(step_count):
        unit_carbon_intensity = memory[agent_id][step][19]
        non_shiftable_load = memory[agent_id][step][20]
        solar_generation = memory[agent_id][step][21]
        unit_price = memory[agent_id][step][24]

        agent_carbon += unit_carbon_intensity * non_shiftable_load
        agent_price += unit_price * non_shiftable_load
    print(f'agent {agent_id}: carbon: {agent_carbon}, price: {agent_price}')
    # print out proportions
    print(f'agent {agent_id}: carbon: {agent_carbon / carbon_baseline}, price: {agent_price / price_baseline}')
