import pickle
import random

with open('../observation.pkl', 'rb') as f:
    memory = pickle.load(f)

data = [point for agent_data in memory for point in agent_data]
data = [(point[3], point[7], point[11], point[15], point[21]) for point in data]
random.shuffle(data)

print(len(data))

train_data = data[:int(len(data) * 0.7)]
valid_data = data[int(len(data) * 0.7):int(len(data) * 0.85)]
test_data = data[int(len(data) * 0.85):]

with open('train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('valid_data.pkl', 'wb') as f:
    pickle.dump(valid_data, f)

with open('test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

