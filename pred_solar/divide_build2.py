import pickle
import random
import os

os.mkdir("./build2")

with open('../observation.pkl', 'rb') as f:
    memory = pickle.load(f)

data = memory[1]
data = [(point[0], point[3], point[7], point[11], point[15], point[21]) for point in data]
random.shuffle(data)

print(len(data))

train_data = data[:int(len(data) * 0.35)] + data[int(len(data) * 0.65):]
valid_data = data[int(len(data) * 0.35):int(len(data) * 0.5)]
test_data = data[int(len(data) * 0.5):int(len(data) * 0.65)]

with open('build2/train_data2.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('build2/valid_data2.pkl', 'wb') as f:
    pickle.dump(valid_data, f)

with open('build2/test_data2.pkl', 'wb') as f:
    pickle.dump(test_data, f)
