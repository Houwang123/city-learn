import citylearn.energy_model
from matplotlib import pyplot as plt
import numpy as np


# for third_charge in np.linspace(-6.4, 6.4, 10):
#     xs = []
#     ys = []
#     print('c3', third_charge)
#     for first_charge in np.linspace(0, 6.4, 500):
#         for second_charge in np.linspace(0, 6.4, 500):
#             battery = citylearn.energy_model.Battery(capacity=6.4, nominal_power=5.0, capacity_loss_coefficient=1e-5,
#                                                      loss_coefficient=0, seconds_per_time_step=3600)
#             battery.charge(first_charge)
#             battery.next_time_step()
#
#             battery.charge(second_charge)
#             battery.next_time_step()
#
#             xs.append(battery.soc[-1])
#
#             battery.charge(third_charge)
#             battery.next_time_step()
#             ys.append(battery.soc[-1])
#     plt.close()
#     plt.title(f'soc after(y) charging {third_charge:.2f} vs previous soc(x)')
#     plt.plot(xs, ys, 'x')
#     # plt.show()
#     # save photo
#     plt.savefig(f'{third_charge:.2f}.png')




# battery = citylearn.energy_model.Battery(capacity=6.4, nominal_power=5.0, capacity_loss_coefficient=1e-5,
#                                          loss_coefficient=0, seconds_per_time_step=3600)
# for i in range(3000):
#     battery.charge(6.4)
#     battery.next_time_step()
#
#     battery.charge(-6.4)
#     battery.next_time_step()
# print(battery.capacity)

xs = np.linspace(-6.4, 6.4, 10000)
ys = []
for c in xs:
    print(c)
    battery = citylearn.energy_model.Battery(capacity=6.4, nominal_power=5.0, capacity_loss_coefficient=1e-5,
                                         loss_coefficient=0, seconds_per_time_step=3600)
    battery.charge(6.4)
    battery.next_time_step()

    battery.charge(c)
    battery.next_time_step()
    ys.append(battery.soc[-1])
    print(battery.soc[-1])

# plot with crosses
plt.plot(xs, ys, 'x')
plt.show()
    # print('soc:', battery.soc, 'capacity:', battery.capacity_history, battery.capacity, 'energy balance:', battery.energy_balance)
# battery.charge(6.4)
#
#
#
# print('soc:', battery.soc, 'capacity:', battery.capacity_history, battery.capacity, 'energy balance:', battery.energy_balance)
#
# battery.charge(-6.4)
# print('soc:', battery.soc, 'capacity:', battery.capacity_history, battery.capacity, 'energy balance:', battery.energy_balance)

# battery.charge(6.4)
#
#
# battery.next_time_step()
# print('soc:', battery.soc, 'capacity:', battery.capacity_history, battery.capacity, 'energy balance:', battery.energy_balance)
#
# battery.charge(-4.4)
# print('soc:', battery.soc, 'capacity:', battery.capacity_history, battery.capacity, 'energy balance:', battery.energy_balance)
