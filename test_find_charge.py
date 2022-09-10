import numpy as np
from find_charge import find_charge, find_charge_given_battery, get_current_efficiency
from citylearn.energy_model import Battery


def test_find_charge():
    for first_charge in np.linspace(0, 6.4, 50):
        for second_charge in np.linspace(0, 6.4, 50):
            for objective in np.linspace(0, 6.4, 50):
                battery = Battery(capacity=6.4, nominal_power=5.0, capacity_loss_coefficient=1e-5,
                                  loss_coefficient=0, seconds_per_time_step=3600)
                battery.charge(first_charge)
                battery.next_time_step()
                battery.charge(second_charge)
                battery.next_time_step()
                charge_command, result_battery = find_charge_given_battery(battery, objective)
                if result_battery is not None and abs(result_battery.soc[-1] - objective) > 1e-5:
                    print(
                        f'objective: {objective}, first_charge: {first_charge}, '
                        f'second_charge: {second_charge}, charge_command: {charge_command}, '
                        f'result_battery.soc: {result_battery.soc}'
                        f'battery capacity: {battery.capacity}')
                    raise RuntimeError('find_charge_given_battery failed')
                charge2, new_capacity, new_soc = find_charge(battery.soc[-1], objective, battery.capacity)
                if charge2 != charge_command:
                    print(
                        f'objective: {objective}, first_charge: {first_charge}, '
                        f'second_charge: {second_charge}, '
                        f'charge_command: {charge_command}, charge2: {charge2} '
                        f'soc: {battery.soc}, capacity: {battery.capacity}')
                    raise RuntimeError('find_charge failed')
                if result_battery is not None and new_capacity != result_battery.capacity:
                    print(
                        f'capacity problem. objective: {objective}, first_charge: {first_charge}, '
                        f'second_charge: {second_charge}, '
                        f'charge_command: {charge_command}, new_capacity: {new_capacity}, '
                        f'result_battery.capacity: {result_battery.capacity}')
                    raise RuntimeError('find_charge failed')

                if result_battery is not None and new_soc != result_battery.soc[-1]:
                    print(
                        f'incorrect predicted soc. '
                        f'pred: {new_soc}, true: {result_battery.soc[-1]}, '
                        f'original soc: {battery.soc[-1]}'
                        f'cur eff: {get_current_efficiency(charge2)}'
                        f'objective: {objective}, first_charge: {first_charge}, '
                        f'second_charge: {second_charge}, '
                        f'charge_command: {charge_command}, ')
                    raise RuntimeError('soc not correct')

# battery = Battery(capacity=6.4, nominal_power=5.0, capacity_loss_coefficient=1e-5,
#                                   loss_coefficient=0, seconds_per_time_step=3600)
# objective = 1.0448979591836736
# first_charge = 1.959183673469388
# second_charge = 5.093877551020409
# capacity = 6.399965268477865
# test_result = -4.945114605992559

# battery.charge(first_charge)
# battery.next_time_step()
# battery.charge(second_charge)
# battery.next_time_step()
# charge_command, result_battery = find_charge_given_battery(battery, objective)
#
# battery.charge(test_result)
# battery.next_time_step()
#
# charge2 = find_charge(6.399990204081633, objective, capacity)
# print(charge_command, result_battery, battery.soc[-1])

test_find_charge()  # test passed, including capacity test
