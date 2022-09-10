import numpy as np
from citylearn.energy_model import Battery
from copy import deepcopy

roots = dict()
count_clash = 0


def find_roots(coefficients):
    global count_clash
    if coefficients in roots:
        count_clash += 1
        return roots[coefficients]
    else:
        p = np.polynomial.polynomial.Polynomial(coefficients)
        r = p.roots()
        roots[coefficients] = r
        return r


def get_max_output_power(now, capacity=6.4, nominal_power=5.0):
    # note: cannot assert now <= capacity; capacity shrinks after charging
    normalized_soc = now / capacity
    if 0 <= normalized_soc <= 0.8:
        return nominal_power
    elif 0.8 < normalized_soc < 1:
        grad = (0.2 - 1) / (1 - 0.8)
        coe = 1 + (normalized_soc - 0.8) * grad
        return nominal_power * coe
    else:  # normalized_soc >= 1
        return nominal_power


def get_current_efficiency(energy, nominal_power=5.0):
    energy_normalized = abs(energy) / nominal_power
    if energy_normalized <= 0.3:
        eff = 0.83
    elif energy_normalized <= 0.7:
        eff = 0.83 + (energy_normalized - 0.3) * (0.9 - 0.83) / (0.7 - 0.3)
    elif energy_normalized <= 0.8:
        eff = 0.9
    elif energy_normalized <= 1.0:
        eff = 0.9 + (energy_normalized - 0.8) * (0.85 - 0.9) / (1.0 - 0.8)
    else:
        raise ValueError(f'energy_normalized should be in [0, 1], but is {energy_normalized}')
    return eff ** 0.5


def get_new_capacity(old_capacity, charge_command, capacity_loss_coe=1e-5, original_capacity=6.4):
    """charge command must not be bounded by max_power_output"""
    capacity_degrade = capacity_loss_coe * original_capacity * abs(charge_command) / (2 * old_capacity)
    return old_capacity - capacity_degrade


def find_charge(now, objective, capacity):
    """given a battery, return the charge that will bring the battery to the objective,
    the new capacity and the new soc (just to account for numerical errors)"""
    if objective > capacity:
        return None, None, None

    # assume that real charging effect increases as demanded charge increases
    delta = objective - now
    nominal_power = 5.0

    max_power = get_max_output_power(now, capacity=capacity)
    max_power_eff = get_current_efficiency(max_power)
    if delta > max_power * max_power_eff or delta < -max_power / max_power_eff:
        return None, None, None

    if delta > 0:
        charge_effect_03 = 0.3 * nominal_power * (0.83 ** 0.5)
        charge_effect_07 = 0.7 * nominal_power * (0.9 ** 0.5)
        charge_effect_08 = 0.8 * nominal_power * (0.9 ** 0.5)
        charge_effect_1 = 1.0 * nominal_power * (0.85 ** 0.5)
        if delta <= charge_effect_03:
            charge_command = delta / (0.83 ** 0.5)
        elif delta <= charge_effect_07:
            # charge x, eff^2 = 0.83 + (x/5 - 0.3) * (0.9-0.83) / (0.7-0.3)
            # solve equation: x * eff = delta
            gradient = (0.9 - 0.83) / (0.7 - 0.3)
            a = gradient / nominal_power
            b = -gradient * 0.3 + 0.83

            charge_command = find_roots((-delta ** 2, 0, b, a))[-1]
        elif delta <= charge_effect_08:
            charge_command = delta / (0.9 ** 0.5)
        elif delta <= charge_effect_1:
            gradient = (0.85 - 0.9) / (1.0 - 0.8)
            a = gradient / nominal_power
            b = -gradient * 0.8 + 0.9

            charge_command = find_roots((-delta ** 2, 0, b, a))[1]
        else:
            raise ValueError(f'charge too large! soc: {now}, delta: {delta}')
        new_soc = now + charge_command * get_current_efficiency(charge_command)
    else:
        charge_effect_03 = 0.3 * nominal_power / (0.83 ** 0.5)
        charge_effect_07 = 0.7 * nominal_power / (0.9 ** 0.5)
        charge_effect_08 = 0.8 * nominal_power / (0.9 ** 0.5)
        charge_effect_1 = 1.0 * nominal_power / (0.85 ** 0.5)
        if delta >= -charge_effect_03:
            charge_command = delta * (0.83 ** 0.5)
        elif delta >= -charge_effect_07:
            # charge x, eff^2 = 0.83 + (x/5 - 0.3) * (0.9-0.83) / (0.7-0.3)
            # solve equation: x / eff = delta
            gradient = (0.9 - 0.83) / (0.7 - 0.3)
            a = gradient / nominal_power
            b = -gradient * 0.3 + 0.83

            charge_command = find_roots((-b * delta ** 2, a * delta ** 2, 1))[0]
        elif delta >= -charge_effect_08:
            charge_command = delta * (0.9 ** 0.5)
        elif delta >= -charge_effect_1:
            gradient = (0.85 - 0.9) / (1.0 - 0.8)
            a = gradient / nominal_power
            b = -gradient * 0.8 + 0.9

            charge_command = find_roots((-b * delta ** 2, a * delta ** 2, 1))[0]
        else:
            raise ValueError(f'discharge too large! soc: {now}, delta: {delta}')
        new_soc = now + charge_command / get_current_efficiency(charge_command)
    new_capacity = get_new_capacity(capacity, charge_command)
    new_soc = max(0, new_soc)
    new_soc = min(new_capacity, new_soc)

    return charge_command, new_capacity, new_soc


def find_charge_given_battery(battery: Battery, objective):
    """given a battery, return the charge that will bring the battery to the objective and the charged battery object"""
    if objective > battery.capacity:
        return None, None

    # assume that real charging effect increases as demanded charge increases
    now = battery.soc[-1]
    delta = objective - now
    nominal_power = 5.0

    max_power = battery.get_max_output_power()
    max_power_eff = battery.get_current_efficiency(max_power)
    if delta > max_power * max_power_eff or delta < -max_power / max_power_eff:
        return None, None

    if delta > 0:
        charge_effect_03 = 0.3 * nominal_power * (0.83 ** 0.5)
        charge_effect_07 = 0.7 * nominal_power * (0.9 ** 0.5)
        charge_effect_08 = 0.8 * nominal_power * (0.9 ** 0.5)
        charge_effect_1 = 1.0 * nominal_power * (0.85 ** 0.5)
        if delta <= charge_effect_03:
            charge_command = delta / (0.83 ** 0.5)
        elif delta <= charge_effect_07:
            # charge x, eff^2 = 0.83 + (x/5 - 0.3) * (0.9-0.83) / (0.7-0.3)
            # solve equation: x * eff = delta
            gradient = (0.9 - 0.83) / (0.7 - 0.3)
            a = gradient / nominal_power
            b = -gradient * 0.3 + 0.83

            p = np.polynomial.polynomial.Polynomial([-delta ** 2, 0, b, a])
            charge_command = p.roots()[-1]
        elif delta <= charge_effect_08:
            charge_command = delta / (0.9 ** 0.5)
        elif delta <= charge_effect_1:
            gradient = (0.85 - 0.9) / (1.0 - 0.8)
            a = gradient / nominal_power
            b = -gradient * 0.8 + 0.9

            p = np.polynomial.polynomial.Polynomial([-delta ** 2, 0, b, a])
            charge_command = p.roots()[1]
        else:
            raise ValueError(f'charge too large! soc: {now}, delta: {delta}')
    else:
        charge_effect_03 = 0.3 * nominal_power / (0.83 ** 0.5)
        charge_effect_07 = 0.7 * nominal_power / (0.9 ** 0.5)
        charge_effect_08 = 0.8 * nominal_power / (0.9 ** 0.5)
        charge_effect_1 = 1.0 * nominal_power / (0.85 ** 0.5)
        if delta >= -charge_effect_03:
            charge_command = delta * (0.83 ** 0.5)
        elif delta >= -charge_effect_07:
            # charge x, eff^2 = 0.83 + (x/5 - 0.3) * (0.9-0.83) / (0.7-0.3)
            # solve equation: x / eff = delta
            gradient = (0.9 - 0.83) / (0.7 - 0.3)
            a = gradient / nominal_power
            b = -gradient * 0.3 + 0.83

            p = np.polynomial.polynomial.Polynomial([-b * delta ** 2, a * delta ** 2, 1])
            charge_command = p.roots()[0]
        elif delta >= -charge_effect_08:
            charge_command = delta * (0.9 ** 0.5)
        elif delta >= -charge_effect_1:
            gradient = (0.85 - 0.9) / (1.0 - 0.8)
            a = gradient / nominal_power
            b = -gradient * 0.8 + 0.9

            p = np.polynomial.polynomial.Polynomial([-b * delta ** 2, a * delta ** 2, 1])
            charge_command = p.roots()[0]

    result_battery = deepcopy(battery)
    result_battery.charge(charge_command)
    result_battery.next_time_step()
    return charge_command, result_battery
