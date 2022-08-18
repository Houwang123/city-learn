# Reward Function

Participants are to edit the `get_reward()` function in [get_reward.py](get_reward.py). Three observations from the environment are provided for the reward calculation and they include:

1. `electricity_consumption`: List of each building's/total district electricity consumption in [kWh].
2. `carbon_emission`: List of each building's/total district carbon emissions in [kg_co2].
3. `electricity_price`: List of each building's/total district electricity price in [$].

Where `G_n` and `C_n` are respectively the `carbon_emission` and `electricity_price` of the building(s) controlled by agent `n`.

__Note__ that `get_reward()` function must return a `list` whose length is equal to the number of agents in the environment.

Do not edit [user_reward.py](user_reward.py) module!

