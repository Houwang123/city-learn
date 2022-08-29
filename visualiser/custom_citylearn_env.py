from citylearn.citylearn import CityLearnEnv
import numpy as np

# CityLearnEnv.render

class CustomCityLearnEnv(CityLearnEnv):
    def render(self):
        """On top of the original render, merge in action & reward info"""

        usage_image = super().render()
        # print(super().rewards)
        return usage_image