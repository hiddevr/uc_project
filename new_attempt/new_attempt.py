import pandas as pd
import pickle
from scipy.stats import gaussian_kde
import numpy as np


class DemandModel:
    def __init__(self, default_price=0.45, ped=-2):
        self.demand_df = pd.read_csv('base_demand.csv')
        self.ped = ped # Price elasticity coefficient
        self.default_price = default_price

    def _base_demand(self, t, start_area):
        return self.demand_df[(self.demand_df['Start Hour'] == t) &
                              (self.demand_df['Start Community Area Name'] == start_area)]['Trip_Count'].iloc[0]

    def estimate_demand(self, t, start_area, new_price):
        base_demand = self._base_demand(t, start_area)
        price_change = (new_price - self.default_price) / self.default_price
        new_demand = base_demand * (1 + self.ped * price_change)
        return max(int(new_demand), 0)

    def inverse_demand(self, t, start_area, new_demand):
        base_demand = self._base_demand(t, start_area)
        if base_demand == 0:
            raise ValueError("Base demand cannot be zero.")

        price_change_ratio = (new_demand / base_demand - 1) / self.ped
        new_price = self.default_price * (1 + price_change_ratio)
        return max(new_price, 0)


class SupplyModel:
    def __init__(self, max_battery_distance):
        self.max_battery_distance = max_battery_distance
        self.supply_df = pd.read_csv('base_supply.csv')
        self.transition_matrix = pd.read_csv('transition_model.csv')
        with open('distributions/trip_distance.pkl', 'rb') as file:
            trip_distance_kde = pickle.load(file)
        with open('distributions/average_distance_per_hour.pkl', 'rb') as f:
            self.average_distances = pickle.load(f)

        self.kde_model = gaussian_kde(trip_distance_kde['dataset'])
        self.kde_model.set_bandwidth(trip_distance_kde['bandwidth'])

    def _next_timestep(self):
        """
        Functie om aantal e-scooters op een plek te berekenen op basis van transition matrix
        Die matrix heeft de kans dat een e-scooter in die area aankomt
        Dus denk gewoon voor elke area 1 + die fraction * needed_scooters in the volgende timestep
        :return:
        """

    def _base_supply(self, t, area):
        return self.supply_df[(self.supply_df['Start Hour'] == t) &
                              (self.supply_df['Start Community Area Name'] == area)]['Needed_Scooters'].iloc[0]

    def _calc_distance_travelled(self, t, area):
        filtered_df = self.supply_df[(self.supply_df['Start Community Area Name'] == area) &
                                     (self.supply_df['Start Hour'] <= t)]
        cumulative_sum = filtered_df['Average_Trip_Duration'].sum()
        return cumulative_sum * (0.015 * t) # Assume average of 0.015 trips per hour per scooter (misschien nog iets doen met meerde keren per dag opladen ofzo, dit is wel heel weinig maar geeft nog een beetje resultaat)

    def estimate_supply(self, t, area):
        base_supply = self._base_supply(t, area)
        print(base_supply)
        avg_travelled = self._calc_distance_travelled(t, area)
        remaining_capacity = self.max_battery_distance - avg_travelled
        probability = 1 - self.kde_model.integrate_box_1d(remaining_capacity, np.inf)
        return probability * base_supply

class PPricing:

    def __init__(self, default_price=0.45, ped=-2):
        self.t = 0
        self.demand_model = DemandModel(default_price, ped)
        self.supply_model = SupplyModel(max_battery_distance=5400) # 5400 is batterij volgens google, onze berekening is gewoon kut

    def _calc_revenue_decrease(self):
        pass

    def _calc_revenue_increase(self):
        pass

    def optimize(self):
        pass

def test():
    demand = DemandModel()
    supply = SupplyModel(max_battery_distance=5400)
    cur_demand = demand.estimate_demand(1, 'EAST GARFIELD PARK', 0.45)
    print(demand.inverse_demand(1, 'EAST GARFIELD PARK', cur_demand + 10))
    print(supply.estimate_supply(21, 'WEST TOWN'))

test()

