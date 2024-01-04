import random
import pandas as pd
import pickle
from scipy.stats import gaussian_kde
import numpy as np
import scipy.optimize as optimize
import cvxpy as cp


class DemandModel:
    def __init__(self, default_price=0.45, ped=-2):
        self.demand_df = pd.read_csv('base_demand.csv')
        self.ped = ped # Price elasticity coefficient
        self.default_price = default_price

    def _base_demand(self, t, start_area):
        # Calculate random demand, based on supply?
        trip_count = self.demand_df[(self.demand_df['Start Hour'] == t) &
                              (self.demand_df['Start Community Area Name'] == start_area)]['Trip_Count'].iloc[0]
        high_demand = self.demand_df[(self.demand_df['Start Hour'] == t) &
                              (self.demand_df['Start Community Area Name'] == start_area)]['High Demand'].iloc[0]
        if high_demand:
            rand_int = random.uniform(1, 1.5)
            return int(rand_int*trip_count)
        else:
            return trip_count

    def register_trips(self, t, area, trips):
        self.demand_df.loc[(self.demand_df['Start Hour'] == t + 1) & (self.demand_df['Start Community Area Name'] == area), 'Trip_Count'] = trips


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
    def __init__(self, max_battery_distance, area):
        self.max_battery_distance = max_battery_distance
        self.supply_df = pd.read_csv('base_supply.csv')
        self.supply_df.sort_values(['Start Hour', 'Start Community Area Name'], inplace=True)
        self.transition_matrix = pd.read_csv('transition_model.csv')
        self.transition_matrix.sort_values(['End Hour', 'End Community Area Name'], inplace=True)
        with open('distributions/trip_distance.pkl', 'rb') as file:
            trip_distance_kde = pickle.load(file)
        with open('distributions/average_distance_per_hour.pkl', 'rb') as f:
            self.average_distances = pickle.load(f)

        self.kde_model = gaussian_kde(trip_distance_kde['dataset'])
        self.kde_model.set_bandwidth(trip_distance_kde['bandwidth'])

    def transition_next_timestep(self, t, no_trips, temp=False, area=None):
        if (t + 1) % 24 == 0:
            return
        else:
            cur_available_scooters = self.supply_df[(self.supply_df['Start Hour'] == t)]['Available_Scooters']
            fractions = self.transition_matrix[(self.transition_matrix['End Hour'] == t)]['Average Fraction']
            next_available_scooters = cur_available_scooters.values + (fractions.values * no_trips)
            if not temp:
                self.supply_df.loc[self.supply_df['Start Hour'] == t + 1, 'Available_Scooters'] = next_available_scooters.astype(int)
            else:
                temp_df = self.supply_df.copy()
                temp_df.loc[temp_df['Start Hour'] == t + 1, 'Available_Scooters'] = next_available_scooters.astype(int)
                new_available_scooters = temp_df[(temp_df['Start Hour'] == t + 1) & (temp_df['Start Community Area Name'] == area)]['Available_Scooters']
                return new_available_scooters

    def base_supply(self, t, area):
        return self.supply_df[(self.supply_df['Start Hour'] == t) &
                                  (self.supply_df['Start Community Area Name'] == area)]['Trip_Count'].iloc[0]

    def _calc_distance_travelled(self, t, area):
        filtered_df = self.supply_df[(self.supply_df['Start Community Area Name'] == area) &
                                     (self.supply_df['Start Hour'] <= t)]
        cumulative_sum = filtered_df['Average_Trip_Duration'].sum()
        return cumulative_sum * (0.015 * t) # Assume average of 0.015 trips per hour per scooter (misschien nog iets doen met meerde keren per dag opladen ofzo, dit is wel heel weinig maar geeft nog een beetje resultaat)

    def estimate_supply(self, t, area, supply=None):
        if not supply:
            base_supply = self.base_supply(t, area)
        else:
            base_supply = supply
        avg_travelled = self._calc_distance_travelled(t, area)
        remaining_capacity = self.max_battery_distance - avg_travelled
        probability = 1 - self.kde_model.integrate_box_1d(remaining_capacity, np.inf)
        return probability * base_supply


class PPricingArea:

    def __init__(self, name, max_battery_distance):
        self.name = name
        self.supply_model =

    def calc_market_clearing_price(self, area):
        base_supply = self.supply_model.estimate_supply(self.t, area)
        p_c = self.demand_model.inverse_demand(self.t, area, base_supply)
        return p_c

class PPricing:
    def __init__(self, default_price=0.45, ped=-2, max_price=1, total_areas=41):
        self.t = 0
        self.total_areas = total_areas
        self.prev_trip_count = None
        self.area = None
        self.optimal_price = None
        self.max_price = max_price
        self.default_price = 0.45
        self.demand_model = DemandModel(default_price, ped)
        self.supply_model = SupplyModel(max_battery_distance=5400) # 5400 is batterij volgens google, onze berekening is gewoon kut

    def _calc_market_clearing_price(self, area):
        base_supply = self.supply_model.estimate_supply(self.t, area)
        p_c = self.demand_model.inverse_demand(self.t, area, base_supply)
        return p_c

    def _calc_revenue(self, p, area):
        base_supply = self.supply_model.estimate_supply(self.t, area)
        base_demand = self.demand_model.estimate_demand(self.t, area, p)
        trips = min(base_demand, base_supply)
        revenue = trips * p
        return -revenue

    def _calc_price_max_revenue(self, area):
        p_d = optimize.minimize_scalar(
            lambda p: self._calc_revenue(p, area),
            bounds=(0, self.max_price),
            method='bounded'
        )
        return p_d.x

    def _calc_optimal_local_price(self, area):
        p_c = self._calc_market_clearing_price(area)
        p_d = self._calc_price_max_revenue(area)
        if p_d <= p_c:
            self.optimal_price =  p_c
            return
        elif p_d > p_c:
            self.optimal_price = p_d
            return
        else:
            return 'error.'

    def _calc_trips(self, area, price):
        base_supply = self.supply_model.estimate_supply(self.t, area)
        base_demand = self.demand_model.estimate_demand(self.t, area, price)
        return min(base_demand, base_supply)

    def _calc_revenue_decrease(self, delta_t_i):
        # delta_t_i: change in trips at location i
        t_p_star = self._calc_trips(self.area, self.optimal_price)
        base_supply = self.supply_model.estimate_supply(self.t, self.area)
        delta_p = self.optimal_price - self.demand_model.inverse_demand(self.t, self.area, t_p_star + delta_t_i)
        rev_dec = t_p_star * delta_p + delta_t_i * delta_p - delta_t_i * self.optimal_price
        return rev_dec

    def _calc_market_clearing_price_scenarios(self, area, p_c, added_trips):
        if self.t - 1 < 0:
            prev_t = 0
        else:
            prev_t = self.t - 1

        if added_trips == 0:
            base_supply = self.supply_model.transition_next_timestep(prev_t, self.prev_trip_count, temp=True, area=area)
            supply = self.supply_model.estimate_supply(self.t, area, base_supply)
            p_c_2 = self.demand_model.inverse_demand(self.t, area, supply)
            p_c_1 = p_c
        else:
            base_supply = self.supply_model.transition_next_timestep(prev_t, self.prev_trip_count - added_trips, temp=True, area=area)
            supply = self.supply_model.estimate_supply(self.t, area, base_supply)
            p_c_1 = self.demand_model.inverse_demand(self.t, area, supply)
            p_c_2 = p_c
        return p_c_1, p_c_2

    def _calc_revenue_increase(self, delta_v_j):
        p_c = self._calc_market_clearing_price(self.area)
        p_c_1, p_c_2 = self._calc_market_clearing_price_scenarios(self.area, p_c, delta_v_j)

        base_supply = self.supply_model.base_supply(self.t, self.area)
        supply_s2 = self.supply_model.estimate_supply(self.t, self.area, base_supply + delta_v_j)
        supply_s1 = self.supply_model.estimate_supply(self.t, self.area, base_supply)
        revenue_s2 = supply_s2 * p_c_2
        revenue_s1 = supply_s1 * p_c_1
        return revenue_s2 - revenue_s1

    def optimize(self):
        transition_matrix = self.supply_model

        delta_V = cp.Variable(self.total_areas)
        delta_T = cp.Variable(self.total_areas)

        objective = cp.Maximize(
            sum([self._calc_revenue_increase(delta_V[j]) for j in range(self.total_areas)]) -
            sum([self._calc_revenue_decrease(delta_T[i]) for i in range(self.total_areas)])
        )

        constraints = [
            delta_T >= 0,
            delta_V >= 0,
            # Add any additional constraints you may have, for example:
            # cp.sum(delta_T) == total_increase_value,  # If you have a total increase value for delta T
            # cp.sum(delta_S) == total_increase_value,  # If you have a total increase value for delta S
        ]

        # Define the problem and solve it
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Extract the solution
        optimal_delta_S = delta_V.value
        optimal_delta_T = delta_T.value

        print("Optimal delta_S:", optimal_delta_S)
        print("Optimal delta_T:", optimal_delta_T)


    def run_algo(self):
        # update prev_trip_count, t
        pass

def test():
    demand = DemandModel()
    supply = SupplyModel(max_battery_distance=5400)
    cur_demand = demand.estimate_demand(1, 'WEST TOWN', 0.45)
    cur_supply = supply.estimate_supply(1, 'WEST TOWN')
    #print(cur_demand)
    #print(supply.estimate_supply(1, 'WEST TOWN'))
    #print(demand.estimate_demand(2, 'WEST TOWN', 0.45))
    #print(supply.estimate_supply(1, 'WEST TOWN', True))
    print(supply.supply_df[supply.supply_df['Start Hour'] == 1]['Available_Scooters'])
    print(supply.supply_df[supply.supply_df['Start Hour'] == 2]['Available_Scooters'])
    trips = min(cur_supply, cur_demand)
    supply.transition_next_timestep(1, trips)
    print(supply.supply_df[supply.supply_df['Start Hour'] == 2]['Available_Scooters'])
    #ppricing = PPricing()
    #print(ppricing._calc_optimal_local_price('WEST TOWN'))
    """
    supply._transition_next_timestep(1, 'WEST TOWN')
    print(supply.estimate_supply(2, 'WEST TOWN'))
    print(demand.estimate_demand(2, 'WEST TOWN', 0.45))
    """

test()

