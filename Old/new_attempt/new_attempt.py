import random
import pandas as pd
import pickle
from scipy.stats import gaussian_kde
import numpy as np
import scipy.optimize as optimize
import pygad


class SupplyDemandModel:
    def __init__(self, area_name, default_price, ped, max_battery_distance):
        self.max_battery_distance = max_battery_distance
        self.area_name = area_name
        self.default_price = default_price
        self.ped = ped
        self.trips_per_hour = 0.015

        self.supply_df = pd.read_csv('base_supply.csv')
        self.supply_df.sort_values(['Start Hour'], inplace=True)
        self.supply_df = self.supply_df.loc[self.supply_df['Start Community Area Name'] == area_name].copy()
        self.transition_matrix = pd.read_csv('transition_model.csv')
        self.transition_matrix = self.transition_matrix.loc[self.transition_matrix['End Community Area Name'] == area_name].copy()
        self.transition_matrix.sort_values(['End Hour'], inplace=True)
        self.demand_df = pd.read_csv('base_demand.csv')
        self.demand_df = self.demand_df.loc[self.demand_df['Start Community Area Name'] == area_name].copy()
        self.demand_df.sort_values(['Start Hour'], inplace=True)

        with open('../../distributions/trip_distance.pkl', 'rb') as file:
            trip_distance_kde = pickle.load(file)
        with open('../../distributions/average_distance_per_hour.pkl', 'rb') as f:
            self.average_distances = pickle.load(f)

        self.kde_model = gaussian_kde(trip_distance_kde['dataset'])
        self.kde_model.set_bandwidth(trip_distance_kde['bandwidth'])

    def transition_next_timestep(self, t, no_trips, temp=False):
        if (t + 1) % 24 == 0:
            return
        else:
            cur_available_scooters = self.supply_df[(self.supply_df['Start Hour'] == t)]['Available_Scooters'].iloc[0]
            fractions = self.transition_matrix[(self.transition_matrix['End Hour'] == t)]['Average Fraction'].iloc[0]
            next_available_scooters = cur_available_scooters + (fractions * no_trips)
            if not temp:
                self.supply_df.loc[self.supply_df['Start Hour'] == t + 1, 'Available_Scooters'] = next_available_scooters.astype(int)
            else:
                return next_available_scooters.astype(int)

    def base_supply(self, t):
        return self.supply_df[self.supply_df['Start Hour'] == t]['Trip_Count'].iloc[0]

    def _calc_distance_travelled(self, t):
        filtered_df = self.supply_df[self.supply_df['Start Hour'] <= t]
        cumulative_sum = filtered_df['Average_Trip_Duration'].sum()
        return cumulative_sum * (self.trips_per_hour * t)

    def estimate_supply(self, t, supply=None):
        if not supply:
            base_supply = self.base_supply(t)
        else:
            base_supply = supply
        avg_travelled = self._calc_distance_travelled(t)
        remaining_capacity = self.max_battery_distance - avg_travelled
        probability = 1 - self.kde_model.integrate_box_1d(remaining_capacity, np.inf)
        return probability * base_supply

    def base_demand(self, t, requests):
        high_demand = self.demand_df[self.demand_df['Start Hour'] == t]['High Demand'].iloc[0]
        if high_demand:
            rand_int = random.uniform(1, 1.5)
            return int(rand_int*requests)
        else:
            return requests

    def estimate_demand(self, t, new_price, requests):
        base_demand = self.base_demand(t, requests)
        price_change = (new_price - self.default_price) / self.default_price
        new_demand = base_demand * (1 + self.ped * price_change)
        return max(int(new_demand), 0)

    def inverse_demand(self, t, new_demand, requests):
        base_demand = self.base_demand(t, requests)
        if base_demand == 0:
            raise ValueError("Base demand cannot be zero.")

        price_change_ratio = (new_demand / base_demand - 1) / self.ped
        new_price = self.default_price * (1 + price_change_ratio)
        return max(new_price, 0)

    def get_requests(self, supply):
        requests = int(supply * random.uniform(0.85, 1))
        if requests <= 0:
            return 1
        else:
            return requests

class PPricingArea:
    def __init__(self, area_name, max_price, default_price, ped, max_battery_distance):
        self.area_name = area_name
        self.max_price = max_price
        self.supply_demand_model = SupplyDemandModel(area_name, default_price, ped, max_battery_distance)
        self.default_price = default_price
        self.t = 0

        self.requests = None
        self.p_c = None
        self.base_supply = None
        self.base_trips = None
        self.optimal_price = None
        self.base_demand = None
        self.available_scooters = None

        self.supply_next_s2 = None
        self.supply_next_s1 = None

    def update(self):
        self.available_scooters = self.supply_demand_model.base_supply(self.t)
        self.base_supply = self.supply_demand_model.estimate_supply(self.t)
        self.requests = self.supply_demand_model.get_requests(self.base_supply)
        self.p_c = self._calc_market_clearing_price()
        self.optimal_price = self._calc_optimal_price()
        self.base_demand = self.supply_demand_model.estimate_demand(self.t, self.default_price, self.requests)
        self.base_trips = self._calc_trips(self.base_supply, self.base_demand)

    def calc_revenue_decrease(self, added_trips):
        optimal_price_demand = self.supply_demand_model.estimate_demand(self.t, self.optimal_price, self.requests)
        optimal_price_trips = self._calc_trips(self.base_supply, optimal_price_demand)
        inverse_demand_optimal_price = self.supply_demand_model.inverse_demand(self.t, optimal_price_trips + added_trips, self.requests)
        delta_p = self.optimal_price - inverse_demand_optimal_price
        rev_dec = optimal_price_trips * delta_p + added_trips * delta_p - added_trips * self.optimal_price
        return rev_dec

    def calc_revenue_increase(self, added_supply):
        revenue_s2 = self._calc_revenue_s2(added_supply)
        revenue_s1 = self.base_trips * self.optimal_price
        return revenue_s2 - revenue_s1

    def _calc_revenue_s2(self, added_supply):
        new_next_available_scooters = self.supply_demand_model.base_supply(self.t + 1) + added_supply # TODO: t+1 fixen.
        new_next_supply = self.supply_demand_model.estimate_supply(self.t+1, new_next_available_scooters)
        new_next_requests = self.supply_demand_model.get_requests(new_next_supply)
        new_next_demand = self.supply_demand_model.estimate_demand(self.t + 1, self.default_price, new_next_requests)

        new_price = self.supply_demand_model.inverse_demand(self.t + 1, new_next_demand, new_next_requests)
        new_trips = self._calc_trips(new_next_supply, new_next_demand)

        return new_price * new_trips

    def _calc_trips(self, supply, demand):
        return min(supply, demand)

    def _calc_market_clearing_price(self):
        base_supply = self.supply_demand_model.estimate_supply(self.t)
        p_c = self.supply_demand_model.inverse_demand(self.t, base_supply, self.requests)
        return p_c

    def _calc_revenue(self, p):
        base_supply = self.supply_demand_model.estimate_supply(self.t)
        base_demand = self.supply_demand_model.estimate_demand(self.t, p, self.requests)
        trips = min(base_demand, base_supply)
        revenue = trips * p
        return -revenue

    def _calc_price_max_revenue(self):
        p_d = optimize.minimize_scalar(
            lambda p: self._calc_revenue(p),
            bounds=(0, self.max_price),
            method='bounded'
        )
        return p_d.x

    def _calc_optimal_price(self):
        p_c = self._calc_market_clearing_price()
        p_d = self._calc_price_max_revenue()
        if p_d <= p_c:
            return p_c
        else:
            return p_d

    def calc_delta_p(self, delta_t):
        price_for_updated_demand = self.supply_demand_model.inverse_demand(self.t, self.base_demand - delta_t, self.requests)
        return self.optimal_price - price_for_updated_demand


class PPricing:
    def __init__(self, default_price=0.45, ped=-2, max_price=1, total_areas=41, max_battery_distance=4500):
        self.t = 0
        self.available_scooters = 0
        self.total_areas = total_areas
        self.max_price = max_price
        self.default_price = default_price
        self.areas = []
        self.ped = ped
        self.max_battery_distance = max_battery_distance
        self.transition_matrix = pd.read_csv('transition_model.csv')

    def init_areas(self, names):
        for area_name in names:
            self.areas.append(PPricingArea(
                area_name = area_name,
                max_price = self.max_price,
                default_price = self.default_price,
                ped = self.ped,
                max_battery_distance = self.max_battery_distance
            ))

    def _multiply_t_transition(self, delta_T):
        self.transition_matrix.sort_values(['End Hour', 'End Community Area Name'], inplace=True)
        curr_fractions = 1 + self.transition_matrix.loc[self.transition_matrix['End Hour'] == self.t]['Average Fraction'].copy()
        next_v_estimate = curr_fractions.values * delta_T
        return next_v_estimate.astype(int)

    def _calc_revenue_decrease(self, delta_T):
        rev_dec = 0
        for area, added_trips in zip(self.areas, delta_T):
            rev_dec += area.calc_revenue_decrease(added_trips)
        return rev_dec

    def _calc_revenue_increase(self, delta_V):
        rev_inc = 0
        for area, added_supply in zip(self.areas, delta_V):
            rev_inc += area.calc_revenue_increase(added_supply)
        return rev_inc

    def objective_function(self, instance, solution, solution_idx):
        delta_T = solution
        delta_V = self._multiply_t_transition(delta_T)

        if not sum(delta_V) == self.available_scooters:
            revenue = -1000000
        else:
            revenue = self._calc_revenue_increase(delta_V) - self._calc_revenue_decrease(delta_T)
        print(revenue)
        return revenue

    def _update_areas(self):
        available_scooters = 0
        for area in self.areas:
            area.t = self.t
            area.update()
            available_scooters += area.available_scooters
        self.available_scooters = available_scooters
        print(available_scooters)
        return

    def _find_optimal_price(self, optimal_delta_T):
        optimal_prices = {}
        for area, added_trips in zip(self.areas, optimal_delta_T):
            delta_p = area.optimal_price - area.calc_delta_p(added_trips)
            optimal_prices[area.area_name] = area.optimal_price - delta_p

        return optimal_prices

    def run_algo(self):
        self._update_areas()
        ga_instance = pygad.GA(num_generations=10000,
                               num_parents_mating=2,
                               fitness_func=self.objective_function,
                               sol_per_pop=5,
                               num_genes=self.total_areas,
                               gene_type=int,
                               gene_space={'low': 0, 'high': 150},
                               parent_selection_type="rank",
                               crossover_type="single_point",
                               mutation_type="random",
                               mutation_percent_genes=10)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        optimal_prices = self._find_optimal_price(solution)
        print(optimal_prices)


def test():
    df = pd.read_csv('base_supply.csv')
    area_names = df['Start Community Area Name'].unique().tolist()

    ppricing = PPricing()
    ppricing.init_areas(area_names)
    ppricing.run_algo()

test()

