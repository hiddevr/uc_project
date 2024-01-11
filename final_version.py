import random
import pandas as pd
import pickle
from scipy.stats import gaussian_kde
import numpy as np
import scipy.optimize as optimize
import tqdm



class SupplyDemandModel:
    def __init__(self, area_name, default_price, ped, max_battery_distance, alpha=1, beta=0.1, min_req=0.95,
                 max_req=1.1):
        self.max_battery_distance = max_battery_distance
        self.area_name = area_name
        self.default_price = default_price
        self.ped = ped
        self.trips_per_hour = 0.015
        self.alpha = alpha
        self.beta = beta
        self.min_req = min_req
        self.max_req = max_req

        self.supply_df = pd.read_csv('Old/new_attempt/base_supply.csv')
        self.supply_df.sort_values(['Start Hour'], inplace=True)
        self.supply_df = self.supply_df.loc[self.supply_df['Start Community Area Name'] == area_name].copy()
        self.transition_matrix = pd.read_csv('Old/new_attempt/transition_model.csv')
        self.transition_matrix = self.transition_matrix.loc[
            self.transition_matrix['End Community Area Name'] == area_name].copy()
        self.transition_matrix.sort_values(['End Hour'], inplace=True)
        self.demand_df = pd.read_csv('Old/new_attempt/base_demand.csv')
        self.demand_df = self.demand_df.loc[self.demand_df['Start Community Area Name'] == area_name].copy()
        self.demand_df.sort_values(['Start Hour'], inplace=True)

        with open('distributions/trip_distance.pkl', 'rb') as file:
            trip_distance_kde = pickle.load(file)
        with open('distributions/average_distance_per_hour.pkl', 'rb') as f:
            self.average_distances = pickle.load(f)

        self.kde_model = gaussian_kde(trip_distance_kde['dataset'])
        self.kde_model.set_bandwidth(trip_distance_kde['bandwidth'])

    def transition_next_timestep(self, t, added_trips, temp=False):
        cur_available_scooters = \
        self.supply_df[(self.supply_df['Start Hour'] == ((t + 1) % 24))]['Available_Scooters'].iloc[0]
        fractions = self.transition_matrix[(self.transition_matrix['End Hour'] == t)]['Average Fraction'].iloc[0]
        next_available_scooters = cur_available_scooters + (fractions * added_trips)
        if not temp:
            self.supply_df.loc[
                self.supply_df['Start Hour'] == ((t + 1) % 24), 'Available_Scooters'] = next_available_scooters.astype(
                int)
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

    def demand_proba(self, p=None):
        if not p:
            p = self.default_price
        probability = self.alpha * np.exp(-self.beta * p)
        return probability

    def get_requests(self, supply):
        requests = int(supply * random.uniform(self.min_req, self.max_req))
        if requests <= 0:
            return 1
        else:
            return requests

    def estimate_demand(self, supply, price):
        demand = self.demand_proba(price) * self.get_requests(supply)
        return max(int(demand), 0)

    def inverse_demand(self, required_demand, supply):
        requests = self.get_requests(supply)
        if requests <= 0 or required_demand <= 0:
            requests = 1
            required_demand = 1
        price = -1 / self.beta * np.log(required_demand / (self.alpha * requests))
        return max(price, 0)


class PPricingArea:
    def __init__(self, area_name, max_price, default_price, ped, max_battery_distance, alpha=1, beta=0.1,
                 min_req = 0.95, max_req=1.1):
        self.area_name = area_name
        self.max_price = max_price
        self.supply_demand_model = SupplyDemandModel(area_name, default_price, ped, max_battery_distance, alpha, beta,
                                                     min_req, max_req)
        self.default_price = default_price
        self.t = 0
        self.available_scooters = None

        self.current_demand = None
        self.current_supply = None
        self.current_optimal_price = None
        self.current_requests = None
        self.current_trips = None

        self.next_supply = None
        self.next_requests = None
        self.next_optimal_price = None
        self.next_demand = None
        self.next_trips = None
        self.next_t = None

    def update(self):
        self.available_scooters = self.supply_demand_model.base_supply(self.t)

        self.current_supply = self.supply_demand_model.estimate_supply(self.t)
        self.current_requests = self.supply_demand_model.get_requests(self.current_supply)
        self.current_demand = self.supply_demand_model.estimate_demand(self.current_supply, self.default_price)
        self.current_optimal_price = self._calc_optimal_price(self.current_supply, self.current_demand)
        self.current_trips = self._calc_trips(self.current_supply, self.current_demand)

        if self.t + 1 > 23:
            self.next_t = 0
        else:
            self.next_t = self.t

        self.next_supply = self.supply_demand_model.estimate_supply(self.next_t)
        self.next_requests = self.supply_demand_model.get_requests(self.next_supply)
        self.next_demand = self.supply_demand_model.estimate_demand(self.next_supply, self.default_price)
        self.next_optimal_price = self._calc_optimal_price(self.next_supply, self.next_demand)
        self.next_trips = self._calc_trips(self.next_supply, self.next_demand)

    def transition(self, added_trips):
        self.supply_demand_model.transition_next_timestep(self.t, added_trips)

    def calc_revenue_decrease(self, added_trips):
        optimal_price_demand = self.supply_demand_model.estimate_demand(self.current_supply, self.current_optimal_price)
        optimal_price_trips = self._calc_trips(self.current_supply, optimal_price_demand)
        inverse_demand_optimal_price = self.supply_demand_model.inverse_demand(optimal_price_trips + added_trips,
                                                                               self.current_supply)
        delta_p = self.current_optimal_price - inverse_demand_optimal_price
        rev_dec = optimal_price_trips * delta_p + added_trips * delta_p - added_trips * self.current_optimal_price
        return -rev_dec

    def calc_revenue_increase(self, added_supply):
        revenue_s2 = self._calc_revenue_s2(added_supply)
        revenue_s1 = self.next_trips * self.next_optimal_price
        rev_inc = revenue_s2 - revenue_s1
        return rev_inc

    def _calc_revenue_s2(self, added_supply):
        new_next_available_scooters = self.supply_demand_model.base_supply(self.next_t) + added_supply
        new_next_supply = self.supply_demand_model.estimate_supply(self.next_t, new_next_available_scooters)
        new_next_demand = self.supply_demand_model.estimate_demand(new_next_supply, self.next_optimal_price)
        new_price = self.supply_demand_model.inverse_demand(new_next_demand, new_next_supply)
        new_trips = self._calc_trips(new_next_supply, new_next_demand)
        return new_price * new_trips

    def _calc_trips(self, supply, demand):
        return min(supply, demand)

    def _calc_market_clearing_price(self, supply, demand):
        p_c = self.supply_demand_model.inverse_demand(demand, supply)
        return p_c

    def _calc_revenue(self, p, supply, demand):
        trips = min(demand, supply)
        revenue = trips * p
        return revenue

    def _calc_price_max_revenue(self, supply, demand):
        p_d = optimize.minimize_scalar(
            lambda p: self._calc_revenue(p, supply, demand),
            bounds=(0, self.max_price),
            method='bounded'
        )
        return p_d.x

    def _calc_optimal_price(self, supply, demand):
        p_c = self._calc_market_clearing_price(supply, demand)
        p_d = self._calc_price_max_revenue(supply, demand)
        if p_d <= p_c:
            return p_c
        else:
            return p_d

    def calc_delta_p(self, delta_t):
        price_for_updated_demand = self.supply_demand_model.inverse_demand(
            self.supply_demand_model.estimate_demand(self.current_supply, self.current_optimal_price) - delta_t,
            self.current_supply)
        return self.current_optimal_price - price_for_updated_demand


class PPricing:
    def __init__(self, default_price=0.45, ped=-2, max_price=1, total_areas=41, max_battery_distance=4500, max_evals=10,
                 min_price=0.1):
        self.t = 0
        self.max_evals = max_evals
        self.min_price = min_price
        self.available_scooters = 0
        self.total_areas = total_areas
        self.max_price = max_price
        self.default_price = default_price
        self.areas = []
        self.ped = ped
        self.max_battery_distance = max_battery_distance
        self.transition_matrix = pd.read_csv('Old/new_attempt/transition_model.csv')
        self.transition_matrix.sort_values(['End Hour', 'End Community Area Name'], inplace=True)
        self.default_revenue = self._calc_default_revenue()

    def _calc_default_revenue(self):
        df = pd.read_csv('Old/new_attempt/base_demand.csv')
        df['revenue'] = df['Price'] * df['Trip_Count']
        grouped_sum = df.groupby('Start Hour')['revenue'].sum()
        return grouped_sum.to_dict()

    def init_areas(self, names, alpha=1, beta=0.1, min_req=0.95, max_req=1.1):
        for area_name in names:
            self.areas.append(PPricingArea(
                area_name=area_name,
                max_price=self.max_price,
                default_price=self.default_price,
                ped=self.ped,
                max_battery_distance=self.max_battery_distance,
                alpha=alpha,
                beta=beta,
                min_req=min_req,
                max_req=max_req
            ))

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

    def _update_areas(self):
        available_scooters = 0
        for area in self.areas:
            area.t = self.t
            area.update()
            available_scooters += area.available_scooters
        self.available_scooters = available_scooters
        return

    def _find_optimal_price(self, optimal_delta_T):
        optimal_prices = {}
        for area, added_trips in zip(self.areas, optimal_delta_T):
            delta_p = area.current_optimal_price - area.calc_delta_p(added_trips)
            if area.current_optimal_price - delta_p > self.min_price:
                optimal_prices[area.area_name] = area.current_optimal_price - delta_p
            else:
                optimal_prices[area.area_name] = self.min_price

            area.transition(added_trips)
        return optimal_prices

    def _objective_function(self, delta_T, delta_V):
        revenue = self._calc_revenue_increase(delta_V) - self._calc_revenue_decrease(delta_T)
        return revenue

    def _sum_constraint(self, delta_T, curr_fractions):
        current_sum = int(np.sum(curr_fractions * delta_T))
        while current_sum != self.available_scooters:
            if current_sum > self.available_scooters:
                indices = np.where(delta_T > 1)[0]
                if len(indices) > 0:
                    delta_T[np.random.choice(indices)] -= 1
            else:
                delta_T[np.random.randint(len(delta_T))] += 1
            current_sum = int(np.sum(curr_fractions * delta_T))
        return delta_T

    def _calc_t_total_revenue(self, optimal_prices):
        new_revenue = 0
        avg_price = 0
        for area in self.areas:
            price = optimal_prices[area.area_name]
            new_revenue += price * area.current_trips
            avg_price += price
        avg_price = avg_price / len(optimal_prices)
        return new_revenue, avg_price

    def _random_search(self, curr_fractions):
        cur_best_revenue = 0
        cur_best_delta_T = None

        for i in range(self.max_evals):
            delta_T = np.random.randint(0, min(self.available_scooters, 250), size=curr_fractions.shape[0])
            delta_T = self._sum_constraint(delta_T, curr_fractions)
            delta_V = (delta_T * curr_fractions).astype(int)
            revenue = self._objective_function(delta_T, delta_V)
            if revenue > cur_best_revenue:
                cur_best_revenue = revenue
                cur_best_delta_T = delta_T
            # print(f'Evaluation: {i}\nCurrent best revenue: {cur_best_revenue}')
        return cur_best_delta_T

    def run_algo(self):
        new_revenue_lst = []
        default_lst = []
        avg_prices = []
        for j in range(24):
            # No data for 3AM and 4AM, so skip.
            if j == 3 or j == 4:
                continue
            print(f'Running for t = {j}')
            self.t = j
            self._update_areas()
            curr_fractions = 1 + self.transition_matrix.loc[self.transition_matrix['End Hour'] == self.t][
                'Average Fraction'].copy().values
            solution = self._random_search(curr_fractions)
            optimal_prices = self._find_optimal_price(solution)
            new_revenue, avg_price = self._calc_t_total_revenue(optimal_prices)
            new_revenue_lst.append(new_revenue)
            avg_prices.append(avg_price)

            print(f'Total revenue using PPricing for t = {j}: {new_revenue}, avg price: {avg_price}')
            print(f'Total revenue using default price for t = {j}: {self.default_revenue.get(j)}')

            default_lst.append(self.default_revenue.get(j))

        print(f'Total revenue PPricing: {sum(new_revenue_lst)}, average: {sum(new_revenue_lst) / len(new_revenue_lst)}')
        print(f'Total revenue default price: {sum(default_lst)}, average: {sum(default_lst) / len(default_lst)}')
        return new_revenue_lst, default_lst, avg_prices




