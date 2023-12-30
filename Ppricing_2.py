import pickle
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint


class PPricing:

    def __init__(self):
        self.max_battery_distance = 1090013.4
        self.alpha = 1
        self.beta = 0.1
        self.p = 0.45

        self.total_available_scooters = 0
        self.t = 0
        self.predicted_requests = None
        self.transition_matrix = pd.DataFrame()
        self.available_scooters = None
        self.requests = None
        self.next_available_scooters = None
        self.next_total_available_scooters = 0

        self.test = 0

    def _probability_passenger_willing_to_pay(self, p=0.45):
        """
        Function for calculating probability of passenger willing to pay price p
        :param p: Price
        :param alpha: Scaling to adjust base demand probability
        :param beta: Rate at which demand decreases as price increases
        :return: Probability of passenger willing to pay price p
        """
        probability = self.alpha * np.exp(-self.beta * p)
        return probability

    def _probability_sufficient_battery(self, t, additional_trips=0):
        """
        Function for calculating if an e-scooter has sufficient battery for a trip.
        :param t: current time period
        :param additional_distance: potentially added trips
        :return: Probability of e-scooter having sufficient battery for trip
        """
        with open('distributions/trip_distance.pkl', 'rb') as file:
            trip_distance_kde = pickle.load(file)

        with open('distributions/average_distance_per_hour.pkl', 'rb') as f:
            average_distances = pickle.load(f)

        kde_model = gaussian_kde(trip_distance_kde['dataset'])
        kde_model.set_bandwidth(trip_distance_kde['bandwidth'])

        avg_travelled = average_distances[t]
        remaining_capacity = self.max_battery_distance - avg_travelled - (additional_trips * t)
        probability = 1 - kde_model.integrate_box_1d(remaining_capacity, np.inf)

        return probability

    def _calc_demand(self, requests):
        requests_array = requests
        demand = requests_array * (1 - self._probability_passenger_willing_to_pay(self.p))
        return demand

    def _calc_inverse_demand(self, no_trips, demand):
        p = -np.log((1 - no_trips / demand) / self.alpha) / self.beta
        return p

    def _calc_next_supply_demand_df(self, transition_matrix, supply_demand_df, t):
        new_supply_demand_lst = []
        for index, row in supply_demand_df.iterrows():
            demand = self._calc_demand(supply_demand_df['requests'])
            supply = self._calc_supply(supply_demand_df['available_scooters'], t)
            total_trips_t = self._calc_trips_amount(demand, supply)
            destination_column = transition_matrix[str(row['region'])]
            added_scooters = (destination_column * total_trips_t).sum()
            unused_scooters = supply - self._calc_trips_amount(demand, supply)
            self.next_available_scooters = added_scooters + unused_scooters
            new_supply_demand_lst.append({
                'requests': row['predicted_requests'],
                'available_scooters': added_scooters + unused_scooters,
                'region': row['region'],
                'predicted_requests': row['predicted_requests'],
            })
        next_supply_demand_df = pd.DataFrame(new_supply_demand_lst)
        self.next_available_scooters = next_supply_demand_df['available_scooters'].values
        return

    def _calc_supply(self, available_scooters, t):
        supply = available_scooters * self._probability_sufficient_battery(t)
        return supply

    def _calc_trips_amount(self, demand, supply):
        self.test = self.test + 1
        print(self.test)
        demand_array = demand
        supply_array = supply
        trips = np.minimum(demand_array, supply_array)
        return trips

    def _calc_revenue_decrease(self, delta_T):
        supply = self._calc_supply(self.available_scooters, self.t)
        demand = self._calc_demand(self.requests)
        total_trips = self._calc_trips_amount(demand, supply)
        no_trips = delta_T + total_trips
        delta_p = self.p - self._calc_inverse_demand(no_trips, demand)
        rev_dec = total_trips * delta_p + delta_T * delta_p - delta_T * self.p
        return rev_dec

    def _calc_revenue_increase(self, delta_V):
        new_available_scooters = self.next_available_scooters + delta_V
        new_supply = self._calc_supply(new_available_scooters, self.t + 1)
        new_demand = self._calc_demand(self.predicted_requests)
        print(new_supply)
        print(new_demand)
        new_trips = self._calc_trips_amount(new_demand, new_supply)
        new_trips_sufficient_battery = new_trips * self._probability_sufficient_battery(self.t + 1, delta_V)

        old_supply = self._calc_supply(self.total_available_scooters, self.t)
        old_demand = self._calc_demand(self.predicted_requests)
        old_trips = self._calc_trips_amount(old_demand, old_supply)
        old_trips_sufficient_battery = self._probability_sufficient_battery(self.t, old_trips)
        rev_increase = (new_trips_sufficient_battery - old_trips_sufficient_battery) * self.p
        return rev_increase

    def _optimize_objective(self, x):
        n = len(x) // 2
        delta_T, delta_V = x[:n], x[n:]
        return -(sum(self._calc_revenue_increase(delta_V)) - sum(self._calc_revenue_decrease(delta_T)))

    def _estimate_gradient(self, x, epsilon=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = np.copy(x)
            x_minus = np.copy(x)
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (self._optimize_objective(x_plus) - self._optimize_objective(x_minus)) / (2 * epsilon)
        return grad

    def _gradient_descent(self, x0, learning_rate=1e-3, max_iterations=1000, tolerance=1e-6):
        x = np.copy(x0)
        for _ in range(max_iterations):
            grad = self._estimate_gradient(x)
            new_x = x - learning_rate * grad
            if np.linalg.norm(new_x - x) < tolerance:
                break
            x = new_x
        return x

    def _optimize_solve(self):
        n = self.transition_matrix.shape[0]
        x0 = np.random.randint(1, 6, 2 * n)

        optimized_x = self._gradient_descent(x0)
        optimal_delta_T = optimized_x[:n]
        optimal_delta_V = optimized_x[n:]
        return optimal_delta_T, optimal_delta_V

    def p_pricing(self, t, supply_demand_df):
        transition_matrix = pd.read_csv('transition_matrix.csv')

        self.total_available_scooters = supply_demand_df['available_scooters'].sum()
        self.t = t
        self.available_scooters = supply_demand_df['available_scooters'].values
        self.requests = supply_demand_df['requests'].values
        self.predicted_requests = supply_demand_df['predicted_requests'].values
        self.transition_matrix = transition_matrix.values
        self._calc_next_supply_demand_df(transition_matrix, supply_demand_df, t)

        optimal_delta_T, optimal_delta_V = self._optimize_solve()
        return optimal_delta_T, optimal_delta_V




