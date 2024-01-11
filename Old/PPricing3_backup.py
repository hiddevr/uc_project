import pickle
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.optimize import approx_fprime


class PPricing:

    def __init__(self):
        self.max_battery_distance = 1090013.4
        self.alpha = 1
        self.beta = 0.1
        self.p = np.full(41, 0.45)

        self.total_available_scooters = 0
        self.t = 0
        self.predicted_requests = None
        self.transition_matrix = None
        self.available_scooters = None
        self.requests = None
        self.next_available_scooters = None
        self.next_total_available_scooters = 0

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

    def _probability_sufficient_battery(self, t):
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
        remaining_capacity = self.max_battery_distance - avg_travelled
        probability = 1 - kde_model.integrate_box_1d(remaining_capacity, np.inf)

        return probability

    def _calc_demand(self, requests):
        requests_array = np.array(requests)  # Convert to numpy array if not already
        demand = requests_array * (1 - self._probability_passenger_willing_to_pay(self.p))
        return demand

    def _calc_inverse_demand(self, no_trips, demand):
        non_zero_result = np.where(demand != 0, 1 - no_trips / demand, 0)
        p = -np.log(np.maximum(non_zero_result / self.alpha, 1e-10)) / self.beta
        p = np.nan_to_num(p)
        return p

    def _calc_supply(self, available_scooters, t):
        supply = np.array(available_scooters) * self._probability_sufficient_battery(t)
        return supply

    def _calc_trips_amount(self, demand, supply):
        demand_array = np.array(demand)
        supply_array = np.array(supply)
        trips = np.minimum(demand_array, supply_array)
        return trips

    def _calc_revenue_increase(self, delta_V):
        new_available_scooters = self.next_available_scooters + delta_V
        new_supply = self._calc_supply(new_available_scooters, self.t + 1)
        new_demand = self._calc_demand(self.predicted_requests)
        new_trips = self._calc_trips_amount(new_demand, new_supply)
        new_trips_sufficient_battery = new_trips * self._probability_sufficient_battery(self.t + 1)

        old_supply = self._calc_supply(self.total_available_scooters, self.t)
        old_demand = self._calc_demand(self.predicted_requests)
        old_trips = self._calc_trips_amount(old_demand, old_supply)
        old_trips_sufficient_battery = old_trips * self._probability_sufficient_battery(self.t)
        rev_increase = (new_trips_sufficient_battery - old_trips_sufficient_battery) * self.p
        return rev_increase

    def _calc_revenue_decrease(self, delta_T):
        supply = self._calc_supply(self.available_scooters, self.t)
        demand = self._calc_demand(self.requests)
        total_trips = self._calc_trips_amount(demand, supply)
        no_trips = delta_T + total_trips
        delta_p = self.p - self._calc_inverse_demand(no_trips, demand)
        rev_dec = total_trips * delta_p + delta_T * delta_p - delta_T * self.p
        return rev_dec

    def _calc_next_supply_demand_df(self, transition_matrix, supply_demand, t):
        demand = self._calc_demand(supply_demand[:, 1])
        supply = self._calc_supply(supply_demand[:, 3], t)
        total_trips_t = self._calc_trips_amount(demand, supply)
        added_scooters = np.sum(transition_matrix * total_trips_t[:, None], axis=1)
        unused_scooters = supply - total_trips_t
        self.next_available_scooters = added_scooters + unused_scooters
        new_supply_demand = np.column_stack((supply_demand[:, 2],  # predicted_requests
                                             added_scooters + unused_scooters,
                                             supply_demand[:, 0]))  # region

        return new_supply_demand

    def _optimize_objective(self, x):
        n = len(x) // 2
        delta_T, delta_V = x[:n], x[n:]
        return -(sum(self._calc_revenue_increase(delta_V)) - sum(self._calc_revenue_decrease(delta_T)))

    def _estimate_gradient(self, x):
        epsilon = np.sqrt(np.finfo(float).eps)
        return approx_fprime(x, self._optimize_objective, epsilon)

    def _random(self, x0, max_iterations=1000):
        x = np.copy(x0)
        best_value = self._optimize_objective(x)
        print(best_value)
        for _ in tqdm(range(max_iterations)):
            random_x = np.random.randint(1, 6, len(x))
            new_value = self._optimize_objective(random_x)
            if new_value > best_value:
                best_value = new_value
                x = random_x

        return x

    def _gradient_descent(self, x0, learning_rate=1e-3, max_iterations=100, tolerance=1e-6):
        x = np.copy(x0)
        for _ in tqdm(range(max_iterations)):
            grad = self._estimate_gradient(x)
            new_x = x - learning_rate * grad
            if np.linalg.norm(new_x - x) < tolerance:
                break
            x = new_x
        return x

    def _optimize_solve(self):
        n = self.transition_matrix.shape[0]
        x0 = np.random.randint(1, 6, 2 * n)

        optimized_x = self._random(x0)
        optimal_delta_T = optimized_x[:n]
        optimal_delta_V = optimized_x[n:]
        return optimal_delta_T, optimal_delta_V

    def _load_transition_matrix(self):
        transition_matrix_df = pd.read_csv('transition_matrix.csv')
        transition_matrix = transition_matrix_df.iloc[:, 1:].values
        return transition_matrix

    def _calc_optimal_price(self, delta_T):
        delta_p = self.p - self._calc_inverse_demand(self._calc_demand(self.requests) - delta_T, self._calc_demand(self.requests))
        optimal_price = np.full(len(self.p), 0)
        for i in range(len(self.p)):
            l_optimal_price = self.p[i] - delta_p[i]
            #print("I is ", i, " self.p: ", self.p[i], " delta: ", delta_p[i], " l_optimal" , l_optimal_price)
            if l_optimal_price <= 0:
                l_optimal_price = self.p[i]
            optimal_price[i] = l_optimal_price

        return optimal_price

    def p_pricing(self, t, supply_demand):
        supply_demand = np.nan_to_num(supply_demand)
        transition_matrix = self._load_transition_matrix()

        self.total_available_scooters = np.sum(supply_demand[:, 3])

        self.t = t
        self.available_scooters = supply_demand[:, 3]
        self.requests = supply_demand[:, 1]
        self.predicted_requests = supply_demand[:, 2]
        self.transition_matrix = transition_matrix

        self._calc_next_supply_demand_df(transition_matrix, supply_demand, t)

        optimal_delta_T, optimal_delta_V = self._optimize_solve()
        return self._calc_optimal_price(optimal_delta_T)


