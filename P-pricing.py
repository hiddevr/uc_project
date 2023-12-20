import pickle
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import numpy as np
from scipy.optimize import minimize_scalar


class PPricing:

    def __init__(self):
        self.max_battery_distance = 1090013.4
        self.alpha = 1
        self.beta = 0.1
        self.p = 0.45
        self.price_bounds = (0.2, 1)

    def _probability_passenger_willing_to_pay(self, p=0.45):
        """
        Function for calculating probability of passenger willing to pay price p
        :param p: Price
        :param alpha: Scaling to adjust base demand probability
        :param beta: Rate at which demand decreases as price increases
        :return: Probability of passenger willing to pay price p
        """
        probability = self.alpha * np.exp(-self.beta)
        return probability

    def _probability_sufficient_battery(self, t):
        """
        Function for calculating if an e-scooter has sufficient battery for a trip.
        :param t: current time period
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

    def _expected_revenue(self, p, demand, supply):
        expected_demand = demand * self._probability_passenger_willing_to_pay(p)
        expected_supply = supply * self._probability_sufficient_battery(p)
        return p * min(expected_demand, expected_supply)

    def _negative_expected_revenue(self, p, demand, supply):
        return -self._expected_revenue(p, demand, supply)

    def _calc_local_optimal(self, t, demand, supply):
        result = minimize_scalar(
            self._negative_expected_revenue,
            bounds=self.price_bounds,
            args=(demand, supply),
            method='bounded'
        )
        if result.success:
            optimal_price = result.x
            max_expected_revenue = -result.fun
            return optimal_price, max_expected_revenue
        else:
            print('No optimal price')

    def _calc_revenue_decrease(self, demand, optimal_price):
        trips_optimal_price = self._probability_passenger_willing_to_pay(optimal_price) * demand
        trips_default_price = self._probability_passenger_willing_to_pay(self.p) * demand
        delta_trips = abs(trips_optimal_price - trips_default_price)
        delta_p = abs(optimal_price - self.p)
        return trips_optimal_price * optimal_price - (trips_optimal_price + delta_trips) * (optimal_price - delta_p)

    def _calc_revenue_increase(self, supply, new_price, t, added_trips):
        actual_supply = self._probability_sufficient_battery(t) * supply
        additional_trips = min(added_trips, actual_supply)
        return additional_trips * self._probability_passenger_willing_to_pay(new_price) * new_price

    def _predict_demand(self, supply_demand_df):
        return 0

    def p_pricing(self, t, supply_demand_df):
        # Assumes df with columns 'requests', 'available_scooters', 'region' for a single time step
        no_regions = supply_demand_df['region'].nunique()
        for index, row in supply_demand_df.iterrows():
            optimal_price = self._calc_local_optimal(t, row['requests'], row['requests'])
            revenue_decrease = self._calc_revenue_decrease()

        predicted_demand = self._predict_demand(supply_demand_df)
        pass


R = [100, 150, 200]  # Requests
V = [50, 60, 70]  # Available e-scooters
R_next = [120, 140, 210]  # Predicted demand

optimal_prices = p_pricing(R, V, R_next)
