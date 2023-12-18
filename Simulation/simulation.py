import itertools

import simpy
import random
from random import choices
from data_file import duration_trips, weights_durations, hours, weights_hours, areas, weights_areas, standard_price

"""
Parameters:
- Distance (in km)
- Raining
- Price
- Scooter available
"""



price_areas = {}
RANDOM_SEED = 42

revenue = 0
trips = 0
minutes = 0


class BikeHub:
    def __init__(self, env, num_bikes):
        self.env = env
        self.num_bikes = simpy.Resource(env, num_bikes)

    def get_bike(self, time):
        yield self.env.timeout(time)


# TODO
# Change this to the P-pricing algorithm
def get_price(start_area):
    standard_price_per_minute = 0.42
    return standard_price_per_minute * price_areas[start_area]


def customer(env, name, hub):
    global revenue
    global trips
    global minutes

    # TODO: make this a function argument, so it is easier
    # to test multiple 'willingness'-factors
    # 1 = pay for every price, 0 = pay only the standard price (0.42)
    willingness = 1
    price = get_price(choices(areas, weights_areas)[0])
    hour = choices(hours, weights_hours)[0]
    travel_time_minutes = choices(duration_trips, weights_durations)[0]
    proc_difference = ((price - standard_price) / standard_price) * 10
    print(
        f'Customer {name} arrives at the hub at {hour}. The price based on P-pricing is {price}, difference is {proc_difference}')

    take_bike = min(random.uniform(0, 5) * proc_difference, 1)
    if take_bike > willingness:
        print(f'Customer {name} did not take a bike, takebike is {take_bike}.')
        return

    with hub.num_bikes.request() as req:
        patience = random.uniform(0, 10)
        results = yield req | env.timeout(patience)

        if req in results:
            print(f'Customer {name} got a bike, takebike is {take_bike}')
            yield env.process(hub.get_bike(travel_time_minutes))
            revenue += (travel_time_minutes * price) + 1
            trips += 1
            minutes += travel_time_minutes
        else:
            print(f'Customer {name} did not get a bike, waiting time too long')


def setup(env, num_bikes):
    hub = BikeHub(env, num_bikes)
    count = itertools.count()
    while True:
        yield env.timeout(1)
        env.process(customer(env, f'{next(count)}', hub))


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    for area in areas:
        price_areas[area] = random.uniform(0.95, 1.05)

    env = simpy.Environment()
    env.process(setup(env, 10000))

    env.run(until=710839)

    print(f"Total revenue is ${revenue}, n of trips: {trips}, avg per trip is ${revenue / trips}, {minutes} minutes")

# No solution in the in the research question
#
