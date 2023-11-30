import itertools

import simpy
import random

"""
Parameters:
- Distance (in km)
- Raining
- Price
- Scooter available
"""

areas = [28, 22, 24, 31, 7, 21, 27, 16, 17, 15, 18, 19, 5, 8, 23, 25, 29, 20, 30, 36, 32, 60]
price_areas = {}
RANDOM_SEED = 42

revenue = 0
trips = 0


class BikeHub:
    def __init__(self, env, num_bikes):
        self.env = env
        self.num_bikes = simpy.Resource(env, num_bikes)

    def get_bike(self, time):
        yield self.env.timeout(time)

# TODO
# Change this to the P-pricing algorithm
def get_price(start_area):
    standard_price_per_minute = 0.32
    return standard_price_per_minute * price_areas[start_area]


def customer(env, name, hub):
    global revenue
    global trips
    # Maybe also random date, time etc?
    is_raining = bool(random.getrandbits(1))
    travel_time_minutes = random.randrange(0, 30)
    price = get_price(random.choice(areas))
    print(f'Customer {name} arrives at the hub. The price based on P-pricing is {price}')

    take_bike = 1 + (-0.3 * is_raining) + (-0.02 * travel_time_minutes) + (-0.1 * price)
    if take_bike < 0.6:
        print(f'Customer {name} did not take a bike.')
        return

    with hub.num_bikes.request() as req:
        patience = random.uniform(0, 10)
        results = yield req | env.timeout(patience)

        if req in results:
            print(f'Customer {name} got a bike!')
            yield env.process(hub.get_bike(travel_time_minutes))
            revenue += (travel_time_minutes * price) + 1
            trips += 1
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
    env.process(setup(env, 10))

    env.run(until=400)

    print(f"Total revenue is ${revenue}, n of trips: {trips}, avg per trip is ${revenue / trips}")



# No solution in the in the research question
#