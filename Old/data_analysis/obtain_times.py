import pandas
from datetime import datetime
# Baseline

# read the trips
df = pandas.read_csv('../Simulation/trips.csv')
# Remove the data which we can't use

trips_per_hour = {}
duration_dict = {}
for i in range(24):
    trips_per_hour[i] = 0
duration = 0
duration_times = set()

for index, row in df.iterrows():
    time = row["Start Time"]
    datetime_time = datetime.strptime(time, "%m/%d/%Y %I:%M:%S %p")
    trips_per_hour[datetime_time.hour] += 1
    dur_time = round(row["Trip Duration"] / 60)
    duration_times.add(dur_time)
    if dur_time not in duration_dict:
        duration_dict[dur_time] = 0
    else:
        duration_dict[dur_time] += 1
    duration += (row["Trip Duration"] / 60)

for i in range(24):
    trips_per_hour[i] = trips_per_hour[i] / len(df)
print(trips_per_hour)

times = []
weights = []
for dur in duration_times:
    weight = duration_dict[dur] / len(df)
    if dur <= 0 or weight == 0:
        continue
    times.append(dur)
    weights.append(weight)


print("-----------------")
print("times: ")
print(times)

print("-----------------------")
print("weights: ")
print(weights)