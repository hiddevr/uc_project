import pandas

# Baseline

# read the trips
df = pandas.read_csv('../Simulation/trips.csv')
# Remove the data which we can't use


duration = 0
for index, row in df.iterrows():
    duration += (row["Trip Duration"] / 60)

#print("Start area: ", start_area_dict)
#print("End area: ", end_area_dict)
#print("Pairs: ", pairs)

print(f"There is a total of {duration} minutes spend on an E-Scooter. This results in an average "
      f"price of ${(len(df) + (duration * 0.42)) / len(df)} per trip. This is a total revenue of ${len(df) + (duration * 0.42)}")