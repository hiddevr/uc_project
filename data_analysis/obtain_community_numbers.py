import pandas

# Baseline

# read the trips
df = pandas.read_csv('../Simulation/trips.csv')
# Remove the data which we can't use
df.dropna(inplace=True)
df.sort_values(by="Start Time", inplace=True)
start_area_dict = {}
end_area_dict = {}
pairs = {}
duration = 0
areas = set()
for index, row in df.iterrows():
    start_area = row["Start Community Area Number"]
    areas.add(start_area)
    end_area = row["End Community Area Number"]
    pair = (start_area, end_area)

    if start_area in start_area_dict:
        start_area_dict[start_area] += 1
    else:
        start_area_dict[start_area] = 1

    if end_area in end_area_dict:
        end_area_dict[end_area] += 1
    else:
        end_area_dict[end_area] = 1

    if pair in pairs:
        pairs[pair] += 1
    else:
        pairs[pair] = 1

    duration += (row["Trip Duration"] / 60)

#print("Start area: ", start_area_dict)
#print("End area: ", end_area_dict)
#print("Pairs: ", pairs)
print(areas)
for area in areas:
    start_area_dict[area] = start_area_dict[area] / len(df)
#for i in range(0, len(start_area_dict)):
    #start_area_dict[i] = start_area_dict[i] / len(df)
print(start_area_dict)
#print(f"There is a total of {duration} minutes spend on an E-Scooter. This results in an average "
      #f"price of ${(len(df) + (duration * 0.39)) / len(df)} per trip. This is a total revenue of ${len(df) + (duration * 0.39)}")