import data as data
import pandas as pd
import numpy as np

import json

from matplotlib import pyplot as plt

data_file = "carbon-tracker.json"

#Data Preprocessing

date_key = "date"
bike_key = "bike"
bus_key = "bus"
car_key = "car"
metro_key = "metro"
tram_key = "tram"
trolley_key = "trolley"
food_key = "food"

bike_effect = 0
bus_effect = 3090
car_effect = 23850
metro_effect = 3600
tram_effect = 455
trolley_effect = 1860
low_meat_effect = 2200.0
vegetarian_effect = 1700.0
vegan_effect = 1500.0
high_meat_effect = 3300.0


def convert_date_to_day(date_string):
    match date_string:
        case "15-04-2024":
            return 1
        case "16-04-2024":
            return 2
        case "17-04-2024":
            return 3
        case "18-04-2024":
            return 4
        case "19-04-2024":
            return 5
        case "20-04-2024":
            return 6
        case "21-04-2024":
            return 7
        case "22-04-2024":
            return 8
        case "23-04-2024":
            return 9
        case "24-04-2024":
            return 10
        case "25-04-2024":
            return 11
        case "26-04-2024":
            return 12
        case "27-04-2024":
            return 13
        case "28-04-2024":
            return 14
        case "29-04-2024":
            return 15
        case "30-04-2024":
            return 16
        case "01-05-2024":
            return 17
        case "02-05-2024":
            return 18
        case "03-05-2024":
            return 19
        case "04-05-2024":
            return 20
        case "05-05-2024":
            return 21


def get_carbon_value(bike, bus, car, metro, tram, trolley, food):
    return float(bike) * bike_effect + \
           float(bus) * bus_effect + \
           float(car) * car_effect + \
           float(metro) * metro_effect + \
           float(tram) * tram_effect + \
           float(trolley) * trolley_effect + \
           get_food_effect(food)


def get_transport_carbon_value(bike, bus, car, metro, tram, trolley):
    return float(bike) * bike_effect + \
           float(bus) * bus_effect + \
           float(car) * car_effect + \
           float(metro) * metro_effect + \
           float(tram) * tram_effect + \
           float(trolley) * trolley_effect


def get_food_effect(food: str) -> float:
    if food == "Vegan":
        return vegan_effect
    elif food == "Vegetarian":
        return vegetarian_effect
    elif food == "LowMeat":
        return low_meat_effect
    else:
        return high_meat_effect


days_list = []
co2_list = []
user_id_list = []
food_co2_list = []
transport_co2_list = []

with open(data_file) as jsonFile:
    data = json.load(jsonFile)
    user_key = 1
    gen_index = 0

    for user in data:
        id_list = np.full(
            shape=420,
            fill_value=False,
            dtype=np.bool_
        )
        for date in data[user]:
            date_val = data[user][date][date_key]
            bike_val = data[user][date][bike_key]
            bus_val = data[user][date][bus_key]
            car_val = data[user][date][car_key]
            metro_val = data[user][date][metro_key]
            tram_val = data[user][date][tram_key]
            trolley_val = data[user][date][trolley_key]
            food_val = data[user][date][food_key]

            days_list.append(convert_date_to_day(date_val))
            co2_list.append(
                get_carbon_value(float(bike_val), float(bus_val),
                                 float(car_val), float(metro_val),
                                 float(tram_val), float(trolley_val),
                                 get_food_effect(food_val))
            )
            food_co2_list.append(get_food_effect(food_val))
            transport_co2_list.append(get_transport_carbon_value(float(bike_val), float(bus_val),
                                 float(car_val), float(metro_val),
                                 float(tram_val), float(trolley_val)))


            id_list[gen_index] = True
            gen_index += 1

        user_id_list.append(("ID_" + str(user_key), id_list))
        user_key += 1

initial_data = {
    'days': days_list,  # New days for prediction
    'co2_total': co2_list
}

column_list = ['days', 'co2_total']

for ff in user_id_list:
    initial_data[ff[0]] = ff[1]
    column_list.append(ff[0])

df = pd.DataFrame(
    initial_data, columns=column_list
)

def daily_average_co2(data):
    results = []
    for user, dates in data.items():
        for date, activities in dates.items():
            day = convert_date_to_day(date)
            if day == 0:
                print(f"Date {date} not mapped correctly") 
            food_co2 = get_food_effect(activities['food'])
            transport_co2 = get_transport_carbon_value(
                float(activities['bike']),
                float(activities['bus']),
                float(activities['car']),
                float(activities['metro']),
                float(activities['tram']),
                float(activities['trolley'])
            )
            results.append({
                'day': day,
                'food_co2': food_co2,
                'transport_co2': transport_co2
            })

    df = pd.DataFrame(results)
    if 'day' not in df:
        print("Day column missing in DataFrame")  
    daily_avg = df.groupby('day').mean().reset_index()
    return daily_avg

daily_avg = daily_average_co2(data)
if 'day' in daily_avg.columns:
    print("Day column exists, proceeding with plotting.")
else:
    print("Day column not found, check data conversion logic.")

if 'day' in daily_avg.columns:
    plt.figure(figsize=(10, 6))
    plt.bar(daily_avg['day'], daily_avg['food_co2'], color='blue')
    plt.title('Daily Average CO2 Emissions from Food')
    plt.xlabel('Day')
    plt.ylabel('CO2 Emissions (grams)')
    plt.savefig('Daily_Food_CO2_Emissions.png')
    plt.show()

    # Plotting Overall Daily CO2 Emissions
    daily_avg['total_co2'] = daily_avg['food_co2'] + daily_avg['transport_co2']
    plt.figure(figsize=(10, 6))
    plt.bar(daily_avg['day'], daily_avg['total_co2'], color='blue')
    plt.title('Overall Daily Average CO2 Emissions')
    plt.xlabel('Day')
    plt.ylabel('Total CO2 Emissions (grams)')
    plt.savefig('Overall_Daily_CO2_Emissions.png')
    plt.show()



    # Plotting Transport CO2
    plt.figure(figsize=(10, 6))
    plt.bar(daily_avg['day'], daily_avg['transport_co2'], color='blue')
    plt.title('Daily Average CO2 Emissions from Transport')
    plt.xlabel('Day')
    plt.ylabel('CO2 Emissions (grams)')
    plt.savefig('Daily_Transport_CO2_Emissions.png')
    plt.show()
