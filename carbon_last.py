import data as data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_validate, KFold
import json

from sklearn.preprocessing import PolynomialFeatures

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


def get_tarnsport_carbon_value(bike, bus, car, metro, tram, trolley):
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
            transport_co2_list.append(get_tarnsport_carbon_value(float(bike_val), float(bus_val),
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

#Algoithm

X = df.drop('co2_total', axis=1)
y = df['co2_total']
#Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44)
#Linear Regression

#lr = LinearRegression()

#lr.fit(X_train, y_train)

#predict = lr.predict(X_test)

#Gradient Boosting

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

predict=gb.predict(X_test)

#Random Forest Regression
#rf = RandomForestRegressor()  # You can tune these parameters
#rf.fit(X_train, y_train)
#predict=rf.predict(X_test)

#Scoring
scoring = ['r2', 'neg_root_mean_squared_error']
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(gb, X, y, cv=kf, scoring=scoring, return_train_score=True)

# Extract R^2 and MSE scores
r2_scores = cv_results['test_r2']
rmse_scores = -cv_results['test_neg_root_mean_squared_error']  # Make RMSE positive

print("Mean R^2 score:", r2_scores.mean())
print("Mean RMSE score:", rmse_scores.mean())

def collect_data_and_predict():
    # Ask for input data
    bike = float(input("Enter bike usage in hours: "))
    bus = float(input("Enter bus usage in hours: "))
    car = float(input("Enter car usage in hours: "))
    metro = float(input("Enter metro usage in hours: "))
    tram = float(input("Enter tram usage in hours: "))
    trolley = float(input("Enter trolley usage in hours: "))
    food = input("Enter food choice (Vegan, Vegetarian, LowMeat, HighMeat): ")
    return get_carbon_value(bike, bus, car, metro, tram, trolley, food)

new_day_data = collect_data_and_predict()
input_info = pd.DataFrame({
    'days': [1],  # New days for prediction
    'ID_21': ['1'],  # New unique IDs for which you need predictions
    'co2_total': [new_day_data]
})

df = pd.concat([df, input_info], ignore_index=True)

df = df.fillna(0)

X_train = df.drop('co2_total', axis=1)
y_train = df['co2_total']

new_data = pd.DataFrame({
    'days': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  # New days for prediction
    'ID': ['21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21', '21']  # New unique IDs for which you need predictions
})

new_data = pd.get_dummies(new_data, columns=['ID'])

for col in X_train.columns:
    if col not in new_data.columns:
        new_data[col] = 0

new_data = new_data[X_train.columns]

gb.fit(X_train, y_train)

predicted_co2 = gb.predict(new_data)

all_days = np.arange(1, 22)
all_co2 = np.concatenate([[new_day_data], predicted_co2])
all_co2_kg = all_co2 / 1000

plt.figure(figsize=(12, 8))
plt.bar(all_days, all_co2_kg, color='teal')
plt.title('Daily CO2 Emissions Over 21 Days')
plt.xlabel('Day')
plt.ylabel('CO2 Emissions(kg)')
plt.xticks(all_days)  # Ensure all days are marked
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
