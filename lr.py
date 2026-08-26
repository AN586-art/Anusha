from sklearn.linear_model import LinearRegression

# Training data
distance = [[10], [20], [30], [40], [50]]
energy = [2, 4, 6, 8, 10]

# Create ML model
model = LinearRegression()

# Train the model
model.fit(distance, energy)

# Predict energy consumption for 60 km
prediction = model.predict([[60]])

print("Distance:", 60, "km")
print("Predicted Energy Consumption:", prediction[0], "kWh")
