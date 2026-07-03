import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/train_aWnotuB.csv")

# Convert DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract Hour
df['Hour'] = df['DateTime'].dt.hour

# Average traffic per hour
hourly = df.groupby('Hour')['Vehicles'].mean()

# Plot
plt.figure(figsize=(10,5))
plt.plot(hourly.index, hourly.values, marker='o')

plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Vehicles")
plt.grid(True)

plt.show()