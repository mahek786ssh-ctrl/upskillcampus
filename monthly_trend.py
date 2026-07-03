import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/train_aWnotuB.csv")

# Convert DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract Month
df['Month'] = df['DateTime'].dt.month

# Average vehicles per month
monthly = df.groupby('Month')['Vehicles'].mean()

plt.figure(figsize=(10,5))
plt.plot(monthly.index, monthly.values, marker='o')

plt.title("Average Traffic Volume by Month")
plt.xlabel("Month")
plt.ylabel("Average Vehicles")
plt.grid(True)

plt.show()