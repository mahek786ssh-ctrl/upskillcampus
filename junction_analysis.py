import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/train_aWnotuB.csv")

# Average vehicles by junction
junction_avg = df.groupby('Junction')['Vehicles'].mean()

plt.figure(figsize=(8,5))
junction_avg.plot(kind='bar')

plt.title("Average Traffic Volume by Junction")
plt.xlabel("Junction")
plt.ylabel("Average Vehicles")
plt.grid(axis='y')

plt.show()