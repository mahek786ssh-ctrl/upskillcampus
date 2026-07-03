import pandas as pd

# Load dataset
df = pd.read_csv("data/train_aWnotuB.csv")

# Convert DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Create new features
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# Weekend feature
df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

print(df.head())

print("\nNew Columns:")
print(df.columns.tolist())