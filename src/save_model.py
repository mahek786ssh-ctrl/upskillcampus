import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("data/train_aWnotuB.csv")

df['DateTime'] = pd.to_datetime(df['DateTime'])

df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

X = df[['Junction','Year','Month','Day','Hour','DayOfWeek','Weekend']]
y = df['Vehicles']

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "models/traffic_model.pkl")

print("Model Saved Successfully!")