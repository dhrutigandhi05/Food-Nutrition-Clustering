import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("data/processed/foods_cleaned.csv")
features = ["Protein", "Fat", "Carbs", "Fibre"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

model = NearestNeighbors(
    n_neighbors=6,
    metric="euclidean"
)

model.fit(X_scaled)

def find_similar_foods(food_name, n=5):
    matches = df[
        df["Food_Description_EN"].str.contains(
            food_name, case=False, regex=False, na=False
        )
    ]

    if matches.empty:
        return "Food not found."

    food_index = matches.index[0]

    distances, indices = model.kneighbors(
        X_scaled[food_index].reshape(1, -1),
        n_neighbors=n + 1
    )

    similar_indices = [i for i in indices[0] if i != food_index][:n]

    return df.loc[
        similar_indices,
        ["Food_Description_EN"] + features + ["Calories"]
    ]

if __name__ == "__main__":
    print(find_similar_foods("Chicken, broiler"))