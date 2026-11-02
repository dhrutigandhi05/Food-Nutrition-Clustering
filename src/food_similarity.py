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

def search_foods(query):
    matches = df[
        df["Food_Description_EN"].str.contains(query, case=False, regex=False, na=False)
    ]

    return matches[["Food_Code", "Food_Description_EN"]]

def find_similar_foods(food_code, n=5):
    matches = df.index[df["Food_Code"] == food_code]

    if len(matches) == 0:
        return pd.DataFrame()

    food_index = matches[0]

    distances, indices = model.kneighbors(
        X_scaled[food_index].reshape(1, -1),
        n_neighbors=min(n + 1, len(df))
    )

    results = []

    for distance, index in zip(distances[0], indices[0]):
        if index == food_index:
            continue

        results.append({
            "Food": df.iloc[index]["Food_Description_EN"],
            "Protein": df.iloc[index]["Protein"],
            "Fat": df.iloc[index]["Fat"],
            "Carbs": df.iloc[index]["Carbs"],
            "Fibre": df.iloc[index]["Fibre"],
            "Calories": df.iloc[index]["Calories"],
            "Distance": round(distance, 3)
        })

        if len(results) == n:
            break

    return pd.DataFrame(results)

if __name__ == "__main__":
    query = input("Search for a food: ")
    matches = search_foods(query)

    if matches.empty:
        print("No matching foods found.")
    else:
        print("\nMatching foods:")
        print(matches.head(20).to_string(index=False))

        try:
            food_code = int(input("\nEnter Food Code: "))
            results = find_similar_foods(food_code)

            if results.empty:
                print("Invalid food code.")
            else:
                print("Most similar foods:")
                print(results.to_string(index=False))

        except ValueError:
            print("Please enter a valid food code.")