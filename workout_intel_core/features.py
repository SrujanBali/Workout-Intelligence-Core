import numpy as np
import pandas as pd
from workout_intel_core.dataset import load_raw_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


def exercise_feature_matrix(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = load_raw_data()

    features_df = df[["Exercise_Name", "muscle_gp", 'Equipment']].copy()
    features_df = features_df.dropna(subset=["Exercise_Name", 'muscle_gp'])
    features_df = features_df.drop_duplicates(subset=["Exercise_Name"])

    features_df = features_df.reset_index(drop=True)

    features_df["Equipment"] = features_df["Equipment"].fillna("Other")

    encoded_features = pd.get_dummies(
        features_df[["Equipment", "muscle_gp"]], prefix=["equip", "muscle"] 
    )

    matrix_df = pd.concat(
        [features_df[["Exercise_Name"]].reset_index(drop=True), encoded_features], axis=1
    )
    return matrix_df

def compute_similarity_matrix(matrix_df: pd.DataFrame) -> pd.DataFrame:
    features_values = matrix_df.drop(columns=["Exercise_Name"]).fillna(0)

    sim_array = cosine_similarity(features_values)

    sim_df = pd.DataFrame(sim_array, index=matrix_df["Exercise_Name"], columns=matrix_df["Exercise_Name"])

    return sim_df

def recommend_exercise(exercise_name: str, similarity_df: pd.DataFrame, top_n: int=5) -> pd.DataFrame:
    if exercise_name not in similarity_df.index:
        return f"Exercise {exercise_name} not found in the dataset."
    recommendations = similarity_df[exercise_name].sort_values(ascending=False)

    recommendations = recommendations[recommendations.index != exercise_name]

    recommendations = recommendations[recommendations >= 0.8]

    return recommendations.iloc[1 : top_n + 1]

if __name__ == "__main__":
    print("Loading data and building feature matirx...")
    matrix = exercise_feature_matrix()
    print(f"Feature matrix shape: {matrix.shape}")
    print("Computing similarity matrix")
    similarity_df = compute_similarity_matrix(matrix)
    print("Sample similarity scores for the first exercise: ")
    print(similarity_df.iloc[0].head())