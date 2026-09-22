import pandas as pd
import os 
from workout_intel_core.config import RAW_DATA_DIR

def load_raw_data(filename="Gym_Exercises_Dataset.csv"):
    filepath = RAW_DATA_DIR / filename

    data = pd.read_csv(filepath)

    return data