import pandas as pd

CSV_PATH = "Crop_recommendationV2.csv"
df = pd.read_csv(CSV_PATH)

# Soil ID mapping from CSV → actual name
CSV_SOIL_MAP = {
    0: "Alluvial soil",
    1: "Black Soil",
    2: "Red soil",
    3: "Clay soil"
}

# Reverse mapping: model output → CSV soil ID
MODEL_TO_CSV_MAP = {
    "Alluvial soil": 0,
    "Black Soil": 1,
    "Red soil": 2,
    "Clay soil": 3
}

def get_soil_properties_and_crops(soil_name):
    soil_name = soil_name.strip()

    if soil_name not in MODEL_TO_CSV_MAP:
        print("❌ Unknown soil name:", soil_name)
        return None

    soil_id = MODEL_TO_CSV_MAP[soil_name]

    # Filter rows where soil_type matches the numeric id
    rows = df[df["soil_type"] == soil_id]

    if rows.empty:
        print("❌ No rows found for soil type ID:", soil_id)
        return None

    avg_props = {
        "N": round(rows["N"].mean(), 2),
        "P": round(rows["P"].mean(), 2),
        "K": round(rows["K"].mean(), 2),
        "Temperature": round(rows["temperature"].mean(), 2),
        "Humidity": round(rows["humidity"].mean(), 2),
        "pH Level": round(rows["ph"].mean(), 2),
        "Rainfall": round(rows["rainfall"].mean(), 2)
    }

    # Recommended crops
    crops = sorted(rows["label"].unique().tolist())

    return {
        "avg": avg_props,
        "crops": crops
    }
