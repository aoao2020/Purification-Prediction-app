import joblib

from pathlib import Path

for variant, pattern in [
    ("product-based", "*product_based*.pkl"),
    ("reaction-based", "*reaction_based*.pkl"),
]:

    print(f"\n=== {variant} ===")

    for model_path in Path("models").glob(pattern):

        print("\n" + "=" * 80)

        print(model_path.name)

        model = joblib.load(model_path)

        print("type:", type(model))

        if hasattr(model, "n_features_in_"):

            print("n_features_in_:", model.n_features_in_)

        if hasattr(model, "feature_name_"):

            print("feature_name_ length:", len(model.feature_name_))

            print("first 20:", model.feature_name_[:20])

            print("last 20:", model.feature_name_[-20:])

        if hasattr(model, "classes_"):

            print("classes:", model.classes_)
