import joblib

from pathlib import Path

for mode in ["simple", "complex"]:

    print(f"\n=== {mode} ===")

    model_dir = Path("models") / mode

    for model_path in model_dir.glob("*.pkl"):

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