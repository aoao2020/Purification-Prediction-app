# check_predictor.py

from pathlib import Path

from src.predictor import PurificationPredictor


def print_probability_table(title, df, top_n=5):
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)

    show_df = df.head(top_n).copy()

    for _, row in show_df.iterrows():
        print(f"{row['candidate']}: {float(row['prob']):.4f}")


def print_result(mode, result):
    print()
    print("=" * 80)
    print(f"MODE: {mode}")
    print("=" * 80)

    print(f"Predicted method: {result['method']}")
    print(f"Top solvent     : {result['top_solvent']}")

    print_probability_table(
        "Method probabilities",
        result["method_prob"],
        top_n=5,
    )

    print_probability_table(
        "Solvent probabilities",
        result["solvent_prob"],
        top_n=7,
    )

    print()
    print("-" * 80)
    print("Top solvent candidates and ratios")
    print("-" * 80)

    for i, item in enumerate(result["candidate"], start=1):
        solvent = item["solvent"]
        prob = item["prob"]
        ratio = item["ratio"]

        print(f"\nCandidate {i}: {solvent} ({prob:.4f})")
        print(f"  TLC ratio          : {ratio['tlc']:.2f}")
        print(f"  Silica start ratio : {ratio['silica_start']:.2f}")
        print(f"  Silica end ratio   : {ratio['silica_end']:.2f}")


def main():
    root_dir = Path(__file__).resolve().parent

    predictor = PurificationPredictor(root_dir=root_dir)

    # -------------------------
    # Simple mode test
    # -------------------------
    simple_result = predictor.predict_all(
        mode="simple",
        product_smiles="CCO",
    )

    print_result(
        mode="simple",
        result=simple_result,
    )

    # -------------------------
    # Complex mode test
    # -------------------------
    complex_result = predictor.predict_all(
        mode="complex",
        reactant_smiles="CC=O",
        product_smiles="CCO",
        agents="DMF, HCl",
    )

    print_result(
        mode="complex",
        result=complex_result,
    )

    print()
    print("=" * 80)
    print("ALL PREDICTOR CHECKS FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()