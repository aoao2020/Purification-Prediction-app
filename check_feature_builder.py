# check_feature_builder.py

from pathlib import Path
import joblib

from src.feature_builder import build_single_feature_dataframe


# =========================
# Reference lists
# =========================

UNIQUE_METHOD_LIST = [
    "GPC",
    "HPLC",
    "Ion Exchange Chromatography",
    "NH Silica Column",
    "Preparative TLC",
    "Reverse Phase Column",
    "Reverse Phase HPLC",
    "Silica Column",
    "TLC",
    "TLC (Alumina)",
    "TLC (Diol Silica)",
    "TLC (NH Silica)",
    "TLC (Reverse Phase)",
]

UNIQUE_SOLVENT_LIST = [
    "EtOAc/Hexane",
    "H2O/CH3CN",
    "MeOH/CH2Cl2",
    "MeOH/CHCl3",
    "MeOH/EtOAc",
]


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "features"

UNIQUE_AGENT_PATH = FEATURES_DIR / "unique_agent.pkl"


MODEL_TEST_CONFIG = {
    # -------------------------
    # simple models
    # -------------------------
    "simple_method": {
        "path": MODELS_DIR / "simple" / "method_LightGBM_morgan_rdkit_simple_model.pkl",
        "fea_type": "morgan_rdkit",
        "target_compound": "product",
        "method": "",
        "solvent": "",
        "use_agents": False,
    },
    "simple_solvent": {
        "path": MODELS_DIR / "simple" / "solvent_type_LightGBM_for_imbalance_morgan_method_simple_model.pkl",
        "fea_type": "morgan_method",
        "target_compound": "product",
        "method": "Silica Column",
        "solvent": "",
        "use_agents": False,
    },
    "simple_tlc_ratio": {
        "path": MODELS_DIR / "simple" / "tlc_solvent_ratio_LightGBM_rdkit_solvent_simple_model.pkl",
        "fea_type": "rdkit_solvent",
        "target_compound": "product",
        "method": "",
        "solvent": "EtOAc/Hexane",
        "use_agents": False,
    },
    "simple_silica_start": {
        "path": MODELS_DIR / "simple" / "silica_solvent_start_ratio_LightGBM_morgan_rdkit_solvent_simple_model.pkl",
        "fea_type": "morgan_rdkit_solvent",
        "target_compound": "product",
        "method": "",
        "solvent": "EtOAc/Hexane",
        "use_agents": False,
    },
    "simple_silica_end": {
        "path": MODELS_DIR / "simple" / "silica_solvent_end_ratio_LightGBM_rdkit_solvent_simple_model.pkl",
        "fea_type": "rdkit_solvent",
        "target_compound": "product",
        "method": "",
        "solvent": "EtOAc/Hexane",
        "use_agents": False,
    },

    # -------------------------
    # complex models
    # -------------------------
    "complex_method": {
        "path": MODELS_DIR / "complex" / "method_LightGBM_for_imbalance_morgan_agent_complex_model.pkl",
        "fea_type": "morgan_agent",
        "target_compound": "reactant_product",
        "method": "",
        "solvent": "",
        "use_agents": True,
    },
    "complex_solvent": {
        "path": MODELS_DIR / "complex" / "solvent_type_LightGBM_for_imbalance_rdkit_agent_method_complex_model.pkl",
        "fea_type": "rdkit_agent_method",
        "target_compound": "reactant_product",
        "method": "Silica Column",
        "solvent": "",
        "use_agents": True,
    },
    "complex_tlc_ratio": {
        "path": MODELS_DIR / "complex" / "tlc_solvent_ratio_LightGBM_morgan_rdkit_agent_solvent_complex_model.pkl",
        "fea_type": "morgan_rdkit_agent_solvent",
        "target_compound": "reactant_product",
        "method": "",
        "solvent": "EtOAc/Hexane",
        "use_agents": True,
    },
    "complex_silica_start": {
        "path": MODELS_DIR / "complex" / "silica_solvent_start_ratio_LightGBM_morgan_agent_solvent_complex_model.pkl",
        "fea_type": "morgan_agent_solvent",
        "target_compound": "reactant_product",
        "method": "",
        "solvent": "EtOAc/Hexane",
        "use_agents": True,
    },
    "complex_silica_end": {
        "path": MODELS_DIR / "complex" / "silica_solvent_end_ratio_LightGBM_rdkit_agent_solvent_complex_model.pkl",
        "fea_type": "rdkit_agent_solvent",
        "target_compound": "reactant_product",
        "method": "",
        "solvent": "EtOAc/Hexane",
        "use_agents": True,
    },
}


# =========================
# Test input
# =========================

TEST_PRODUCT_SMILES = "CCO"
TEST_REACTANT_SMILES = "CC=O"
TEST_AGENTS = "DMF, HCl"


def load_unique_agents():
    if not UNIQUE_AGENT_PATH.exists():
        raise FileNotFoundError(
            f"unique_agent.pkl was not found: {UNIQUE_AGENT_PATH}"
        )

    unique_agents = joblib.load(UNIQUE_AGENT_PATH)

    if not isinstance(unique_agents, list):
        unique_agents = list(unique_agents)

    return unique_agents


def get_model_n_features(model):
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)

    if hasattr(model, "feature_name_"):
        return len(model.feature_name_)

    raise AttributeError("Could not determine model feature dimension.")


def main():
    print("=" * 80)
    print("Check feature_builder.py")
    print("=" * 80)

    unique_agents = load_unique_agents()

    print(f"unique agents: {len(unique_agents)}")
    print(f"first 10 agents: {unique_agents[:10]}")
    print("=" * 80)

    all_ok = True

    for name, cfg in MODEL_TEST_CONFIG.items():
        print()
        print("=" * 80)
        print(name)
        print("=" * 80)

        model_path = cfg["path"]

        if not model_path.exists():
            print(f"[ERROR] model file not found: {model_path}")
            all_ok = False
            continue

        model = joblib.load(model_path)
        expected_n_features = get_model_n_features(model)

        ref_agents_list = unique_agents if cfg["use_agents"] else None

        try:
            X = build_single_feature_dataframe(
                product_smiles=TEST_PRODUCT_SMILES,
                reactant_smiles=TEST_REACTANT_SMILES,
                agents=TEST_AGENTS,
                method=cfg["method"],
                solvent=cfg["solvent"],
                fp_type="count_morgan",
                ref_agents_list=ref_agents_list,
                ref_method_list=UNIQUE_METHOD_LIST,
                ref_solvent_list=UNIQUE_SOLVENT_LIST,
                fp_dim=1024,
                fea_type=cfg["fea_type"],
                target_compound=cfg["target_compound"],
            )

            actual_n_features = X.shape[1]

            print(f"fea_type         : {cfg['fea_type']}")
            print(f"target_compound  : {cfg['target_compound']}")
            print(f"expected features: {expected_n_features}")
            print(f"actual features  : {actual_n_features}")

            if actual_n_features != expected_n_features:
                print("[NG] feature dimension mismatch")
                all_ok = False
                continue

            pred = model.predict(X)
            print(f"prediction       : {pred}")
            print("[OK] feature dimension matched and prediction succeeded")

        except Exception as e:
            print("[ERROR] failed")
            print(type(e).__name__, e)
            all_ok = False

    print()
    print("=" * 80)

    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")

    print("=" * 80)


if __name__ == "__main__":
    main()