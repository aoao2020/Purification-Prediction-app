# src/predictor.py

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.feature_builder import (
    build_single_feature_dataframe,
)


# =====================================================
# Constants
# =====================================================

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
    "H$_{2}$O/CH$_{3}$CN",
    "MeOH/CH$_{2}$Cl$_{2}$",
    "MeOH/CHCl$_{3}$",
    "MeOH/EtOAc",
]

METHOD_FEATURE_MAP = {
    "nh silica": "NH Silica Column",
    "reverse phase": "Reverse Phase Column",
    "silica": "Silica Column",
}

# The replacement models were trained with 2048-bit Morgan fingerprints.
MORGAN_FP_DIM = 2048

# The reaction-based models contain 413 agent one-hot columns.  Limit the
# existing reference list to the feature width expected by those models.
REACTION_AGENT_FEATURE_COUNT = 413

# Ratio models were trained on the two aggregated method labels.  Only the
# first column (Silica) is used by the current silica-ratio models.
RATIO_METHOD_LIST = ["Silica", "Other"]


# =====================================================
# Predictor
# =====================================================

class PurificationPredictor:

    def __init__(
        self,
        root_dir: Path,
    ):

        self.root_dir = Path(root_dir)

        self.models_dir = self.root_dir / "models"
        self.features_dir = self.root_dir / "features"

        self.unique_agents = joblib.load(
            self.features_dir / "unique_agent.pkl"
        )
        self.model_agents = list(self.unique_agents)[:REACTION_AGENT_FEATURE_COUNT]

        self.models = {}

        self._load_models()

    # =================================================
    # Load models
    # =================================================

    def _load_models(self):

        self.models["product-based"] = {

            "method":
            joblib.load(
                self.models_dir
                / "method_LightGBM_morgan_product_based_model.pkl"
            ),

            "solvent":
            joblib.load(
                self.models_dir
                / "solvent_type_LightGBM_morgan_rdkit_method_product_based_smote_model.pkl"
            ),

            "tlc":
            joblib.load(
                self.models_dir
                / "tlc_solvent_ratio_LightGBM_morgan_rdkit_solvent_method_product_based_model.pkl"
            ),

            "silica_start":
            joblib.load(
                self.models_dir
                / "silica_solvent_initial_ratio_LightGBM_morgan_solvent_method_product_based_model.pkl"
            ),

            "silica_end":
            joblib.load(
                self.models_dir
                / "silica_solvent_final_ratio_LightGBM_rdkit_solvent_method_product_based_model.pkl"
            ),

        }

        self.models["reaction-based"] = {

            "method":
            joblib.load(
                self.models_dir
                / "method_LightGBM_morgan_agent_reaction_based_smote_model.pkl"
            ),

            "solvent":
            joblib.load(
                self.models_dir
                / "solvent_type_LightGBM_morgan_agent_method_reaction_based_smote_model.pkl"
            ),

            "tlc":
            joblib.load(
                self.models_dir
                / "tlc_solvent_ratio_LightGBM_morgan_rdkit_agent_solvent_method_reaction_based_model.pkl"
            ),

            "silica_start":
            joblib.load(
                self.models_dir
                / "silica_solvent_initial_ratio_LightGBM_morgan_agent_solvent_method_reaction_based_model.pkl"
            ),

            "silica_end":
            joblib.load(
                self.models_dir
                / "silica_solvent_final_ratio_LightGBM_rdkit_agent_solvent_method_reaction_based_model.pkl"
            ),

        }

    # =================================================
    # Internal feature generation
    # =================================================

    def _build_X(
        self,
        mode,
        fea_type,
        target_compound,
        product_smiles,
        reactant_smiles="",
        agents="",
        method="",
        solvent="",
        ref_method_list=None,
    ):

        use_agent = "agent" in fea_type

        return build_single_feature_dataframe(

            product_smiles=product_smiles,

            reactant_smiles=reactant_smiles,

            agents=agents,

            method=method,

            solvent=solvent,

            fp_type="count_morgan",

            ref_agents_list=(
                self.model_agents
                if use_agent
                else None
            ),

            ref_method_list=ref_method_list or UNIQUE_METHOD_LIST,

            ref_solvent_list=UNIQUE_SOLVENT_LIST,

            fp_dim=MORGAN_FP_DIM,

            fea_type=fea_type,

            target_compound=target_compound,

        )

    # =================================================
    # Feature setting
    # =================================================

    def _setting(self, mode):

        if mode == "product-based":

            return {

                "target": "product",

                "method":
                "morgan",

                "solvent":
                "morgan_rdkit_method",

                "tlc":
                "morgan_rdkit_solvent_method",

                "silica_start":
                "morgan_solvent_method",

                "silica_end":
                "rdkit_solvent_method",

            }

        elif mode == "reaction-based":

            return {

                "target":
                "reactant_product",

                "method":
                "morgan_agent",

                "solvent":
                "morgan_agent_method",

                "tlc":
                "morgan_rdkit_agent_solvent_method",

                "silica_start":
                "morgan_agent_solvent_method",

                "silica_end":
                "rdkit_agent_solvent_method",

            }

        raise ValueError(mode)

    # =================================================
    # Method prediction
    # =================================================

    def predict_method(
        self,
        mode,
        product_smiles,
        reactant_smiles="",
        agents="",
    ):

        setting = self._setting(mode)

        model = self.models[mode]["method"]

        X = self._build_X(

            mode,

            setting["method"],

            setting["target"],

            product_smiles,

            reactant_smiles,

            agents,

        )

        pred = model.predict(X)[0]

        proba = model.predict_proba(X)[0]

        df = pd.DataFrame({

            "candidate":
            model.classes_,

            "prob":

            proba

        })

        df = df.sort_values(
            "prob",
            ascending=False,
        )

        return pred, df

    # =================================================
    # Solvent prediction
    # =================================================

    def predict_solvent(
        self,
        mode,
        method,
        product_smiles,
        reactant_smiles="",
        agents="",
    ):

        setting = self._setting(mode)

        model = self.models[mode]["solvent"]

        method_feature = METHOD_FEATURE_MAP.get(
            str(method).strip().lower(),
            method,
        )

        X = self._build_X(

            mode,

            setting["solvent"],

            setting["target"],

            product_smiles,

            reactant_smiles,

            agents,

            method=method_feature,

        )

        pred = model.predict(X)[0]

        proba = model.predict_proba(X)[0]

        df = pd.DataFrame({

            "candidate":
            model.classes_,

            "prob":
            proba,

        })

        df = df.sort_values(
            "prob",
            ascending=False,
        )

        return pred, df

    # =================================================
    # Ratio prediction
    # =================================================

    def predict_ratio(
        self,
        mode,
        solvent,
        product_smiles,
        reactant_smiles="",
        agents="",
        method="",
    ):

        setting = self._setting(mode)

        result = {}

        for task in [

            "tlc",

            "silica_start",

            "silica_end",

        ]:

            model = self.models[mode][task]

            X = self._build_X(

                mode,

                setting[task],

                setting["target"],

                product_smiles,

                reactant_smiles,

                agents,

                method=method,

                solvent=solvent,

                ref_method_list=RATIO_METHOD_LIST,

            )

            value = float(
                model.predict(X)[0]
            )

            value = max(
                0,
                min(
                    100,
                    value,
                )
            )

            result[task] = value

        return result

    # =================================================
    # Full pipeline
    # =================================================

    def predict_all(
        self,
        mode,
        product_smiles,
        reactant_smiles="",
        agents="",
    ):

        method_pred, method_df = \
            self.predict_method(

                mode,

                product_smiles,

                reactant_smiles,

                agents,

            )

        solvent_pred, solvent_df = \
            self.predict_solvent(

                mode,

                method_pred,

                product_smiles,

                reactant_smiles,

                agents,

            )

        top2 = solvent_df.head(2)

        solvent_result = []

        for _, row in top2.iterrows():

            solvent = row["candidate"]

            prob = float(
                row["prob"]
            )

            ratio = self.predict_ratio(

                mode,

                solvent,

                product_smiles,

                reactant_smiles,

                agents,

                method=method_pred,

            )

            solvent_result.append({

                "solvent":
                solvent,

                "prob":
                prob,

                "ratio":
                ratio,

            })

        return {

            "method":
            method_pred,

            "method_prob":
            method_df,

            "top_solvent":
            solvent_pred,

            "solvent_prob":
            solvent_df,

            "candidate":
            solvent_result,

        }
