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
    "H2O/CH3CN",
    "MeOH/CH2Cl2",
    "MeOH/CHCl3",
    "MeOH/EtOAc",
]


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

        self.models = {}

        self._load_models()

    # =================================================
    # Load models
    # =================================================

    def _load_models(self):

        self.models["simple"] = {

            "method":
            joblib.load(
                self.models_dir
                / "simple"
                / "method_LightGBM_morgan_rdkit_simple_model.pkl"
            ),

            "solvent":
            joblib.load(
                self.models_dir
                / "simple"
                / "solvent_type_LightGBM_for_imbalance_morgan_method_simple_model.pkl"
            ),

            "tlc":
            joblib.load(
                self.models_dir
                / "simple"
                / "tlc_solvent_ratio_LightGBM_rdkit_solvent_simple_model.pkl"
            ),

            "silica_start":
            joblib.load(
                self.models_dir
                / "simple"
                / "silica_solvent_start_ratio_LightGBM_morgan_rdkit_solvent_simple_model.pkl"
            ),

            "silica_end":
            joblib.load(
                self.models_dir
                / "simple"
                / "silica_solvent_end_ratio_LightGBM_rdkit_solvent_simple_model.pkl"
            ),

        }

        self.models["complex"] = {

            "method":
            joblib.load(
                self.models_dir
                / "complex"
                / "method_LightGBM_for_imbalance_morgan_agent_complex_model.pkl"
            ),

            "solvent":
            joblib.load(
                self.models_dir
                / "complex"
                / "solvent_type_LightGBM_for_imbalance_rdkit_agent_method_complex_model.pkl"
            ),

            "tlc":
            joblib.load(
                self.models_dir
                / "complex"
                / "tlc_solvent_ratio_LightGBM_morgan_rdkit_agent_solvent_complex_model.pkl"
            ),

            "silica_start":
            joblib.load(
                self.models_dir
                / "complex"
                / "silica_solvent_start_ratio_LightGBM_morgan_agent_solvent_complex_model.pkl"
            ),

            "silica_end":
            joblib.load(
                self.models_dir
                / "complex"
                / "silica_solvent_end_ratio_LightGBM_rdkit_agent_solvent_complex_model.pkl"
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
                self.unique_agents
                if use_agent
                else None
            ),

            ref_method_list=UNIQUE_METHOD_LIST,

            ref_solvent_list=UNIQUE_SOLVENT_LIST,

            fp_dim=1024,

            fea_type=fea_type,

            target_compound=target_compound,

        )

    # =================================================
    # Feature setting
    # =================================================

    def _setting(self, mode):

        if mode == "simple":

            return {

                "target": "product",

                "method":
                "morgan_rdkit",

                "solvent":
                "morgan_method",

                "tlc":
                "rdkit_solvent",

                "silica_start":
                "morgan_rdkit_solvent",

                "silica_end":
                "rdkit_solvent",

            }

        elif mode == "complex":

            return {

                "target":
                "reactant_product",

                "method":
                "morgan_agent",

                "solvent":
                "rdkit_agent_method",

                "tlc":
                "morgan_rdkit_agent_solvent",

                "silica_start":
                "morgan_agent_solvent",

                "silica_end":
                "rdkit_agent_solvent",

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

        X = self._build_X(

            mode,

            setting["solvent"],

            setting["target"],

            product_smiles,

            reactant_smiles,

            agents,

            method=method,

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

                solvent=solvent,

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