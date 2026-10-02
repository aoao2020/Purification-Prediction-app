"""Exercise manual-condition routing with the real prediction models."""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.predictor import PurificationPredictor


def predict(app):
    next(button for button in app.button if button.label == "Predict").click().run()
    assert not app.exception, [error.message for error in app.exception]
    assert not app.error, [error.value for error in app.error]


def main():
    for model in ["Product-based model (PBM)", "Reaction-based model (RBM)"]:
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=60).run()
        app.toggle(key="advanced_mode").set_value(True).run()
        app.radio[0].set_value(model).run()
        assert app.selectbox(key="custom_solvent").disabled
        next(b for b in app.button if b.label == "Predict").click().run()
        assert app.error[0].value == "Please select a Method for advanced prediction."

        app.selectbox(key="custom_method").set_value("Silica").run()
        app.text_input(key="product_smiles_input").set_value("CCO")
        if model.startswith("Reaction"):
            app.text_input(key="reactant_smiles_input").set_value("CC=O")

        # A supplied method must never be replaced by a method prediction.
        with patch.object(PurificationPredictor, "predict_method", side_effect=AssertionError("Unexpected method prediction")):
            predict(app)
            assert len(app.dataframe) == 2
            assert any("Silica start ratio" in item.value for item in app.info)
            assert not any("Predicted Chromatography Method" in item.value for item in app.markdown)

            app.selectbox(key="custom_solvent").set_value("MeOH/CH$_{2}$Cl$_{2}$").run()
            with patch.object(PurificationPredictor, "predict_solvent", side_effect=AssertionError("Unexpected solvent prediction")):
                predict(app)
                assert not app.dataframe
                assert any("MeOH:CH2Cl2" in item.value for item in app.info)
                assert any("Silica end ratio" in item.value for item in app.info)

                app.selectbox(key="custom_method").set_value("NH Silica").run()
                predict(app)
                assert any("unavailable for this purification method" in item.value for item in app.warning)
                assert not any("Silica start ratio" in item.value for item in app.info)

        # Switching back ignores any retained manual selections.
        app.toggle(key="advanced_mode").set_value(False).run()
        assert not app.selectbox
        predict(app)
        assert any("Predicted Chromatography Method" in item.value for item in app.markdown)
        assert any(table.label == "Show probability table: Purification method" for table in app.expander)
        print(f"PASS: {model}")


if __name__ == "__main__":
    main()
