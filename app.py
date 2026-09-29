# app.py

from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from streamlit_ketcher import st_ketcher

from src.predictor import PurificationPredictor


# =====================================================
# Page config
# =====================================================

st.set_page_config(
    page_title="Purification Condition Prediction App",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"] p {
        font-size: 1.125rem;
        line-height: 1.6;
    }
    [data-testid="stTextInput"] input {
        font-size: 1.125rem;
        min-height: 2.75rem;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--text-color, inherit);
    }
    [data-testid="stCaptionContainer"] p {
        font-size: 1.25rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    .st-key-model_selection [data-testid="stRadio"] label p {
        font-size: 1.375rem;
        line-height: 1.5;
        font-weight: 600;
    }
    .st-key-model_selection [role="radiogroup"] {
        gap: 0.75rem 2rem;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMain"] [data-testid="stVerticalBlock"] {
        gap: 1.25rem;
    }
    .st-key-predict_action button {
        min-height: 3.5rem;
        padding: 0.75rem 1.5rem;
    }
    .st-key-predict_action button p {
        font-size: 1.375rem;
        font-weight: 700;
    }
    [class*="st-key-draw_action_"] button {
        min-height: 3rem;
        padding: 0.5rem 1rem;
        border-width: 2px;
    }
    [class*="st-key-draw_action_"] button p {
        font-size: 1.125rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Load predictor
# =====================================================

MODEL_CACHE_VERSION = "product-reaction-v2"


@st.cache_resource
def load_predictor(model_version):
    # Include the model version in Streamlit's cache key. This prevents a
    # predictor loaded from the previous model set from surviving a hot reload.
    del model_version
    root_dir = Path(__file__).resolve().parent
    return PurificationPredictor(root_dir=root_dir)


predictor = load_predictor(MODEL_CACHE_VERSION)


# =====================================================
# Display maps
# =====================================================

METHOD_DISPLAY_MAP = {
    "silica": "Silica",
    "nh silica": "NH Silica",
    "reverse phase": "Reverse Phase",
    "other": "Other",
}


# =====================================================
# Utility functions
# =====================================================

def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def display_method_name(method):
    method = str(method).strip()
    return METHOD_DISPLAY_MAP.get(method.lower(), method)


def parse_mol(smiles):
    smiles = clean_text(smiles)

    if not smiles:
        return None, ""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, ""

    canonical = Chem.MolToSmiles(mol)

    return mol, canonical


@st.dialog("Draw structure", width="large")
def draw_structure_dialog(*, target_key, editor_key, structure_label):
    st.markdown(f"**{structure_label}**")
    st.caption(
        "Draw or edit the structure, click Apply in the editor, "
        "then select Use this structure."
    )

    initial_smiles = clean_text(st.session_state.get(target_key, ""))
    drawn_smiles = st_ketcher(
        initial_smiles,
        key=editor_key,
        height=520,
    )
    drawn_smiles = clean_text(drawn_smiles)

    if drawn_smiles:
        st.markdown("**Generated SMILES**")
        st.code(drawn_smiles, language=None, wrap_lines=True)

    accept_col, cancel_col = st.columns([2, 1])

    if accept_col.button(
        "Use this structure",
        key=f"accept_{editor_key}",
        type="primary",
        use_container_width=True,
        disabled=not drawn_smiles,
    ):
        mol, canonical_smiles = parse_mol(drawn_smiles)

        if mol is None:
            st.error("The drawn structure could not be converted to a valid SMILES.")
        else:
            st.session_state[target_key] = canonical_smiles
            st.rerun()

    if cancel_col.button(
        "Cancel",
        key=f"cancel_{editor_key}",
        use_container_width=True,
    ):
        st.rerun()


def smiles_input_with_draw_button(
    *,
    label,
    input_key,
    placeholder,
    draw_key,
    editor_key,
    column_widths=(3, 1),
):
    input_col, draw_col = st.columns(
        column_widths,
        vertical_alignment="bottom",
    )

    smiles = input_col.text_input(
        f"{label} or Draw Button",
        key=input_key,
        placeholder=placeholder,
    )

    with draw_col.container(key=f"draw_action_{draw_key}"):
        if st.button(
            "⌬ Draw",
            key=draw_key,
            use_container_width=True,
        ):
            draw_structure_dialog(
                target_key=input_key,
                editor_key=editor_key,
                structure_label=label,
            )

    return smiles


def split_solvent_pair(solvent):
    if "/" in str(solvent):
        left, right = str(solvent).split("/", 1)
        return left.strip(), right.strip()

    return str(solvent), "other"


def format_ratio(solvent, value):
    left, right = split_solvent_pair(solvent)

    value = float(value)
    value = max(0.0, min(100.0, value))
    other = 100.0 - value

    return f"{left}:{right} = {value:.0f}:{other:.0f}"


def is_silica_column_method(method):
    return str(method).strip().lower() == "silica"


def is_other_method(method):
    return str(method).strip().lower() == "other"


def is_other_solvent(solvent):
    return str(solvent).strip().lower() == "other"


def display_probability_table(title, df, top_n=None):
    with st.expander(f"Show probability table: {title}"):
        show_df = df.copy()

        if top_n is not None:
            show_df = show_df.head(top_n)

        show_df["prob"] = show_df["prob"].map(lambda x: round(float(x), 4))
        show_df = show_df.rename(
            columns={
                "candidate": "Candidate",
                "prob": "Probability",
            }
        )

        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True,
        )


def display_ratio_for_silica(solvent, ratio_dict):
    start_compact = format_ratio(
        solvent,
        ratio_dict["silica_start"],
    )

    end_compact = format_ratio(
        solvent,
        ratio_dict["silica_end"],
    )

    st.info(f"**Silica start ratio**: {start_compact}")
    st.info(f"**Silica end ratio**: {end_compact}")


def display_tlc_prediction(
    *,
    mode,
    method,
    solvent,
    solvent_probability_df,
    product_smiles,
    reactant_smiles,
    agents,
):
    st.markdown("### Predicted TLC Conditions")

    with st.container(border=True):
        st.write(f"**Solvent system**: {solvent}")

        if is_other_solvent(solvent):
            st.warning(
                'TLC ratio prediction is unavailable for "other" solvent systems.'
            )
        else:
            ratio_dict = predictor.predict_ratio(
                mode=mode,
                solvent=solvent,
                product_smiles=product_smiles,
                reactant_smiles=reactant_smiles,
                agents=agents,
                method=method,
            )

            tlc_ratio = format_ratio(solvent, ratio_dict["tlc"])
            st.info(f"**Predicted TLC solvent ratio**: {tlc_ratio}")

    display_probability_table(
        title="TLC solvent system",
        df=solvent_probability_df,
    )


def display_solvent_candidate(
    *,
    mode,
    method_candidate,
    candidate_number,
    solvent_candidate,
    product_smiles,
    reactant_smiles,
    agents,
):
    with st.container(border=True):
        st.markdown(
            f"##### Solvent Candidate {candidate_number}"
        )
        st.write(f"**Solvent system**: {solvent_candidate}")

        if is_other_solvent(solvent_candidate):
            st.warning(
                'Ratio prediction is unavailable for "other" solvent systems.'
            )
            return

        if not is_silica_column_method(method_candidate):
            st.warning(
                "Ratio prediction is unavailable for this purification method."
            )
            return

        ratio_dict = predictor.predict_ratio(
            mode=mode,
            solvent=solvent_candidate,
            product_smiles=product_smiles,
            reactant_smiles=reactant_smiles,
            agents=agents,
            method=method_candidate,
        )

        display_ratio_for_silica(
            solvent=solvent_candidate,
            ratio_dict=ratio_dict,
        )


# =====================================================
# UI
# =====================================================

st.title("🧪 Purification Condition Prediction App")

st.caption(
    "Predict purification method candidates, solvent system candidates, "
    "and solvent ratios from molecular information."
)

with st.expander("How to use", expanded=False):
    st.write("1. Select a prediction mode.")
    st.write("2. Enter molecular information.")
    st.write("3. Click **Predict**.")
    st.write("4. Review top purification method candidates and their solvent candidates.")
    st.info(
        "Product-based mode uses Product SMILES only. "
        "Reaction-based mode uses Reactant SMILES, Product SMILES, and selected Agents/Reagents."
    )

st.markdown("## Input")

st.caption(
    "Not familiar with SMILES? Use the ⌬ Draw button to create a molecular "
    "structure and automatically fill in the SMILES input."
)

with st.container(key="model_selection"):
    mode_label = st.radio(
        "Prediction model",
        [
            "Product-based model (PBM)",
            "Reaction-based model (RBM)",
        ],
        format_func=lambda label: (
            f"{label}：Product only"
            if label.startswith("Product-based")
            else f"{label}：Product, Reactant, Agents"
        ),
        horizontal=True,
    )

mode = "product-based" if mode_label.startswith("Product-based") else "reaction-based"

reactant_smiles = ""
agents = ""
selected_agents = []

if mode == "product-based":
    product_smiles = smiles_input_with_draw_button(
        label="Product SMILES",
        input_key="product_smiles_input",
        placeholder="Example: CCO",
        draw_key="draw_product_product_based",
        editor_key="product_structure_editor",
    )

else:
    col1, col2 = st.columns(2)

    with col1:
        reactant_smiles = smiles_input_with_draw_button(
            label="Reactant SMILES",
            input_key="reactant_smiles_input",
            placeholder="Example: CC=O",
            draw_key="draw_reactant_reaction_based",
            editor_key="reactant_structure_editor",
        )

        product_smiles = smiles_input_with_draw_button(
            label="Product SMILES",
            input_key="product_smiles_input",
            placeholder="Example: CCO",
            draw_key="draw_product_reaction_based",
            editor_key="product_structure_editor",
        )

    with col2:
        selected_agents = st.multiselect(
            "Agents / Reagents",
            options=predictor.unique_agents,
            default=[],
            help=(
                "Select reagents from the training reagent dictionary. "
                "This avoids spelling mismatches in agent one-hot features."
            ),
        )

        agents = ", ".join(selected_agents)

        if selected_agents:
            st.caption(f"Selected agents: {agents}")
        else:
            st.caption("No agents selected. Agent features will be all zero.")

    st.info(
        "Reaction-based mode uses Reactant SMILES, Product SMILES, and selected Agents/Reagents "
        "to generate model features."
    )

with st.container(key="predict_action"):
    predict_clicked = st.button(
        "Predict",
        type="primary",
        use_container_width=True,
    )


# =====================================================
# Prediction
# =====================================================

if predict_clicked:

    product_smiles = clean_text(product_smiles)
    reactant_smiles = clean_text(reactant_smiles)
    agents = clean_text(agents)

    if not product_smiles:
        st.error("Please enter Product SMILES.")
        st.stop()

    if mode == "reaction-based":
        if not reactant_smiles:
            st.error("Please enter Reactant SMILES for Reaction-based mode.")
            st.stop()

        if not selected_agents:
            st.warning(
                "No Agents/Reagents were selected. "
                "Prediction will continue with all agent features set to zero."
            )

    product_mol, canonical_product = parse_mol(product_smiles)

    if product_mol is None:
        st.error("Failed to parse Product SMILES. Please check the input.")
        st.stop()

    reactant_mol = None
    canonical_reactant = ""

    if mode == "reaction-based":
        reactant_mol, canonical_reactant = parse_mol(reactant_smiles)

        if reactant_mol is None:
            st.error("Failed to parse Reactant SMILES. Please check the input.")
            st.stop()

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Input Summary")

        st.write(f"**Prediction model**: {mode_label}")
        st.write(f"**Product SMILES**: `{canonical_product}`")

        if mode == "reaction-based":
            st.write(f"**Reactant SMILES**: `{canonical_reactant}`")
            st.write(
                f"**Agents / Reagents**: "
                f"{agents if agents else 'Not selected'}"
            )

        st.markdown("### Molecular Structure")

        try:
            st.image(
                MolToImage(product_mol),
                caption="Product molecular structure",
                use_container_width=True,
            )
        except Exception:
            st.warning("Product molecular structure could not be generated.")

        if mode == "reaction-based":
            try:
                st.image(
                    MolToImage(reactant_mol),
                    caption="Reactant molecular structure",
                    use_container_width=True,
                )
            except Exception:
                st.warning("Reactant molecular structure could not be generated.")

    with right_col:
        try:
            # -------------------------------------------------
            # 1. Predict method probabilities
            # -------------------------------------------------
            method_pred, method_prob_df = predictor.predict_method(
                mode=mode,
                product_smiles=product_smiles,
                reactant_smiles=reactant_smiles,
                agents=agents,
            )

            top_methods = method_prob_df.head(2).reset_index(drop=True)

            st.markdown(
                f"### Predicted Chromatography Method：{display_method_name(method_pred)}"
            )
            display_probability_table(
                title="Purification method",
                df=method_prob_df,
            )

            # TLC prediction is shown first, using the top-ranked method and
            # its top-ranked solvent system.
            if is_other_method(method_pred):
                st.markdown("### Predicted TLC Conditions")
                st.warning(
                    "TLC and solvent prediction are unavailable when the "
                    "predicted purification method is Other."
                )
            else:
                tlc_solvent_pred, tlc_solvent_prob_df = predictor.predict_solvent(
                    mode=mode,
                    method=method_pred,
                    product_smiles=product_smiles,
                    reactant_smiles=reactant_smiles,
                    agents=agents,
                )

                display_tlc_prediction(
                    mode=mode,
                    method=method_pred,
                    solvent=tlc_solvent_pred,
                    solvent_probability_df=tlc_solvent_prob_df,
                    product_smiles=product_smiles,
                    reactant_smiles=reactant_smiles,
                    agents=agents,
                )

            st.markdown("### Predicted CC Conditions")

            for method_idx, method_row in top_methods.iterrows():
                method_candidate = method_row["candidate"]
                method_display = display_method_name(method_candidate)

                with st.container():
                    st.markdown(
                        f"#### Method Candidate {method_idx + 1}：{method_display}"
                    )

                    if is_other_method(method_candidate):
                        st.warning(
                            "Solvent prediction is unavailable when the "
                            "purification method is Other."
                        )
                        continue

                    # -------------------------------------------------
                    # 2. Predict solvent probabilities for each method candidate
                    # -------------------------------------------------
                    solvent_pred, solvent_prob_df = predictor.predict_solvent(
                        mode=mode,
                        method=method_candidate,
                        product_smiles=product_smiles,
                        reactant_smiles=reactant_smiles,
                        agents=agents,
                    )

                    top_solvents = solvent_prob_df.head(2).reset_index(drop=True)

                    for solvent_idx, solvent_row in top_solvents.iterrows():
                        solvent_candidate = solvent_row["candidate"]

                        display_solvent_candidate(
                            mode=mode,
                            method_candidate=method_candidate,
                            candidate_number=solvent_idx + 1,
                            solvent_candidate=solvent_candidate,
                            product_smiles=product_smiles,
                            reactant_smiles=reactant_smiles,
                            agents=agents,
                        )

                    display_probability_table(
                        title=f"Solvent system for {method_display}",
                        df=solvent_prob_df,
                    )

            st.markdown("---")

            with st.expander("Notes"):
                st.caption(
                    "Purification method candidates are shown up to top 2."
                )
                st.caption(
                    "For each method candidate, solvent system candidates are predicted separately."
                )
                st.caption(
                    "Solvent and TLC predictions are not available when the predicted method is Other."
                )
                st.caption(
                    "Silica start and end ratios are displayed only when the predicted method is Silica Column."
                )
                st.caption(
                    'Ratio prediction is not displayed for "other" solvent systems.'
                )
                st.caption(
                    "The TLC solvent ratio uses the top solvent system predicted for the top purification method."
                )
                st.caption(
                    "Solvent ratios are displayed as A:B = x:y. "
                    "For example, EtOAc:Hexane = 40:60 means EtOAc 40% and Hexane 60%."
                )

        except Exception as e:
            st.error("An error occurred during prediction.")
            st.exception(e)
