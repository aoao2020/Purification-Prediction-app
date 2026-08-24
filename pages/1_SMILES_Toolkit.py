from __future__ import annotations

import base64
import csv
import html
from io import StringIO

import streamlit as st

from src.smiles_drawer import (
    load_structure_file,
    parse_smiles,
    png_from_molecule,
    render_svg,
    summarize_molecule,
)


st.set_page_config(
    page_title="SMILES Toolkit",
    page_icon="⌬",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {max-width: 1240px; padding-top: 2.2rem;}
    .drawer-hero {display: block; width: 100%; padding: .35rem 0 1.6rem;
                  overflow: visible;}
    .drawer-kicker {display: block; font-size: .78rem; font-weight: 700;
                    letter-spacing: .13em; line-height: 1.6; padding: .15rem 0;
                    color: #18a47b; text-transform: uppercase; overflow: visible;}
    .drawer-title {display: block; font-size: clamp(2rem, 5vw, 3.4rem); font-weight: 720;
                   letter-spacing: -.045em; line-height: 1.2; padding: .08em 0;
                   margin: .15rem 0 .45rem; overflow: visible;}
    .drawer-subtitle {display: block; width: 100%; color: #687179; font-size: 1.02rem;
                      line-height: 1.7; margin: 0; white-space: normal;
                      overflow-wrap: anywhere; word-break: normal; overflow: visible;}
    .molecule-stage {background: #fbfcfa; border: 1px solid #dde5e0; border-radius: 18px;
                     min-height: 430px; padding: 28px; display: flex; align-items: center;
                     justify-content: center; box-shadow: 0 14px 35px rgba(24, 50, 40, .06);}
    .molecule-stage img {width: 100%; height: auto; max-height: 500px;
                         object-fit: contain; display: block;}
    .formula {font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              color: #168263; font-size: 1.15rem; font-weight: 650;}
    [data-testid="stMetricValue"] {font-size: 1.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="drawer-hero">
      <div class="drawer-kicker">Molecular canvas</div>
      <div class="drawer-title">SMILES Toolkit</div>
      <div class="drawer-subtitle">Validate, inspect, visualize, and export molecular structures from SMILES.</div>
    </header>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "Nicotine": "CN1CCC[C@H]1c2cccnc2",
}


def load_example(example_smiles: str) -> None:
    st.session_state.drawer_smiles = example_smiles


if "drawer_smiles" not in st.session_state:
    st.session_state.drawer_smiles = EXAMPLES["Aspirin"]

input_method = st.radio(
    "Input method",
    ["Enter SMILES", "Upload a ChemDraw file"],
    horizontal=True,
)

input_col, option_col = st.columns([1.65, 1], gap="large")

mol = None
input_error = None
uploaded_file = None

with input_col:
    if input_method == "Enter SMILES":
        smiles = st.text_area(
            "SMILES",
            key="drawer_smiles",
            height=112,
            placeholder="Example: CC(=O)Oc1ccccc1C(=O)O",
            help=(
                "Supports standard SMILES, isomeric SMILES with stereochemistry, "
                "and multi-component SMILES."
            ),
        )

        example_cols = st.columns(len(EXAMPLES))
        for column, (name, example_smiles) in zip(example_cols, EXAMPLES.items()):
            column.button(
                name,
                use_container_width=True,
                on_click=load_example,
                args=(example_smiles,),
            )

        try:
            mol = parse_smiles(smiles)
        except ValueError as exc:
            input_error = str(exc)

    else:
        uploaded_file = st.file_uploader(
            "File exported from ChemDraw",
            type=["cdxml", "mol", "sdf"],
            help=(
                "CDXML is recommended. Use CDXML or SDF for files containing "
                "multiple structures."
            ),
        )
        st.caption("Supported formats: CDXML, MOL, and SDF (maximum 10 MB)")

        if uploaded_file is not None:
            try:
                imported_molecules = load_structure_file(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                selected_index = 0
                if len(imported_molecules) > 1:
                    selected_index = st.selectbox(
                        "Structure to convert",
                        options=range(len(imported_molecules)),
                        format_func=lambda index: imported_molecules[index].name,
                    )
                    st.caption(
                        f"Detected {len(imported_molecules)} structures."
                    )
                    structure_rows = [
                        {
                            "Structure name": item.name,
                            "Isomeric SMILES": summarize_molecule(item.mol).canonical_smiles,
                        }
                        for item in imported_molecules
                    ]
                    with st.expander("Detected structures"):
                        st.dataframe(
                            structure_rows,
                            use_container_width=True,
                            hide_index=True,
                        )
                        csv_buffer = StringIO()
                        writer = csv.DictWriter(
                            csv_buffer,
                            fieldnames=["Structure name", "Isomeric SMILES"],
                        )
                        writer.writeheader()
                        writer.writerows(structure_rows)
                        st.download_button(
                            "Download all structures as CSV",
                            data=csv_buffer.getvalue().encode("utf-8-sig"),
                            file_name="structures_smiles.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                mol = imported_molecules[selected_index].mol
            except ValueError as exc:
                input_error = str(exc)

with option_col:
    st.markdown("**Display options**")
    atom_indices = st.toggle("Show atom indices", value=False)
    bond_indices = st.toggle("Show bond indices", value=False)
    monochrome = st.toggle("Monochrome", value=False)
    export_scale = st.select_slider(
        "Export resolution",
        options=["Standard", "High resolution", "Extra large"],
        value="High resolution",
    )

sizes = {
    "Standard": (720, 448),
    "High resolution": (1200, 746),
    "Extra large": (1800, 1120),
}
export_width, export_height = sizes[export_scale]

if mol is not None:
    summary = summarize_molecule(mol)
    display_svg = render_svg(
        mol,
        width=900,
        height=560,
        atom_indices=atom_indices,
        bond_indices=bond_indices,
        monochrome=monochrome,
    )
    export_svg = render_svg(
        mol,
        width=export_width,
        height=export_height,
        atom_indices=atom_indices,
        bond_indices=bond_indices,
        monochrome=monochrome,
    )

    st.success("Valid SMILES", icon="✅")
    canvas_col, detail_col = st.columns([1.65, 1], gap="large")

    with canvas_col:
        encoded_svg = base64.b64encode(display_svg.encode("utf-8")).decode("ascii")
        st.markdown(
            '<div class="molecule-stage">'
            f'<img src="data:image/svg+xml;base64,{encoded_svg}" alt="2D molecular structure">'
            "</div>",
            unsafe_allow_html=True,
        )

        png_bytes = png_from_molecule(
            mol,
            width=export_width,
            height=export_height,
            atom_indices=atom_indices,
            bond_indices=bond_indices,
            monochrome=monochrome,
        )
        download_left, download_right = st.columns(2)
        download_left.download_button(
            "Download SVG",
            data=export_svg.encode("utf-8"),
            file_name="molecule.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )
        download_right.download_button(
            "Download PNG",
            data=png_bytes,
            file_name="molecule.png",
            mime="image/png",
            use_container_width=True,
        )

    with detail_col:
        st.markdown("### Molecular information")
        st.markdown(
            f'<div class="formula">{html.escape(summary.formula)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Molecular formula")

        metric_a, metric_b = st.columns(2)
        metric_a.metric("Molecular weight", f"{summary.molecular_weight:.2f}")
        metric_b.metric("Exact mass", f"{summary.exact_mass:.4f}")
        metric_a.metric("Heavy atoms", summary.heavy_atom_count)
        metric_b.metric("Rings", summary.ring_count)
        metric_a.metric("HBD", summary.h_bond_donors)
        metric_b.metric("HBA", summary.h_bond_acceptors)
        metric_a.metric("LogP", f"{summary.logp:.2f}")
        metric_b.metric("Total atoms", summary.atom_count)

        st.markdown("#### Canonical SMILES")
        st.code(summary.canonical_smiles, language=None, wrap_lines=True)
        st.download_button(
            "Download Canonical SMILES",
            data=(summary.canonical_smiles + "\n").encode("utf-8"),
            file_name="molecule.smi",
            mime="chemical/x-daylight-smiles",
            use_container_width=True,
        )

elif input_error:
    st.error(input_error, icon="⚠️")
    if input_method == "Enter SMILES":
        st.markdown(
            """
            **Things to check**

            - Opening and closing parentheses are balanced
            - Ring closure numbers occur in pairs
            - Element symbols such as `Cl` and `Br` use the correct capitalization
            - Square brackets `[]` for charges or isotopes are closed
            """
        )
elif input_method == "Upload a ChemDraw file":
    st.info(
        "In ChemDraw, select the structure and use File > Save As to export it "
        "as a CDXML file, then upload it here.",
        icon="📄",
    )

with st.expander("About this tool"):
    st.write(
        "RDKit is used for structure parsing, normalization, property calculation, "
        "and 2D coordinate generation. Use the results for research review and "
        "verify important decisions against primary sources or analytical data."
    )
    st.caption(
        "CDXML may not preserve every ChemDraw feature. Review the converted "
        "stereochemistry, charges, and bond orders in the preview."
    )
