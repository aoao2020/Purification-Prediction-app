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
    page_title="SMILES Drawer",
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
      <div class="drawer-title">SMILES Drawer</div>
      <div class="drawer-subtitle">SMILESから、確認しやすく保存しやすい2D分子構造を生成します。</div>
    </header>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = {
    "アスピリン": "CC(=O)Oc1ccccc1C(=O)O",
    "カフェイン": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "イブプロフェン": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "ニコチン": "CN1CCC[C@H]1c2cccnc2",
}


def load_example(example_smiles: str) -> None:
    st.session_state.drawer_smiles = example_smiles


if "drawer_smiles" not in st.session_state:
    st.session_state.drawer_smiles = EXAMPLES["アスピリン"]

input_method = st.radio(
    "入力方法",
    ["SMILESを入力", "ChemDrawファイルを読み込む"],
    horizontal=True,
)

input_col, option_col = st.columns([1.65, 1], gap="large")

mol = None
input_error = None
uploaded_file = None

with input_col:
    if input_method == "SMILESを入力":
        smiles = st.text_area(
            "SMILES",
            key="drawer_smiles",
            height=112,
            placeholder="例: CC(=O)Oc1ccccc1C(=O)O",
            help="標準SMILES、立体化学を含むisomeric SMILES、複数成分を含むSMILESに対応します。",
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
            "ChemDrawから保存したファイル",
            type=["cdxml", "mol", "sdf"],
            help="おすすめはCDXMLです。複数構造を扱う場合はCDXMLまたはSDFを使用してください。",
        )
        st.caption("対応形式: CDXML・MOL・SDF（最大10 MB）")

        if uploaded_file is not None:
            try:
                imported_molecules = load_structure_file(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                selected_index = 0
                if len(imported_molecules) > 1:
                    selected_index = st.selectbox(
                        "変換する構造",
                        options=range(len(imported_molecules)),
                        format_func=lambda index: imported_molecules[index].name,
                    )
                    st.caption(f"{len(imported_molecules)}件の構造を検出しました。")
                    structure_rows = [
                        {
                            "構造名": item.name,
                            "Isomeric SMILES": summarize_molecule(item.mol).canonical_smiles,
                        }
                        for item in imported_molecules
                    ]
                    with st.expander("検出した構造の一覧"):
                        st.dataframe(
                            structure_rows,
                            use_container_width=True,
                            hide_index=True,
                        )
                        csv_buffer = StringIO()
                        writer = csv.DictWriter(
                            csv_buffer,
                            fieldnames=["構造名", "Isomeric SMILES"],
                        )
                        writer.writeheader()
                        writer.writerows(structure_rows)
                        st.download_button(
                            "全構造のSMILESをCSVで保存",
                            data=csv_buffer.getvalue().encode("utf-8-sig"),
                            file_name="structures_smiles.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                mol = imported_molecules[selected_index].mol
            except ValueError as exc:
                input_error = str(exc)

with option_col:
    st.markdown("**表示オプション**")
    atom_indices = st.toggle("原子番号を表示", value=False)
    bond_indices = st.toggle("結合番号を表示", value=False)
    monochrome = st.toggle("モノクロ", value=False)
    export_scale = st.select_slider(
        "書き出し解像度",
        options=["標準", "高解像度", "特大"],
        value="高解像度",
    )

sizes = {
    "標準": (720, 448),
    "高解像度": (1200, 746),
    "特大": (1800, 1120),
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

    st.success("有効なSMILESです", icon="✅")
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
            "SVGを保存",
            data=export_svg.encode("utf-8"),
            file_name="molecule.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )
        download_right.download_button(
            "PNGを保存",
            data=png_bytes,
            file_name="molecule.png",
            mime="image/png",
            use_container_width=True,
        )

    with detail_col:
        st.markdown("### 分子情報")
        st.markdown(
            f'<div class="formula">{html.escape(summary.formula)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("分子式")

        metric_a, metric_b = st.columns(2)
        metric_a.metric("分子量", f"{summary.molecular_weight:.2f}")
        metric_b.metric("精密質量", f"{summary.exact_mass:.4f}")
        metric_a.metric("重原子数", summary.heavy_atom_count)
        metric_b.metric("環の数", summary.ring_count)
        metric_a.metric("HBD", summary.h_bond_donors)
        metric_b.metric("HBA", summary.h_bond_acceptors)
        metric_a.metric("LogP", f"{summary.logp:.2f}")
        metric_b.metric("全原子数", summary.atom_count)

        st.markdown("#### Canonical SMILES")
        st.code(summary.canonical_smiles, language=None, wrap_lines=True)
        st.download_button(
            "Canonical SMILESを保存",
            data=(summary.canonical_smiles + "\n").encode("utf-8"),
            file_name="molecule.smi",
            mime="chemical/x-daylight-smiles",
            use_container_width=True,
        )

elif input_error:
    st.error(input_error, icon="⚠️")
    if input_method == "SMILESを入力":
        st.markdown(
            """
            **チェックポイント**

            - 開き括弧と閉じ括弧の数が合っているか
            - 環を表す番号が対になっているか
            - `Cl`、`Br`などの原子記号の大文字・小文字が正しいか
            - 電荷や同位体を表す角括弧 `[]` が閉じているか
            """
        )
elif input_method == "ChemDrawファイルを読み込む":
    st.info(
        "ChemDrawで構造を選び、ファイル → 別名で保存からCDXML形式で保存してアップロードしてください。",
        icon="📄",
    )

with st.expander("このツールについて"):
    st.write(
        "構造の解釈・正規化・物性計算・2D座標生成にはRDKitを使用しています。"
        "表示結果は研究上の確認用として利用し、重要な判断では原典や分析結果と照合してください。"
    )
    st.caption(
        "CDXMLはChemDrawの全機能を完全には表現できない場合があります。"
        "変換後の立体化学・電荷・結合次数をプレビューで確認してください。"
    )
