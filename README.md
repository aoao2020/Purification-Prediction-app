# Purification App v2

Streamlitで動作する、精製条件予測とSMILES描画のアプリです。

## 起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

サイドバーから次のページを選択できます。

- `app`: 精製条件予測
- `SMILES Drawer`: SMILESの2D構造描画、分子情報表示、SVG/PNG出力

## SMILES Drawer

SMILESを入力すると即時に構造を検証・描画します。Canonical SMILES、分子式、分子量、
精密質量、LogPなどを確認でき、構造図をSVGまたはPNGとして保存できます。

ChemDrawから保存したCDXML、MOL、SDFファイルも読み込めます。複数構造を含むCDXML・SDFは
構造を選んで確認でき、全構造のIsomeric SMILESをCSVで一括保存できます。
