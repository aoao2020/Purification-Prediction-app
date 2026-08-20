"""SMILES parsing, molecular properties, and 2D rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


@dataclass(frozen=True)
class MoleculeSummary:
    canonical_smiles: str
    formula: str
    molecular_weight: float
    exact_mass: float
    atom_count: int
    heavy_atom_count: int
    ring_count: int
    h_bond_donors: int
    h_bond_acceptors: int
    logp: float


@dataclass(frozen=True)
class ImportedMolecule:
    name: str
    mol: Chem.Mol


SUPPORTED_STRUCTURE_FORMATS = {".cdxml", ".mol", ".sdf"}
MAX_STRUCTURE_FILE_SIZE = 10 * 1024 * 1024
DRAWING_PADDING = 0.16
ATOM_LABEL_PADDING = 0.06


def _decode_chemical_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _display_name(mol: Chem.Mol, index: int) -> str:
    if mol.HasProp("_Name"):
        name = mol.GetProp("_Name").strip()
        if name:
            return name
    return f"構造 {index + 1}"


def load_structure_file(filename: str, data: bytes) -> list[ImportedMolecule]:
    """Load one or more molecules from a ChemDraw-compatible text format."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_STRUCTURE_FORMATS:
        raise ValueError("対応形式は CDXML、MOL、SDF です。")
    if not data:
        raise ValueError("ファイルが空です。")
    if len(data) > MAX_STRUCTURE_FILE_SIZE:
        raise ValueError("ファイルサイズは10 MB以下にしてください。")

    try:
        if suffix == ".cdxml":
            mols = list(
                Chem.MolsFromCDXML(
                    _decode_chemical_text(data),
                    sanitize=True,
                    removeHs=True,
                )
            )
        elif suffix == ".mol":
            mol = Chem.MolFromMolBlock(
                _decode_chemical_text(data),
                sanitize=True,
                removeHs=True,
                strictParsing=False,
            )
            mols = [mol] if mol is not None else []
        else:
            supplier = Chem.ForwardSDMolSupplier(
                BytesIO(data),
                sanitize=True,
                removeHs=True,
                strictParsing=False,
            )
            mols = [mol for mol in supplier if mol is not None]
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "構造を読み取れませんでした。ChemDrawからCDXMLまたはMOLとして保存し直してください。"
        ) from exc

    if not mols:
        raise ValueError("ファイル内に読み取り可能な分子構造がありません。")

    return [
        ImportedMolecule(name=_display_name(mol, index), mol=mol)
        for index, mol in enumerate(mols)
    ]


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string and raise a user-facing error if it is invalid."""
    value = str(smiles or "").strip()
    if not value:
        raise ValueError("SMILESを入力してください。")

    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise ValueError("SMILESを解釈できません。括弧、環番号、原子記号を確認してください。")

    # Generate stable 2D coordinates once so every export has the same layout.
    AllChem.Compute2DCoords(mol)
    return mol


def summarize_molecule(mol: Chem.Mol) -> MoleculeSummary:
    """Calculate a compact set of commonly used molecular properties."""
    return MoleculeSummary(
        canonical_smiles=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=Descriptors.MolWt(mol),
        exact_mass=Descriptors.ExactMolWt(mol),
        atom_count=mol.GetNumAtoms(),
        heavy_atom_count=mol.GetNumHeavyAtoms(),
        ring_count=rdMolDescriptors.CalcNumRings(mol),
        h_bond_donors=Lipinski.NumHDonors(mol),
        h_bond_acceptors=Lipinski.NumHAcceptors(mol),
        logp=Descriptors.MolLogP(mol),
    )


def _configure_draw_options(
    options: rdMolDraw2D.MolDrawOptions,
    *,
    atom_indices: bool,
    bond_indices: bool,
    monochrome: bool,
) -> None:
    options.addAtomIndices = atom_indices
    options.addBondIndices = bond_indices
    # Extra internal whitespace prevents atom labels and annotations at the
    # molecular bounds from being clipped by the SVG/PNG canvas.
    options.padding = DRAWING_PADDING
    options.additionalAtomLabelPadding = ATOM_LABEL_PADDING

    if monochrome:
        options.useBWAtomPalette()


def render_svg(
    mol: Chem.Mol,
    *,
    width: int = 900,
    height: int = 560,
    atom_indices: bool = False,
    bond_indices: bool = False,
    monochrome: bool = False,
) -> str:
    """Render a molecule as scalable SVG text."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    _configure_draw_options(
        options,
        atom_indices=atom_indices,
        bond_indices=bond_indices,
        monochrome=monochrome,
    )

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def png_from_molecule(
    mol: Chem.Mol,
    *,
    width: int = 900,
    height: int = 560,
    atom_indices: bool = False,
    bond_indices: bool = False,
    monochrome: bool = False,
) -> bytes:
    """Render a molecule as PNG bytes using RDKit's Cairo drawer."""
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    _configure_draw_options(
        options,
        atom_indices=atom_indices,
        bond_indices=bond_indices,
        monochrome=monochrome,
    )

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
