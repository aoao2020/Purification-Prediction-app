# src/feature_builder.py

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# ============================================================
# Constants
# ============================================================

NORMALIZED_AGENT_DICT = {
    "DMF(1drop)": "DMF",
    "DMF(cat)": "DMF",
    "HCl(g)": "HCl",
    "H2(1bar)": "H2",
    "H2(5bar)": "H2",
    "conc.HCl": "HCl",
}

RDKIT_DELETE_LIST = [
    "Ipc",
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex",
    "SPS",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
    "BalabanJ",
]


# ============================================================
# Basic utilities
# ============================================================

def _is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def _safe_str(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _make_empty_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("")
    if mol is None:
        raise ValueError("Failed to create empty RDKit molecule.")
    return mol


def _normalize_agent_text(agent: str) -> str:
    agent = unicodedata.normalize("NFKC", agent)
    agent = re.sub(r"[‐-‒–—−]", "-", agent)

    agent = re.sub(r"^[0-9]\)\s*", "", agent)
    agent = re.sub(r"^[0-9]+\%", "", agent)
    agent = re.sub(r"^\d+(\.\d+)?\s*M\s*", "", agent)

    agent = agent.replace(" ", "")
    agent = agent.replace("then", "")

    agent = NORMALIZED_AGENT_DICT.get(agent, agent)

    if bool(re.match(r"^\d+(\.\d+)?M", agent)):
        agent = re.sub(r"^\d+(\.\d+)?M", "", agent)

    return agent


# ============================================================
# Molecule parsing
# ============================================================

def parse_product_molecule(
    smiles: str,
    delete_compounds_list: Optional[list[str]] = None,
) -> Chem.Mol:
    """
    Parse Product SMILES.

    If the SMILES contains multiple components separated by ".",
    the first valid component not included in delete_compounds_list is used.
    """
    smiles = _safe_str(smiles)

    if not smiles:
        raise ValueError("Product SMILES is empty.")

    if delete_compounds_list is None:
        delete_compounds_list = []

    delete_mols = [
        Chem.MolFromSmiles(smi)
        for smi in delete_compounds_list
        if Chem.MolFromSmiles(smi) is not None
    ]

    smiles_parts = list(dict.fromkeys(smiles.split(".")))

    for part in smiles_parts:
        mol = Chem.MolFromSmiles(part)

        if mol is None:
            continue

        is_delete_compound = any(
            mol.HasSubstructMatch(delete_mol) and delete_mol.HasSubstructMatch(mol)
            for delete_mol in delete_mols
        )

        if not is_delete_compound:
            return mol

    raise ValueError(f"Product compound error: {smiles}")


def parse_reactant_molecules(
    smiles: str,
    delete_compounds_list: Optional[list[str]] = None,
) -> list[Chem.Mol]:
    """
    Parse Reactant SMILES.

    If multiple components are separated by ".", all valid non-deleted components
    are used. If no valid molecule remains, an empty molecule is returned.
    """
    smiles = _safe_str(smiles)

    if delete_compounds_list is None:
        delete_compounds_list = []

    delete_mols = [
        Chem.MolFromSmiles(smi)
        for smi in delete_compounds_list
        if Chem.MolFromSmiles(smi) is not None
    ]

    if not smiles:
        return [_make_empty_mol()]

    mols: list[Chem.Mol] = []

    smiles_parts = list(dict.fromkeys(smiles.split(".")))

    for part in smiles_parts:
        mol = Chem.MolFromSmiles(part)

        if mol is None:
            continue

        is_delete_compound = any(
            mol.HasSubstructMatch(delete_mol) and delete_mol.HasSubstructMatch(mol)
            for delete_mol in delete_mols
        )

        if not is_delete_compound:
            mols.append(mol)

    if len(mols) == 0:
        return [_make_empty_mol()]

    return mols


# ============================================================
# Morgan fingerprint
# ============================================================

def calc_morgan_count(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """
    Calculate count-based Morgan fingerprint.

    This matches the original generate_feature() default:
        fp_type="count_morgan"
    """
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float64)

    for idx, count in fp.GetNonzeroElements().items():
        arr[int(idx)] = float(count)

    return arr


def calc_morgan_bit(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """
    Calculate bit-based Morgan fingerprint.
    This is included for compatibility, although the current models use count_morgan.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.asarray(list(fp), dtype=np.float64)


def calc_morgan_for_mol(
    mol: Chem.Mol,
    fp_type: str = "count_morgan",
    fp_dim: int = 1024,
) -> np.ndarray:
    if fp_type == "count_morgan":
        return calc_morgan_count(mol, radius=2, n_bits=fp_dim)

    if fp_type == "bit_morgan":
        return calc_morgan_bit(mol, radius=2, n_bits=fp_dim)

    raise ValueError(f"Invalid fp_type: {fp_type}")


def calc_morgan_for_reactant_mols(
    mols: list[Chem.Mol],
    fp_type: str = "count_morgan",
    fp_dim: int = 1024,
) -> np.ndarray:
    """
    For reactants with multiple molecules:
    - count_morgan: sum counts across molecules
    - bit_morgan: OR across molecules
    """
    fps = [calc_morgan_for_mol(mol, fp_type=fp_type, fp_dim=fp_dim) for mol in mols]

    if len(fps) == 0:
        return np.zeros(fp_dim, dtype=np.float64)

    stacked = np.vstack(fps)

    if fp_type == "bit_morgan":
        return (stacked.sum(axis=0) > 0).astype(np.float64)

    return stacked.sum(axis=0).astype(np.float64)


# ============================================================
# RDKit descriptors
# ============================================================

def get_mol_descriptors(
    mol: Chem.Mol,
    delete_list: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Calculate RDKit descriptors and remove unstable descriptors.

    The output order follows rdkit.Chem.Descriptors._descList after deletion.
    """
    if delete_list is None:
        delete_list = []

    descriptor_dict: dict[str, float] = {}

    for name, func in Descriptors._descList:
        if name in delete_list:
            continue

        try:
            value = func(mol)
        except Exception:
            value = np.nan

        try:
            value = float(value)
        except Exception:
            value = np.nan

        if np.isinf(value):
            value = np.nan

        descriptor_dict[name] = value

    return descriptor_dict


def calc_rdkit_for_mol(mol: Chem.Mol) -> np.ndarray:
    desc = get_mol_descriptors(mol, delete_list=RDKIT_DELETE_LIST)
    values = np.asarray(list(desc.values()), dtype=np.float64)

    # LightGBM can handle NaN, but keeping them explicit is safer.
    return values


def calc_rdkit_for_reactant_mols(mols: list[Chem.Mol]) -> np.ndarray:
    """
    For multiple reactant molecules, RDKit descriptors are averaged.
    This follows the original generate_feature() logic.
    """
    rows = [calc_rdkit_for_mol(mol) for mol in mols]

    if len(rows) == 0:
        return np.zeros(198, dtype=np.float64)

    mat = np.vstack(rows)
    return np.nanmean(mat, axis=0).astype(np.float64)


def get_rdkit_descriptor_names(prefix: str = "product_rdkit") -> list[str]:
    desc = get_mol_descriptors(_make_empty_mol(), delete_list=RDKIT_DELETE_LIST)
    return [f"{prefix}_{key}" for key in desc.keys()]


# ============================================================
# One-hot features
# ============================================================

def make_agent_onehot(
    agents: str,
    ref_agents_list: list[str],
    delimiter: str = ", ",
) -> np.ndarray:
    """
    Convert Agents text into one-hot vector.

    Original logic:
        strip_agents = [x.strip() for x in agents.split(', ')]
    """
    if ref_agents_list is None:
        raise ValueError("ref_agents_list is required for agent features.")

    agents = _safe_str(agents)

    one_hot = np.zeros(len(ref_agents_list), dtype=np.float64)

    if not agents:
        return one_hot

    agents = unicodedata.normalize("NFKC", agents)
    agents = re.sub(r"[‐-‒–—−]", "-", agents)

    # Main rule from the original script: split by ", "
    raw_tokens = [x.strip() for x in agents.split(delimiter)]

    # Fallback for users who type comma without a space.
    if len(raw_tokens) == 1 and "," in agents:
        raw_tokens = [x.strip() for x in agents.split(",")]

    normalized_tokens = [_normalize_agent_text(token) for token in raw_tokens if token.strip()]

    ref_index = {str(agent): i for i, agent in enumerate(ref_agents_list)}

    for token in normalized_tokens:
        if token in ref_index:
            one_hot[ref_index[token]] = 1.0

    return one_hot


def make_exact_onehot(
    value: str,
    ref_list: list[str],
) -> np.ndarray:
    if ref_list is None:
        raise ValueError("ref_list is required for one-hot features.")

    value = _safe_str(value)

    one_hot = np.zeros(len(ref_list), dtype=np.float64)

    for i, ref in enumerate(ref_list):
        if value == str(ref):
            one_hot[i] = 1.0

    return one_hot


# ============================================================
# Single-row feature generation
# ============================================================

def build_single_feature_vector(
    *,
    product_smiles: str,
    reactant_smiles: str = "",
    agents: str = "",
    method: str = "",
    solvent: str = "",
    fp_type: str = "count_morgan",
    ref_agents_list: Optional[list[str]] = None,
    ref_method_list: Optional[list[str]] = None,
    ref_solvent_list: Optional[list[str]] = None,
    fp_dim: int = 1024,
    fea_type: str = "morgan_agent_solvent",
    target_compound: str = "product",
    delete_compounds_list: Optional[list[str]] = None,
) -> tuple[list[str], np.ndarray]:
    """
    Build a feature vector for a single prediction sample.

    Feature order exactly follows the original generate_feature():
        tokens = fea_type.split("_")
        selected = registry[tokens in order]

    For reactant_product:
        morgan block = Reactant Morgan -> Product Morgan
        rdkit block  = Reactant RDKit  -> Product RDKit
    """
    if delete_compounds_list is None:
        delete_compounds_list = []

    if target_compound not in {"product", "reactant_product"}:
        raise ValueError(f"Unknown target_compound: {target_compound}")

    use_reactant = target_compound == "reactant_product"

    if "agent" in fea_type and ref_agents_list is None:
        raise ValueError("ref_agents_list is required because 'agent' is included in fea_type.")

    if "method" in fea_type and ref_method_list is None:
        raise ValueError("ref_method_list is required because 'method' is included in fea_type.")

    if "solvent" in fea_type and ref_solvent_list is None:
        raise ValueError("ref_solvent_list is required because 'solvent' is included in fea_type.")

    product_mol = parse_product_molecule(
        product_smiles,
        delete_compounds_list=delete_compounds_list,
    )

    reactant_mols: list[Chem.Mol] = []

    if use_reactant:
        reactant_mols = parse_reactant_molecules(
            reactant_smiles,
            delete_compounds_list=delete_compounds_list,
        )

    registry: dict[str, list[tuple[str, list[str], np.ndarray]]] = {}

    # Morgan
    if "morgan" in fea_type:
        product_morgan = calc_morgan_for_mol(
            product_mol,
            fp_type=fp_type,
            fp_dim=fp_dim,
        )

        product_morgan_names = [f"product_morgan_{i}" for i in range(fp_dim)]

        if use_reactant:
            reactant_morgan = calc_morgan_for_reactant_mols(
                reactant_mols,
                fp_type=fp_type,
                fp_dim=fp_dim,
            )
            reactant_morgan_names = [f"reactant_morgan_{i}" for i in range(fp_dim)]

            registry["morgan"] = [
                ("reactant", reactant_morgan_names, reactant_morgan),
                ("product", product_morgan_names, product_morgan),
            ]
        else:
            registry["morgan"] = [
                ("product", product_morgan_names, product_morgan),
            ]

    # Agent
    if "agent" in fea_type:
        agent_vec = make_agent_onehot(
            agents=agents,
            ref_agents_list=ref_agents_list,
        )
        registry["agent"] = [
            ("agent", list(ref_agents_list), agent_vec),
        ]

    # Method
    if "method" in fea_type:
        method_vec = make_exact_onehot(
            value=method,
            ref_list=ref_method_list,
        )
        registry["method"] = [
            ("method", list(ref_method_list), method_vec),
        ]

    # Solvent
    if "solvent" in fea_type:
        solvent_vec = make_exact_onehot(
            value=solvent,
            ref_list=ref_solvent_list,
        )
        registry["solvent"] = [
            ("solvent", list(ref_solvent_list), solvent_vec),
        ]

    # RDKit
    if "rdkit" in fea_type:
        product_rdkit = calc_rdkit_for_mol(product_mol)
        product_rdkit_names = get_rdkit_descriptor_names(prefix="product_rdkit")

        if use_reactant:
            reactant_rdkit = calc_rdkit_for_reactant_mols(reactant_mols)
            reactant_rdkit_names = get_rdkit_descriptor_names(prefix="reactant_rdkit")

            registry["rdkit"] = [
                ("reactant", reactant_rdkit_names, reactant_rdkit),
                ("product", product_rdkit_names, product_rdkit),
            ]
        else:
            registry["rdkit"] = [
                ("product", product_rdkit_names, product_rdkit),
            ]

    tokens = [token for token in fea_type.split("_") if token]

    selected_blocks: list[tuple[str, list[str], np.ndarray]] = []

    for token in tokens:
        if token not in registry:
            raise ValueError(
                f"Feature token '{token}' was requested in fea_type='{fea_type}', "
                f"but it was not generated."
            )

        selected_blocks.extend(registry[token])

    feature_names: list[str] = []
    feature_arrays: list[np.ndarray] = []

    for _, names, values in selected_blocks:
        feature_names.extend(names)
        feature_arrays.append(np.asarray(values, dtype=np.float64))

    feature_vector = np.concatenate(feature_arrays).astype(np.float64)

    return feature_names, feature_vector


def build_single_feature_dataframe(
    *,
    product_smiles: str,
    reactant_smiles: str = "",
    agents: str = "",
    method: str = "",
    solvent: str = "",
    fp_type: str = "count_morgan",
    ref_agents_list: Optional[list[str]] = None,
    ref_method_list: Optional[list[str]] = None,
    ref_solvent_list: Optional[list[str]] = None,
    fp_dim: int = 1024,
    fea_type: str = "morgan_agent_solvent",
    target_compound: str = "product",
    delete_compounds_list: Optional[list[str]] = None,
) -> pd.DataFrame:
    feature_names, feature_vector = build_single_feature_vector(
        product_smiles=product_smiles,
        reactant_smiles=reactant_smiles,
        agents=agents,
        method=method,
        solvent=solvent,
        fp_type=fp_type,
        ref_agents_list=ref_agents_list,
        ref_method_list=ref_method_list,
        ref_solvent_list=ref_solvent_list,
        fp_dim=fp_dim,
        fea_type=fea_type,
        target_compound=target_compound,
        delete_compounds_list=delete_compounds_list,
    )

    columns = [f"Column_{i}" for i in range(len(feature_vector))]

    return pd.DataFrame([feature_vector], columns=columns)


# ============================================================
# Batch-compatible function
# ============================================================

def generate_feature(
    input_df: pd.DataFrame,
    fp_type: str = "count_morgan",
    ref_agents_list: Optional[list[str]] = None,
    ref_method_list: Optional[list[str]] = None,
    ref_solvent_list: Optional[list[str]] = None,
    fp_dim: int = 1024,
    fea_type: str = "morgan_agent_solvent",
    target_compound: str = "product",
    delete_compounds_list: Optional[list[str]] = None,
):
    """
    Batch-compatible version of the original generate_feature().

    Returns:
        id_list, fea_name_list, fea_matrix
    """
    if delete_compounds_list is None:
        delete_compounds_list = []

    if "ID" in input_df.columns:
        id_list = input_df["ID"].tolist()
    else:
        id_list = list(range(len(input_df)))

    feature_name_list: Optional[list[str]] = None
    feature_matrix: list[np.ndarray] = []

    for _, row in input_df.iterrows():
        product = _safe_str(row.get("Product", ""))
        reactant = _safe_str(row.get("Reactant", ""))
        agents = _safe_str(row.get("Agents", ""))
        method = _safe_str(row.get("Method", ""))
        solvent = _safe_str(row.get("Solvent", ""))

        names, vec = build_single_feature_vector(
            product_smiles=product,
            reactant_smiles=reactant,
            agents=agents,
            method=method,
            solvent=solvent,
            fp_type=fp_type,
            ref_agents_list=ref_agents_list,
            ref_method_list=ref_method_list,
            ref_solvent_list=ref_solvent_list,
            fp_dim=fp_dim,
            fea_type=fea_type,
            target_compound=target_compound,
            delete_compounds_list=delete_compounds_list,
        )

        if feature_name_list is None:
            feature_name_list = names

        feature_matrix.append(vec)

    if feature_name_list is None:
        feature_name_list = []

    return id_list, feature_name_list, feature_matrix


# ============================================================
# Debug helper
# ============================================================

def describe_feature_vector(
    *,
    product_smiles: str,
    reactant_smiles: str = "",
    agents: str = "",
    method: str = "",
    solvent: str = "",
    fp_type: str = "count_morgan",
    ref_agents_list: Optional[list[str]] = None,
    ref_method_list: Optional[list[str]] = None,
    ref_solvent_list: Optional[list[str]] = None,
    fp_dim: int = 1024,
    fea_type: str = "morgan_agent_solvent",
    target_compound: str = "product",
) -> None:
    names, vec = build_single_feature_vector(
        product_smiles=product_smiles,
        reactant_smiles=reactant_smiles,
        agents=agents,
        method=method,
        solvent=solvent,
        fp_type=fp_type,
        ref_agents_list=ref_agents_list,
        ref_method_list=ref_method_list,
        ref_solvent_list=ref_solvent_list,
        fp_dim=fp_dim,
        fea_type=fea_type,
        target_compound=target_compound,
    )

    print("=" * 80)
    print(f"fea_type: {fea_type}")
    print(f"target_compound: {target_compound}")
    print(f"n_features: {len(vec)}")
    print("first 20 feature names:")
    print(names[:20])
    print("last 20 feature names:")
    print(names[-20:])
    print("=" * 80)