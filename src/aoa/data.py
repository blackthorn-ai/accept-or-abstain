"""Loaders for the public datasets used in the study (see PROVENANCE.md)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Training rows of the published out-of-sample additive tests (Ahneman et al. 2018,
# as used by Schwaller et al. 2021 and by the yield-paper redistribution).
BH_TEST_TRAIN_ROWS = {"Test1": 3057, "Test2": 3055, "Test3": 3058, "Test4": 3055}
BH_COMPONENTS = ["Aryl halide", "Ligand", "Base", "Additive"]

# Structures of the Suzuki-Miyaura components (Perera et al. 2018). Internal standard,
# counter-ion water and other spectator fragments are omitted.
SM_REACTANT_1 = {
    "6-chloroquinoline": "Clc1ccc2ncccc2c1",
    "6-Bromoquinoline": "Brc1ccc2ncccc2c1",
    "6-triflatequinoline": "O=S(=O)(Oc1ccc2ncccc2c1)C(F)(F)F",
    "6-Iodoquinoline": "Ic1ccc2ncccc2c1",
    "6-quinoline-boronic acid hydrochloride": "OB(O)c1ccc2ncccc2c1",
    "Potassium quinoline-6-trifluoroborate": "F[B-](F)(F)c1ccc2ncccc2c1.[K+]",
    "6-Quinolineboronic acid pinacol ester": "CC1(C)OB(c2ccc3ncccc3c2)OC1(C)C",
}
SM_REACTANT_2 = {
    "2a, Boronic Acid": "Cc1ccc2c(cnn2C2CCCCO2)c1B(O)O",
    "2b, Boronic Ester": "Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1",
    "2c, Trifluoroborate": "Cc1ccc2c(cnn2C2CCCCO2)c1[B-](F)(F)F.[K+]",
    "2d, Bromide": "Cc1ccc2c(cnn2C2CCCCO2)c1Br",
}
SM_LIGAND = {
    "P(tBu)3": "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
    "P(Ph)3": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    "AmPhos": "CN(C)c1ccc(P(C(C)(C)C)C(C)(C)C)cc1",
    "P(Cy)3": "C1CCC(P(C2CCCCC2)C2CCCCC2)CC1",
    "P(o-Tol)3": "Cc1ccccc1P(c1ccccc1C)c1ccccc1C",
    "CataCXium A": "CCCCP(C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2",
    "SPhos": "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1",
    "dtbpf": "CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C.CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C.[Fe+2]",
    "XPhos": "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1",
    "dppf": "c1ccc(P(c2ccccc2)c2ccc[cH-]2)cc1.c1ccc(P(c2ccccc2)c2ccc[cH-]2)cc1.[Fe+2]",
    "Xantphos": "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21",
    "None": "",
}
SM_REAGENT = {
    "NaOH": "[Na+].[OH-]",
    "NaHCO3": "O=C([O-])O.[Na+]",
    "CsF": "[Cs+].[F-]",
    "K3PO4": "O=P([O-])([O-])[O-].[K+].[K+].[K+]",
    "KOH": "[K+].[OH-]",
    "LiOtBu": "CC(C)(C)[O-].[Li+]",
    "Et3N": "CCN(CC)CC",
    "None": "",
}
SM_SOLVENT = {
    "MeCN": "CC#N",
    "THF": "C1CCOC1",
    "DMF": "CN(C)C=O",
    "MeOH": "CO",
    "MeOH/H2O_V2 9:1": "CO.O",
    "THF_V2": "C1CCOC1.O",
}
SM_COMPONENTS = ["Reactant 1", "Reactant 2", "Ligand", "Reagent", "Solvent"]


def _inchikey(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.MolToInchiKey(m)


def _clean(df: pd.DataFrame, smiles_col: str, y_col: str) -> pd.DataFrame:
    df = df[[smiles_col, y_col]].copy()
    df.columns = ["smiles", "y"]
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()
    df["smiles"] = df["smiles"].astype(str)
    df["inchikey"] = [_inchikey(s) for s in df["smiles"]]
    df = df.dropna(subset=["inchikey"])
    return df.reset_index(drop=True)


def load_prospective() -> pd.DataFrame:
    """E1: the SangsterLogP prospective set (327 compounds)."""
    df = pd.read_excel(RAW / "SangsterLogP_Datasets.xlsx", sheet_name=2)
    return _clean(df, "SMILES", "logP").drop_duplicates("inchikey").reset_index(drop=True)


def load_ex12() -> pd.DataFrame:
    """E2: the 183 measured amide logP values (fluorinated compounds and parents)."""
    df = pd.read_csv(RAW / "fluoric_IDs_SMILES_importances_pKa_logP.csv", dtype=str)
    df = df[df["logP"].notna() & df["Amide SMILES"].notna()].copy()
    df["logP"] = df["logP"].str.replace(",", ".", regex=False)
    out = pd.DataFrame(
        {"smiles": df["Amide SMILES"].astype(str), "y": pd.to_numeric(df["logP"], errors="coerce"),
         "f_group": df["F Group"].values}
    ).dropna(subset=["y"])
    out["inchikey"] = [_inchikey(s) for s in out["smiles"]]
    out = out.dropna(subset=["inchikey"]).reset_index(drop=True)
    patt = Chem.MolFromSmarts("[CX4]F")
    out["sp3_CF"] = [Chem.MolFromSmiles(s).HasSubstructMatch(patt) for s in out["smiles"]]
    return out


def load_logp_pool(exclude_keys: set[str]) -> pd.DataFrame:
    """D1: SangsterLogP main sheet, de-duplicated, external-set compounds removed."""
    df = pd.read_excel(RAW / "SangsterLogP_Datasets.xlsx", sheet_name=0)
    df = _clean(df, "SMILES", "logP").drop_duplicates("inchikey")
    df = df[~df["inchikey"].isin(exclude_keys)]
    patt = Chem.MolFromSmarts("[CX4]F")
    df["sp3_CF"] = [Chem.MolFromSmiles(s).HasSubstructMatch(patt) for s in df["smiles"]]
    return df.reset_index(drop=True)


def load_bh(sheet: str) -> tuple[pd.DataFrame, int | None]:
    """D3a: Buchwald-Hartwig sheet; returns the table and the number of training rows."""
    df = pd.read_excel(RAW / "Buchwald_Hartwig.xlsx", sheet_name=sheet)
    df = df.rename(columns={"Output": "y"})
    for c in BH_COMPONENTS:
        df[c] = df[c].fillna("").astype(str)
    return df.reset_index(drop=True), BH_TEST_TRAIN_ROWS.get(sheet)


def load_sm() -> pd.DataFrame:
    """D3b: Suzuki-Miyaura, components mapped to structures."""
    df = pd.read_excel(RAW / "Suzuki-Miyaura.xlsx", sheet_name=0)
    out = pd.DataFrame(
        {
            "Reactant 1": df["Reactant_1_Name"].str.strip().map(SM_REACTANT_1),
            "Reactant 2": df["Reactant_2_Name"].str.strip().map(SM_REACTANT_2),
            "Ligand": df["Ligand_Short_Hand"].fillna("None").str.strip().map(SM_LIGAND),
            "Reagent": df["Reagent_1_Short_Hand"].fillna("None").str.strip().map(SM_REAGENT),
            "Solvent": df["Solvent_1_Short_Hand"].str.strip().map(SM_SOLVENT),
            "ligand_name": df["Ligand_Short_Hand"].fillna("None").str.strip(),
            "y": df["Product_Yield_PCT_Area_UV"].astype(float),
        }
    )
    assert out.notna().all().all(), "unmapped Suzuki-Miyaura component"
    return out


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "load_prospective",
    "load_ex12",
    "load_logp_pool",
    "load_bh",
    "load_sm",
    "BH_COMPONENTS",
    "SM_COMPONENTS",
    "BH_TEST_TRAIN_ROWS",
    "np",
]
