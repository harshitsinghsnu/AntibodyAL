"""
Computes ground-truth epitope (antigen-side) and paratope (antibody-side)
contact residues directly from the deposited crystal/model structures, using
a standard heavy-atom distance cutoff (4.5 A - the conventional definition
of a structural "contact residue" in antibody-antigen interface studies,
e.g. Ofran/Schueler-Furman-style contact mapping). This is the actual
ground truth (the deposited 3D coordinates), not a transcription from any
single paper's prose, so it generalizes to all 6 antigen systems including
SARS-CoV-2 (AAYL49-52, no literature provided) and influenza (3GBN, 4FQI,
where the literature - Ekiert et al. 2009 Science and Bangaru et al. 2022
Frontiers in Virology - is used in REPORT.md as independent confirmation,
not a substitute for this geometric ground truth).

Antigen/antibody chain assignments (from each PDB's COMPND records):
    3GBN: antigen=A,B (HA1+HA2, CR6261 target);  antibody H=heavy, L=light
    4FQI: antigen=A,B (HA1+HA2, CR9114 target);  antibody H=heavy, L=light
    AAYL49/50/51/52: antigen=A (SARS-CoV-2 Spike HR2 peptide); antibody B=VH, C=VL
"""
import json
from pathlib import Path

import numpy as np
import pymol2

STRUCT_DIR = Path(r"D:\explainable_AL\data\structures")
OUT_PATH = Path(r"D:\explainable_AL\data\structures\ground_truth_contacts.json")
CONTACT_CUTOFF = 4.5  # Angstrom, heavy-atom min-distance, standard interface contact definition

STRUCTURES = {
    "3GBN":  {"file": "3GBN.pdb",      "antigen_chains": ["A", "B"], "heavy_chain": "H", "light_chain": "L"},
    "4FQI":  {"file": "4FQI.pdb",      "antigen_chains": ["A", "B"], "heavy_chain": "H", "light_chain": "L"},
    "AAYL49": {"file": "AAYL49_bca.pdb", "antigen_chains": ["A"], "heavy_chain": "B", "light_chain": "C"},
    "AAYL50": {"file": "AAYL50_bca.pdb", "antigen_chains": ["A"], "heavy_chain": "B", "light_chain": "C"},
    "AAYL51": {"file": "AAYL51_bca.pdb", "antigen_chains": ["A"], "heavy_chain": "B", "light_chain": "C"},
    "AAYL52": {"file": "AAYL52_bca.pdb", "antigen_chains": ["A"], "heavy_chain": "B", "light_chain": "C"},
}


def get_contact_residues(p2, antibody_sel: str, antigen_sel: str) -> tuple[list, list]:
    """Returns (paratope_residues, epitope_residues) as lists of
    {'chain', 'resi', 'resn'} dicts, using a heavy-atom min-distance cutoff."""
    p2.cmd.select("_ab_contact", f"({antibody_sel}) within {CONTACT_CUTOFF} of ({antigen_sel})")
    p2.cmd.select("_ag_contact", f"({antigen_sel}) within {CONTACT_CUTOFF} of ({antibody_sel})")

    paratope, epitope = [], []
    p2.cmd.iterate("_ab_contact and name CA+CB",
                    "paratope.append({'chain': chain, 'resi': resi, 'resn': resn})",
                    space={"paratope": paratope})
    p2.cmd.iterate("_ag_contact and name CA+CB",
                    "epitope.append({'chain': chain, 'resi': resi, 'resn': resn})",
                    space={"epitope": epitope})

    # de-dup (CA and CB of same residue both selected) and sort; resi may carry
    # a PDB insertion code (e.g. "52A"), so sort by the leading numeric part
    def sort_key(r):
        digits = "".join(c for c in r["resi"] if c.isdigit())
        return (r["chain"], int(digits) if digits else 0, r["resi"])

    def dedup(rs):
        seen = {}
        for r in rs:
            seen[(r["chain"], r["resi"])] = r
        return sorted(seen.values(), key=sort_key)

    return dedup(paratope), dedup(epitope)


def main():
    out = {}
    with pymol2.PyMOL() as p2:
        for name, cfg in STRUCTURES.items():
            path = STRUCT_DIR / cfg["file"]
            p2.cmd.reinitialize()
            p2.cmd.load(str(path), "struct")

            antigen_sel = "struct and chain " + "+".join(cfg["antigen_chains"])
            heavy_sel = f"struct and chain {cfg['heavy_chain']}"
            light_sel = f"struct and chain {cfg['light_chain']}"
            antibody_sel = f"struct and chain {cfg['heavy_chain']}+{cfg['light_chain']}"

            paratope, epitope = get_contact_residues(p2, antibody_sel, antigen_sel)
            h_paratope, _ = get_contact_residues(p2, heavy_sel, antigen_sel)
            l_paratope, _ = get_contact_residues(p2, light_sel, antigen_sel)

            out[name] = {
                "epitope_residues": epitope,
                "paratope_residues": paratope,
                "paratope_heavy_chain": h_paratope,
                "paratope_light_chain": l_paratope,
                "contact_cutoff_angstrom": CONTACT_CUTOFF,
            }
            print(f"{name}: {len(epitope)} epitope residues, {len(paratope)} paratope residues "
                  f"({len(h_paratope)} heavy / {len(l_paratope)} light)")
            print(f"  epitope: {[(r['chain'], r['resi'], r['resn']) for r in epitope]}")
            print(f"  paratope (H): {[(r['resi'], r['resn']) for r in h_paratope]}")
            print(f"  paratope (L): {[(r['resi'], r['resn']) for r in l_paratope]}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
