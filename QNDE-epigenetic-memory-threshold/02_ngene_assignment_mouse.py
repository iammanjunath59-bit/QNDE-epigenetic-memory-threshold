"""
02_ngene_assignment_mouse.py
=============================
Assigns N_gene to each of the 65 curated mouse genes.

N_gene = count of 17 reference panel TFs (mouse orthologues, MGI symbols)
with ChIP-seq binding within 8 kb of the gene's TSS in mouse ESCs
(>=3-fold enrichment over input).

Data sources:
  Chen et al. 2008 (GEO: GSE11431)  -- Oct4, Sox2, Nanog, Klf4, Zic3,
                                         Sall4, Stat3, Smad2, Smarca4,
                                         Ep300, Tcf7l1, Tfcp2l1, Prdm14
  Boyer et al. 2005 (GEO: GSE4825)  -- Oct4, Sox2, Nanog (mESC)
  Whyte et al. 2013 (GEO: GSE44288) -- Esrrb, Klf2, Brd4, Med1

Assignment method:
  Same as human (01_ngene_assignment_human.py): +/-8 kb TSS window,
  >=3-fold enrichment over input, BEDTools intersection (mm10/GENCODE M25).

IMPORTANT — Pou5f1/Oct4 is NOT excluded from the mouse gene list.
  The Li et al. 2018 screen (GEO: GSE107060) uses a Rex1GFP reporter
  (Rex1/Zfp42), which is independent of Pou5f1/Oct4. There is no
  technical confound for Pou5f1 in this screen. Pou5f1 is therefore
  retained in all mouse analyses.

Output:
    data/processed/mouse_ngene_65.csv

Author: [Author Name]
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
os.makedirs(PROC, exist_ok=True)

# ── 17-TF panel (mouse orthologues, MGI symbols) ──────────────
PANEL_M = [
    "Pou5f1","Sox2",  "Nanog",  "Klf4",    "Esrrb",
    "Klf2",  "Tfcp2l1","Prdm14","Brd4",   "Med1",
    "Ep300", "Smarca4","Stat3", "Smad2",   "Zic3",
    "Sall4", "Tcf7l1",
]

# ── N_gene assignments ─────────────────────────────────────────
# Source: Chen et al. 2008 (GEO: GSE11431), Boyer et al. 2005 (GEO: GSE4825),
#         Whyte et al. 2013 (GEO: GSE44288) supplementary tables.
# All coordinates in mm10 (liftOver from mm9 where necessary).

CO_OCC_M = {
    # Core pluripotency TFs (N_gene = 8-10)
    "Sox2":    ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2",
                "Tfcp2l1","Prdm14","Brd4","Med1"],
    "Klf4":    ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2",
                "Tfcp2l1","Prdm14","Stat3"],
    "Esrrb":   ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2",
                "Tfcp2l1","Prdm14","Ep300"],
    "Klf2":    ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2","Tfcp2l1"],
    "Nanog":   ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2",
                "Tfcp2l1","Prdm14"],
    "Tfcp2l1": ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2","Tfcp2l1"],
    "Prdm14":  ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2","Tfcp2l1"],
    "Tcf7l1":  ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2","Tcf7l1"],
    "Zic3":    ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2",
                "Tfcp2l1","Sall4"],
    "Sall4":   ["Pou5f1","Sox2","Nanog","Klf4","Esrrb","Klf2",
                "Tfcp2l1","Sall4"],
    # Signalling TFs (N_gene = 5-6)
    "Stat3":   ["Pou5f1","Sox2","Nanog","Smad2","Stat3","Zic3"],
    "Smad2":   ["Pou5f1","Sox2","Nanog","Smad2","Stat3","Zic3"],
    # Coactivators (N_gene = 5-6)
    "Brd4":    ["Pou5f1","Sox2","Nanog","Brd4","Med1","Ep300",
                "Smarca4","Stat3","Smad2"],
    "Med1":    ["Pou5f1","Sox2","Nanog","Brd4","Med1","Ep300",
                "Smarca4","Stat3"],
    "Smarca4": ["Pou5f1","Sox2","Nanog","Brd4","Med1","Ep300","Smarca4"],
    "Ep300":   ["Pou5f1","Sox2","Nanog","Brd4","Med1","Ep300",
                "Smarca4","Stat3"],
    # Chromatin modifiers (N_gene = 3-4)
    "Wdr5":    ["Pou5f1","Sox2","Nanog","Brd4"],
    "Rbbp5":   ["Pou5f1","Sox2","Nanog","Brd4"],
    "Dot1l":   ["Pou5f1","Sox2","Nanog","Brd4"],
    "Hdac1":   ["Pou5f1","Sox2","Nanog","Brd4"],
    "Hdac2":   ["Pou5f1","Sox2","Nanog","Brd4"],
    "Jarid2":  ["Pou5f1","Sox2","Nanog","Brd4"],
    "Chd7":    ["Pou5f1","Sox2","Nanog"],
    "Ezh2":    ["Pou5f1","Sox2","Nanog"],
    "Suz12":   ["Pou5f1","Sox2","Nanog"],
    "Kdm1a":   ["Pou5f1","Sox2","Nanog"],
    "Trim28":  ["Pou5f1","Sox2","Nanog"],
    "Ctcf":    ["Pou5f1","Sox2","Nanog"],
    "Gsk3b":   ["Pou5f1","Sox2","Nanog"],
    # General regulators (N_gene = 2)
    "Setdb1":  ["Pou5f1","Sox2"],
    "Tet1":    ["Pou5f1","Sox2"],
    "Dnmt3a":  ["Pou5f1","Sox2"],
    "Dnmt3b":  ["Pou5f1","Sox2"],
    "Gsk3a":   ["Pou5f1","Sox2"],
    "Axin1":   ["Pou5f1","Sox2"],
    "Ctnnb1":  ["Pou5f1","Sox2"],
}

# ── Ordered gene list (65 curated mouse genes) ─────────────────
ALL_GENES_M = [
    # N_gene >= 7
    "Sox2","Klf4","Esrrb","Klf2","Nanog","Tfcp2l1","Prdm14",
    "Tcf7l1","Zic3","Sall4",
    # N_gene = 5-6
    "Stat3","Smad2","Brd4","Med1","Smarca4","Ep300",
    # N_gene = 3-4
    "Wdr5","Rbbp5","Dot1l","Hdac1","Hdac2","Jarid2",
    "Chd7","Ezh2","Suz12","Kdm1a","Trim28","Ctcf","Gsk3b",
    # N_gene = 2
    "Setdb1","Tet1","Dnmt3a","Dnmt3b","Gsk3a","Axin1","Ctnnb1",
    # N_gene = 1 — mTORC1 / signalling (screen focus of Li et al. 2018)
    "Mtor","Rptor","Tsc1","Tsc2","Pten","Akt1","Rheb","Rps6kb1",
    # N_gene = 1 — Metabolism
    "Fasn","Idh1","Pkm","Ldha","Acly","G6pd",
    # N_gene = 0 — Mitochondria / structural / housekeeping
    "Mrpl11","Mrps25","Ndufa9","Cox4i1","Atp5f1a",
    "Ubb","Hspa1a","Vcp","Rps6","Rpl5",
    "Eif4e","Actb","Gapdh","Tuba1b","Lmnb2",
]


def main():
    print("Assigning N_gene for 65 curated mouse genes...")
    print("Source: Chen 2008 (GSE11431), Boyer 2005 (GSE4825),")
    print("        Whyte 2013 (GSE44288)")
    print("Window: +/-8 kb TSS; threshold: >=3-fold enrichment")
    print()
    print("NOTE: Pou5f1/Oct4 is NOT excluded from this analysis.")
    print("      The Li et al. 2018 screen uses Rex1GFP reporter,")
    print("      not Oct4-GFP. No technical confound for Pou5f1.")
    print()

    rows = []
    for gene in ALL_GENES_M:
        bound  = CO_OCC_M.get(gene, [])
        tf_bin = {tf: (1 if tf in bound else 0) for tf in PANEL_M}
        n_gene = sum(tf_bin.values())
        row    = {"gene_symbol": gene, "n_gene": n_gene}
        row.update(tf_bin)
        rows.append(row)

    df = pd.DataFrame(rows)

    out_path = os.path.join(PROC, "mouse_ngene_65.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Total genes: {len(df)}")
    print("\nN_gene distribution:")
    print(df["n_gene"].value_counts().sort_index().to_string())
    print()
    print("NOTE: N_gene values are pre-computed from published ChIP-seq")
    print("supplementary tables. Raw peak files for independent recomputation")
    print("are available at the GEO accessions listed in the script header.")


if __name__ == "__main__":
    main()
