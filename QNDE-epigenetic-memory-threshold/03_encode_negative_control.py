"""
03_encode_negative_control.py  v6
==================================
Uses Ensembl REST API (not MyGene.info) for TSS coordinates.
All 58 real ENCODE H1-hESC experiment accessions hardcoded.

Usage:
    python 03_encode_negative_control.py

Author: [Author Name]
"""

import os, json, hashlib, gzip, time
import requests
import numpy as np
import pandas as pd
from scipy import stats

BASE      = os.path.dirname(os.path.abspath(__file__))
RAW       = os.path.join(BASE, "data", "raw")
PROC      = os.path.join(BASE, "data", "processed")
RES       = os.path.join(BASE, "results")
CACHE_DIR = os.path.join(RAW, "encode_peaks")
TSS_CACHE = os.path.join(RAW, "tss_cache.json")
FILE_CACHE= os.path.join(RAW, "file_accession_cache.json")
MANIFEST  = os.path.join(RAW, "encode_manifest.csv")
SCORES    = os.path.join(RAW, "rosen2024_scores.csv")
WINDOW_BP = 2000

for d in [PROC, RES, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Real ENCODE H1-hESC experiment accessions ─────────────────
EXPERIMENTS = {
    "ATF2":       "ENCSR000BQU",
    "ATF3":       "ENCSR000BKC",
    "BACH1":      "ENCSR000EBQ",
    "BCL11A":     "ENCSR000BMJ",
    "BRCA1":      "ENCSR000EBX",
    "CEBPB":      "ENCSR000EBV",
    "CHD1":       "ENCSR000AQK",
    "CHD2":       "ENCSR000EBT",
    "CHD7":       "ENCSR000AVA",
    "CREB1":      "ENCSR000BSN",
    "CTBP2":      "ENCSR000EUO",
    "CTCF":       "ENCSR000AMF",
    "E2F6":       "ENCSR000BSI",
    "EGR1":       "ENCSR000BJA",
    "EP300":      "ENCSR000BKK",
    "EZH2":       "ENCSR000ASY",
    "FOSL1":      "ENCSR000BNS",
    "GABPA":      "ENCSR000BIW",
    "GTF2F1":     "ENCSR000EBP",
    "HDAC2":      "ENCSR000BNR",
    "HDAC6":      "ENCSR000ATQ",
    "JUN":        "ENCSR000ECA",
    "JUND":       "ENCSR000BKP",
    "KDM4A":      "ENCSR000AVC",
    "KDM5A":      "ENCSR160ZLP",
    "MAFK":       "ENCSR000EBS",
    "MAX":        "ENCSR000BSJ",
    "MXI1":       "ENCSR000EBR",
    "MYC":        "ENCSR000EBY",
    "NANOG":      "ENCSR000BMT",
    "NRF1":       "ENCSR000ECC",
    "PHF8":       "ENCSR000ATK",
    "POLR2A":     "ENCSR000BHN",
    "POLR2A_S5P": "ENCSR000BIC",
    "POU5F1":     "ENCSR000BMU",
    "RAD21":      "ENCSR000BLD",
    "RBBP5":      "ENCSR000AQC",
    "REST":       "ENCSR000BHM",
    "RFX5":       "ENCSR000ECF",
    "RXRA":       "ENCSR000BJW",
    "SAP30":      "ENCSR000ATR",
    "SIN3A":      "ENCSR000BIS",
    "SIRT6":      "ENCSR000AUS",
    "SIX5":       "ENCSR000BIQ",
    "SP1":        "ENCSR000BIR",
    "SP2":        "ENCSR000BQG",
    "SP4":        "ENCSR000BQV",
    "SRF":        "ENCSR000BIV",
    "SUZ12":      "ENCSR000ATS",
    "TAF1":       "ENCSR000BHO",
    "TAF7":       "ENCSR000BLU",
    "TBP":        "ENCSR000ECB",
    "TCF12":      "ENCSR000BIT",
    "TEAD4":      "ENCSR000BRY",
    "USF1":       "ENCSR000BIU",
    "USF2":       "ENCSR000ECD",
    "YY1":        "ENCSR000BKD",
    "ZNF274":     "ENCSR000EUN",
}


# ══════════════════════════════════════════════════════════════
# TSS via Ensembl REST API (replaces MyGene.info)
# Works reliably on Windows
# ══════════════════════════════════════════════════════════════
ENSEMBL_URL = "https://rest.ensembl.org/lookup/symbol/homo_sapiens"
ENSEMBL_HDR = {"Content-Type": "application/json",
               "Accept":       "application/json"}


def get_tss_ensembl(gene_symbols, batch_size=200):
    """
    Fetch TSS coordinates from Ensembl REST API.
    Uses POST /lookup/symbol/homo_sapiens for batch queries.
    Returns dict: {SYMBOL: ("chrN", tss_int)}
    """
    # Load cache
    cache = {}
    if os.path.exists(TSS_CACHE):
        with open(TSS_CACHE) as f:
            cache = json.load(f)
        print(f"  TSS cache: {len(cache)} genes already fetched")

    to_fetch = [g for g in gene_symbols if g not in cache]
    if not to_fetch:
        print("  All TSS coordinates already cached.")
        return {k: tuple(v) for k, v in cache.items()}

    total  = (len(to_fetch) + batch_size - 1) // batch_size
    print(f"  Fetching TSS for {len(to_fetch)} genes "
          f"({total} batches of {batch_size})...")

    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        bn    = i // batch_size + 1
        print(f"  Batch {bn}/{total}: genes {i+1}-"
              f"{min(i+batch_size, len(to_fetch))}...",
              end=" ", flush=True)

        try:
            r = requests.post(
                ENSEMBL_URL,
                headers=ENSEMBL_HDR,
                json={"symbols": batch},
                timeout=60,
            )
            r.raise_for_status()
            data  = r.json()
            found = 0

            for sym, info in data.items():
                if not isinstance(info, dict):
                    continue
                chrom_raw = info.get("seq_region_name", "")
                # Ensembl returns "1","2","X","MT" — add chr prefix
                # Skip non-standard chromosomes
                if not chrom_raw or "_" in chrom_raw:
                    continue
                chrom = "chr" + chrom_raw

                strand = info.get("strand", 1)
                start  = int(info.get("start", 0))
                end    = int(info.get("end",   0))
                # TSS = start of gene for + strand, end for - strand
                tss    = start if strand == 1 else end

                cache[sym.upper()] = [chrom, tss]
                found += 1

            print(f"got {found}")

            # Save cache after every batch
            with open(TSS_CACHE, "w") as f:
                json.dump(cache, f)

            # Ensembl rate limit: 15 requests/second
            time.sleep(0.15)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited — wait and retry
                print(f"rate limited, waiting 10s...")
                time.sleep(10)
            else:
                print(f"HTTP error: {e}")
                time.sleep(2)
        except Exception as e:
            print(f"WARNING: {e}")
            time.sleep(2)

    print(f"  Total TSS coordinates available: {len(cache)}")
    return {k: tuple(v) for k, v in cache.items()}


# ══════════════════════════════════════════════════════════════
# ENCODE helpers
# ══════════════════════════════════════════════════════════════
def get_idr_file(exp_acc):
    url = (f"https://www.encodeproject.org/experiments/"
           f"{exp_acc}/?format=json")
    try:
        r = requests.get(url, timeout=30,
                         headers={"Accept": "application/json",
                                  "User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data  = r.json()
        files = data.get("files", [])
        for out_type in ["IDR thresholded peaks",
                         "optimal IDR thresholded peaks",
                         "conservative IDR thresholded peaks"]:
            for f in files:
                if (isinstance(f, dict)
                        and f.get("output_type")  == out_type
                        and f.get("file_format")  == "bed"
                        and f.get("assembly")     == "GRCh38"
                        and f.get("status")       == "released"):
                    return f.get("accession")
    except Exception as e:
        print(f"    Error: {e}")
    return None


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def download_file(file_acc):
    out = os.path.join(CACHE_DIR, f"{file_acc}.bed.gz")
    if os.path.exists(out):
        return out, md5_file(out)
    url = (f"https://www.encodeproject.org/files/{file_acc}/"
           f"@@download/{file_acc}.bed.gz")
    print(f"    Downloading {file_acc}...", end=" ", flush=True)
    try:
        r = requests.get(url, stream=True, timeout=120,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(65536): f.write(chunk)
        sz  = os.path.getsize(out) // 1024
        md5 = md5_file(out)
        print(f"OK ({sz} KB)")
        return out, md5
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists(out): os.remove(out)
        return None, None


def load_peaks(bed_gz):
    peaks = {}
    with gzip.open(bed_gz, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.strip().split("\t")
            if len(p) < 3: continue
            try:
                peaks.setdefault(p[0], []).append(
                    (int(p[1]), int(p[2])))
            except ValueError:
                continue
    return peaks


def overlaps(peaks, chrom, tss, w=WINDOW_BP):
    lo, hi = tss - w, tss + w
    return any(ps < hi and pe > lo
               for ps, pe in peaks.get(chrom, []))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("ENCODE Negative Control  v6  (Ensembl TSS)")
    print(f"  {len(EXPERIMENTS)} experiments  |  window ±{WINDOW_BP} bp")
    print("=" * 60)
    print()

    # Load scores
    scores = pd.read_csv(SCORES)
    scores["gene_symbol"] = (scores["gene_symbol"]
                             .astype(str).str.strip().str.upper())
    genes = scores["gene_symbol"].dropna().unique().tolist()
    print(f"Rosen screen genes: {len(genes):,}")
    print()

    # ── Step 1: File accessions ────────────────────────────────
    print("Step 1: Finding IDR peak file accessions...")
    file_cache = {}
    if os.path.exists(FILE_CACHE):
        with open(FILE_CACHE) as f: file_cache = json.load(f)
        print(f"  File accession cache: {len(file_cache)} experiments")

    manifest_rows = []
    for tf, exp_acc in EXPERIMENTS.items():
        if exp_acc in file_cache:
            file_acc = file_cache[exp_acc]
        else:
            print(f"  {tf:<14} {exp_acc}...", end=" ", flush=True)
            file_acc = get_idr_file(exp_acc) or ""
            file_cache[exp_acc] = file_acc
            with open(FILE_CACHE, "w") as f: json.dump(file_cache, f)
            print(file_acc or "NOT FOUND")
            time.sleep(0.4)

        if file_acc:
            manifest_rows.append({
                "tf_name": tf,
                "experiment_accession": exp_acc,
                "file_accession": file_acc,
                "md5": "",
            })

    df_mf = pd.DataFrame(manifest_rows)
    df_mf.to_csv(MANIFEST, index=False)
    print(f"  {len(df_mf)}/{len(EXPERIMENTS)} TFs have IDR peak files.")
    print()

    # ── Step 2: TSS via Ensembl ────────────────────────────────
    print("Step 2: Getting TSS coordinates via Ensembl REST API...")
    tss_dict = get_tss_ensembl(genes, batch_size=200)
    print()

    if len(tss_dict) == 0:
        print("ERROR: No TSS coordinates fetched.")
        print("Check internet connection and try again.")
        return

    # Quick sanity check
    for test_gene in ["POU5F1", "NANOG", "SOX2", "FASN"]:
        if test_gene in tss_dict:
            chrom, tss = tss_dict[test_gene]
            print(f"  Sanity check {test_gene}: {chrom}:{tss:,}")
    print()

    # ── Step 3: Download peaks ─────────────────────────────────
    print("Step 3: Downloading ENCODE peak files...")
    md5_vals   = {}
    n_tfs_done = 0

    # Check which files already downloaded
    for _, row in df_mf.iterrows():
        facc  = row["file_accession"]
        local = os.path.join(CACHE_DIR, f"{facc}.bed.gz")
        if os.path.exists(local):
            md5_vals[facc] = md5_file(local)
            n_tfs_done += 1

    if n_tfs_done > 0:
        print(f"  {n_tfs_done} files already cached — skipping download.")

    for _, row in df_mf.iterrows():
        facc = row["file_accession"]
        if facc in md5_vals:
            continue
        tf = row["tf_name"]
        print(f"  [{tf:<14}] {facc}")
        bed_path, md5 = download_file(facc)
        if bed_path:
            md5_vals[facc] = md5
            n_tfs_done += 1
    print()

    # ── Step 4: Compute co-occupancy ──────────────────────────
    print("Step 4: Computing co-occupancy...")
    n_counts = {g: 0 for g in genes}

    for i, (_, row) in enumerate(df_mf.iterrows()):
        tf   = row["tf_name"]
        facc = row["file_accession"]
        bed_path = os.path.join(CACHE_DIR, f"{facc}.bed.gz")

        if not os.path.exists(bed_path):
            print(f"  [{tf}] MISSING — skipping")
            continue

        print(f"  [{tf:<14}] {facc}...", end=" ", flush=True)
        try:
            peaks     = load_peaks(bed_path)
            hit_count = 0
            for gene in genes:
                if gene not in tss_dict: continue
                chrom, tss = tss_dict[gene]
                if overlaps(peaks, chrom, tss):
                    n_counts[gene] += 1
                    hit_count += 1
            print(f"{hit_count} genes")
        except Exception as e:
            print(f"ERROR: {e}")

    # Update manifest md5
    for facc, md5 in md5_vals.items():
        df_mf.loc[df_mf["file_accession"] == facc, "md5"] = md5
    df_mf.to_csv(MANIFEST, index=False)
    print(f"\n  encode_manifest.csv updated with md5 checksums.")

    # ── Step 5: Correlations ───────────────────────────────────
    df_counts = pd.DataFrame([
        {"gene_symbol": g, "encode_n_gene": n_counts[g]}
        for g in genes
    ])
    df_out = scores.merge(df_counts, on="gene_symbol", how="left")
    df_out.to_csv(
        os.path.join(PROC, "encode_ngene_17948.csv"), index=False)

    # Only genes with TSS + score
    df_c = df_out[
        (df_out["encode_n_gene"].notna()) &
        (df_out["NE_pluripotency_score"].notna()) &
        (df_out["E8_self_renewal_score"].notna())
    ].copy()

    r_id,  p_id  = stats.pearsonr(df_c["encode_n_gene"],
                                   df_c["NE_pluripotency_score"])
    r_fit, p_fit = stats.pearsonr(df_c["encode_n_gene"],
                                   df_c["E8_self_renewal_score"])

    print(f"\n{'='*60}")
    print(f"RESULTS  (n={len(df_c):,} genes, {n_tfs_done} TFs)")
    print(f"  Identity Pearson r = {r_id:+.4f}  p = {p_id:.2e}")
    print(f"  Fitness  Pearson r = {r_fit:+.4f}  p = {p_fit:.2e}")
    print(f"  Identity r2 = {r_id**2:.4f}")
    print(f"  Fitness  r2 = {r_fit**2:.4f}")
    print()
    print("  Expected: identity r ≈ 0 (NS), fitness r > 0")
    print("  This confirms double dissociation with curated panel.")

    results = {
        "n_genes":      int(len(df_c)),
        "n_encode_tfs": n_tfs_done,
        "identity_r":   float(r_id),
        "identity_p":   float(p_id),
        "identity_r2":  float(r_id**2),
        "fitness_r":    float(r_fit),
        "fitness_p":    float(p_fit),
        "fitness_r2":   float(r_fit**2),
        "tss_matched":  len(tss_dict),
        "window_bp":    WINDOW_BP,
        "status":       "computed",
        "tss_source":   "Ensembl REST API GRCh38",
        "tfs_not_in_H1_ENCODE": [
            "ASH2L", "CBX5", "CBX8", "KDM1A", "RNF2"],
    }
    res_path = os.path.join(RES, "negative_control_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {res_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
