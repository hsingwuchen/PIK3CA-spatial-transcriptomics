"""
Visium HD FF Human Breast Cancer (DCIS) -- targeted mast-cell pipeline
Dataset : visium-hd-cytassist-gene-expression-libraries-human-breast-cancer-ff-ultima-4
          Space Ranger v4.0.1, slide H1-649ZJJX

Cell type identification:
  Cancer-enriched bins (ca_high) are still defined by epithelial-rich Leiden
  clusters on a curated TME marker panel.

  Mast-cell-enriched bins (mc_high) are NOT defined by Leiden clusters. Mast
  cells are expected to be rare in this Visium HD 8 um bin-level analysis, so
  mc_high is defined with targeted TPSAB1/TPSB2/CPA3 co-expression plus a
  high mast-marker score threshold across UMI-filtered tissue bins.

  Lipid-positive cancer bins are also thresholded by cancer-bin logCPM
  quantiles instead of using expression > 0, which is too sensitive to ambient
  RNA and low-UMI edge effects.

  References:
    - Traag VA et al. (2019) Sci Rep 9:5233  [Leiden algorithm]
    - Luecken MD & Theis FJ (2019) Mol Syst Biol 15:e8746  [best-practice scRNA-seq workflow]
    - Wolf FA et al. (2018) Genome Biol 19:15  [SCANPY]

Outputs (D:/VisiumHD_BC_DCIS_leiden/):
  Figure 0: VisiumHD_FF_leiden_clusters.png   (spatial cluster map + UMAP)
  Figure 1: VisiumHD_FF_PIK3CA_GS_comparison.png
  Figure 2: VisiumHD_FF_4panel_fixed.png
  Figure 3: VisiumHD_FF_lipid7_panel.png
  Table:    lipid_proximity_stats_full.tsv
  Figure 4: VisiumHD_FF_proximity_summary.png
"""

import h5py, numpy as np, pandas as pd, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc_colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact, mannwhitneyu
import scanpy as sc

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DATA = os.environ.get("BASE_DATA", "/content/drive/MyDrive/VisiumHD/DCIS")
H5        = os.environ.get("H5", f"{BASE_DATA}/feature_slice.h5")
SIGF      = os.environ.get("SIGF", "/content/drive/MyDrive/VisiumHD/reference/pik3ca_gs_with_symbols.csv")
MPP       = 0.5468406924593424   # um/pixel full-res, slide H1-649ZJJX

MAST_HIGH_CONF_PREV_RANGE = (0.002, 0.005)  # high-confidence: top 0.2-0.5%
MAST_SENSITIVE_PREV_RANGE = (0.005, 0.010)  # sensitive: top 0.5-1.0%
MAST_HIGH_CONF_PREV = 0.005                # default within high-confidence range
MAST_SENSITIVE_PREV = 0.010                # default within sensitive range
ACTIVE_MAST_MASK = os.environ.get("ACTIVE_MAST_MASK", "sensitive")  # "high_confidence" or "sensitive"
OUT_ROOT = os.environ.get("TARGETED_MAST_OUT_ROOT", "/content/drive/MyDrive/VisiumHD/DCIS/targeted_mast_compare")
OUT = os.environ.get("TARGETED_MAST_OUT", f"{OUT_ROOT}/{ACTIVE_MAST_MASK}")
LIPID_POS_QUANTILE  = 0.90   # top 10% of cancer-bin logCPM for each lipid gene
LIPID_POS_QUANTILES = [0.95, 0.90, 0.75, 0.50]  # sensitivity: top 5%, 10%, 25%, 50%

os.makedirs(OUT, exist_ok=True)

# ── Curated gene panel for Leiden clustering ───────────────────────────────────
# ~110 genes covering all major TME cell types in breast cancer
CLUSTER_GENES = sorted(set([
    # Luminal epithelial / cancer
    'EPCAM', 'KRT8', 'KRT18', 'KRT19', 'KRT7', 'CDH1', 'MUC1', 'CLDN4', 'CLDN3',
    'FOXA1', 'GATA3', 'ESR1', 'PGR', 'ERBB2', 'CCND1', 'AR',
    # Myoepithelial (DCIS boundary)
    'KRT5', 'KRT14', 'TP63', 'CNN1', 'OXTR',
    # Mast cells
    'TPSAB1', 'TPSB2', 'CPA3', 'KIT', 'MS4A2', 'HDC', 'FCER1A',
    # T cells / NK
    'CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD4', 'GZMB', 'PRF1', 'FOXP3',
    'NCAM1', 'KLRB1', 'NKG7', 'GNLY',
    # B cells / Plasma
    'MS4A1', 'CD19', 'CD79A', 'IGHG1', 'IGKC', 'MZB1',
    # Macrophages / Monocytes
    'CD68', 'CSF1R', 'AIF1', 'CD14', 'LYZ', 'MRC1', 'CD163',
    # Dendritic cells
    'CLEC9A', 'CLEC4C', 'LILRA4', 'IRF8',
    # CAFs / Fibroblasts
    'COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'FAP', 'PDPN', 'POSTN', 'FN1',
    # Pericytes
    'RGS5', 'MCAM', 'PDGFRB', 'TAGLN',
    # Endothelial
    'PECAM1', 'VWF', 'CDH5', 'ENG', 'CLDN5', 'LYVE1',
    # Adipocytes
    'FABP4', 'PLIN1', 'ADIPOQ', 'LEP',
    # Lipid metabolism
    'ALOX5', 'ALOX5AP', 'PTGS1', 'PTGS2', 'PLA2G2A', 'PLA2G4A',
    'ELOVL5', 'ELOVL2', 'HPGDS', 'PTGES', 'PTGES2', 'PTGIS',
    # Proliferation
    'MKI67', 'TOP2A', 'PCNA', 'CCNB1', 'CDK1',
    # Stress / Hypoxia
    'HIF1A', 'VEGFA',
]))

# Marker sets for cluster annotation
CA_MARKERS    = ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'KRT7', 'CDH1', 'MUC1', 'CLDN4', 'CLDN3']
MC_MARKERS    = ['TPSAB1', 'TPSB2', 'CPA3']
FIBRO_MARKERS = ['FAP', 'PDPN', 'POSTN', 'DCN', 'LUM']
ENDO_MARKERS  = ['PECAM1', 'VWF', 'CDH5', 'LYVE1']
TCELL_MARKERS = ['CD3D', 'CD3E', 'CD8A', 'CD4', 'NKG7']
MACRO_MARKERS = ['CD68', 'CSF1R', 'AIF1', 'LYZ']
BCELL_MARKERS = ['MS4A1', 'CD79A']  # IGHG1/IGKC removed: too high ambient in breast tissue
MYOEPI_MARK   = ['KRT5', 'KRT14', 'TP63', 'CNN1']
ADIPO_MARKERS = ['FABP4', 'PLIN1', 'ADIPOQ']

MARKER_SETS = {
    'Cancer':       CA_MARKERS,
    'Mast':         MC_MARKERS,
    'Fibroblast':   FIBRO_MARKERS,
    'Endothelial':  ENDO_MARKERS,
    'T/NK cell':    TCELL_MARKERS,
    'Macrophage':   MACRO_MARKERS,
    'B/Plasma':     BCELL_MARKERS,
    'Myoepithelial': MYOEPI_MARK,
    'Adipocyte':    ADIPO_MARKERS,
}

# ── Load HDF5 ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading HDF5 ...")
hf = h5py.File(H5, 'r')
print(f"  Space Ranger : {hf.attrs.get('software_version', 'unknown')}")

gene_names = [x.decode() for x in hf['features']['name'][:]]
name2idx   = {n: i for i, n in enumerate(gene_names)}
print(f"  Genes : {len(gene_names):,}")

meta   = json.loads(hf.attrs['metadata_json'])
_m_tmp = hf['masks']['square_008um']
GRID   = max(int(_m_tmp['row'][:].max()), int(_m_tmp['col'][:].max())) + 1
print(f"  Grid  : {GRID}x{GRID}")

M_FULL = np.array(meta['transform_matrices']['spot_colrow_to_microscope_colrow'])

m = hf['masks']['square_008um']
tissue_mask = np.zeros((GRID, GRID), dtype=bool)
mr, mc_ = m['row'][:], m['col'][:]
valid_m = (mr < GRID) & (mc_ < GRID)
tissue_mask[mr[valid_m], mc_[valid_m]] = True
print(f"  Tissue bins : {tissue_mask.sum():,}")

ut = hf['umis']['total']
ur2, uc2, ud2 = ut['row'][:], ut['col'][:], ut['data'][:]
total_mat = np.zeros((GRID, GRID), dtype=np.float32)
vld = (ur2 >> 2 < GRID) & (uc2 >> 2 < GRID)
np.add.at(total_mat, (ur2[vld] >> 2, uc2[vld] >> 2), ud2[vld].astype(np.float32))
total_umi_all = float(ud2.sum())
print(f"  Total UMI : {total_umi_all:,.0f}")

# ── Helper functions ───────────────────────────────────────────────────────────
def read8(g):
    if g not in name2idx:
        return np.zeros((GRID, GRID), np.float32)
    key = str(name2idx[g])
    if key not in hf['feature_slices']:
        return np.zeros((GRID, GRID), np.float32)
    grp = hf['feature_slices'][key]
    r8 = grp['row'][:] >> 2
    c8 = grp['col'][:] >> 2
    mat = np.zeros((GRID, GRID), np.float32)
    v = (r8 < GRID) & (c8 < GRID)
    np.add.at(mat, (r8[v], c8[v]), grp['data'][v].astype(np.float32))
    return mat

def ln(mat):
    return np.log1p(np.where(total_mat > 0, mat / total_mat * 1e4, 0.)).astype(np.float32)

def fl(arr):
    return np.fliplr(arr)

def to_mic(rows, cols):
    rows = np.asarray(rows, float)
    cols = np.asarray(cols, float)
    pts  = np.stack([cols * 4 + 1.5, rows * 4 + 1.5, np.ones(len(rows))])
    mic  = M_FULL @ pts
    return mic[0], mic[1]

# ── Build expression matrix for clustering ────────────────────────────────────
print("\n" + "=" * 60)
umi_mask = tissue_mask & (total_mat >= 10)  # exclude in-tissue bins with < 10 UMI (no cell)

# Stroma pre-filter: exclude COL1A1/COL1A2/COL3A1-dominant bins from clustering
_col_score = np.zeros((GRID, GRID), dtype=np.float32)
for _g in ['COL1A1', 'COL1A2', 'COL3A1']:
    _col_score += ln(read8(_g))
_col_score /= 3
_col_thr = float(np.percentile(_col_score[umi_mask & (_col_score > 0)], 75))
_stroma_pre = umi_mask & (_col_score >= _col_thr)
cluster_mask = umi_mask & ~_stroma_pre
print(f"  Stroma pre-filter: removed {_stroma_pre.sum():,} high-collagen bins (COL1A1/COL1A2/COL3A1 ≥ P75)")
print(f"Building expression matrix ({cluster_mask.sum():,} bins x {len(CLUSTER_GENES)} genes) ...")

tissue_rows, tissue_cols = np.where(cluster_mask)
n_bins = len(tissue_rows)

X = np.zeros((n_bins, len(CLUSTER_GENES)), dtype=np.float32)
available_genes = []
for j, gene in enumerate(CLUSTER_GENES):
    mat = read8(gene)
    lnmat = ln(mat)
    X[:, j] = lnmat[tissue_rows, tissue_cols]
    if lnmat[tissue_mask].max() > 0:
        available_genes.append(gene)

print(f"  Available in dataset : {len(available_genes)}/{len(CLUSTER_GENES)} genes")

# ── Leiden clustering via scanpy ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Leiden clustering ...")

adata = sc.AnnData(X=sp.csr_matrix(X))
adata.var_names = CLUSTER_GENES

# Scale (z-score per gene) before PCA - data is already log-normalized
sc.pp.scale(adata, max_value=10)
print("  PCA ...")
sc.tl.pca(adata, n_comps=30, svd_solver='arpack', random_state=42)
print("  Building neighbor graph (k=15) ...")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=42)
print("  Running UMAP ...")
sc.tl.umap(adata, random_state=42)
print("  Leiden clustering (resolution=0.0003) ...")
sc.tl.leiden(adata, resolution=0.0003, random_state=42)

cluster_labels = adata.obs['leiden'].values.astype(str)
n_clusters = len(set(cluster_labels))
print(f"  Found {n_clusters} clusters")
for c in sorted(set(cluster_labels), key=int):
    print(f"    Cluster {c}: {(cluster_labels==c).sum():,} bins")

# ── Cluster annotation ────────────────────────────────────────────────────────
print("\nAnnotating clusters ...")

# Recompute raw log-CPM means for annotation (not scaled)
X_log = np.zeros((n_bins, len(CLUSTER_GENES)), dtype=np.float32)
for j, gene in enumerate(CLUSTER_GENES):
    mat = read8(gene)
    lnmat = ln(mat)
    X_log[:, j] = lnmat[tissue_rows, tissue_cols]

gene_to_j = {g: j for j, g in enumerate(CLUSTER_GENES)}

def marker_score(cluster_idx, markers):
    idxs = [gene_to_j[g] for g in markers if g in gene_to_j]
    if not idxs:
        return 0.
    return float(X_log[np.ix_(cluster_idx, idxs)].mean())

cluster_to_type = {}
cluster_scores  = {}
for c in sorted(set(cluster_labels), key=int):
    idx = np.where(cluster_labels == c)[0]
    scores = {ct: marker_score(idx, mkrs) for ct, mkrs in MARKER_SETS.items()}
    cluster_scores[c] = scores
    max_score = max(scores.values())
    if max_score > 0.05:
        best = max(scores, key=scores.get)
    else:
        best = 'Other'
    cluster_to_type[c] = best
    print(f"  Cluster {c:>3s} ({len(idx):>6,} bins)  -> {best:15s}  "
          f"[Cancer={scores['Cancer']:.3f}  Mast={scores['Mast']:.3f}  "
          f"Fibro={scores['Fibroblast']:.3f}  Endo={scores['Endothelial']:.3f}]")

hf.close()

# Identify cancer by top epithelial-rich Leiden cluster(s). Mast cells are
# intentionally handled below by targeted bin-level marker co-expression.
sorted_by_ca = sorted(cluster_scores, key=lambda c: cluster_scores[c]['Cancer'], reverse=True)
sorted_by_mc = sorted(cluster_scores, key=lambda c: cluster_scores[c]['Mast'],   reverse=True)

cancer_clusters = [c for c, t in cluster_to_type.items() if t == 'Cancer']
mast_clusters   = [c for c, t in cluster_to_type.items() if t == 'Mast']

if not cancer_clusters:
    print("  WARNING: No Cancer cluster found. Using top-2 clusters by CA score.")
    cancer_clusters = sorted_by_ca[:1]
    for c in cancer_clusters:
        cluster_to_type[c] = 'Cancer'
    print(f"  Using clusters: {cancer_clusters}")

if mast_clusters:
    print(f"  INFO: Leiden mast-like cluster(s) seen for QC only: {mast_clusters}")
else:
    top_mc_c = sorted_by_mc[0]
    top_mc_score = cluster_scores[top_mc_c]['Mast']
    print(f"  INFO: No Leiden mast cluster used. Top cluster by MC score is C{top_mc_c} ({top_mc_score:.3f}); "
          "mast mask will be marker-score based.")

# ── Map cluster labels to 2D grid ────────────────────────────────────────────
cluster_grid = np.full((GRID, GRID), -1, dtype=np.int16)
cluster_grid[tissue_rows, tissue_cols] = np.array([int(c) for c in cluster_labels])

ca_high = np.zeros((GRID, GRID), dtype=bool)
for c in cancer_clusters:
    ca_high |= (cluster_grid == int(c))
ca_high &= tissue_mask


# ── Reopen HDF5 for downstream analyses ───────────────────────────────────────
hf = h5py.File(H5, 'r')

# Targeted mast-cell marker masks on all UMI-filtered tissue bins, not Leiden
# clusters.
#   High-confidence: top 0.2-0.5% mast score and >=2 core markers positive.
#   Sensitive: top 0.5-1.0% mast score with >=2 core markers, or CPA3 plus
#              TPSAB1/TPSB2 co-expression.
MC_CORE = ['TPSAB1', 'TPSB2', 'CPA3']
MC_SUPPORT = ['KIT', 'MS4A2', 'HDC', 'HPGDS', 'FCER1A']
if not (MAST_HIGH_CONF_PREV_RANGE[0] <= MAST_HIGH_CONF_PREV <= MAST_HIGH_CONF_PREV_RANGE[1]):
    raise ValueError("MAST_HIGH_CONF_PREV must be between 0.002 and 0.005")
if not (MAST_SENSITIVE_PREV_RANGE[0] <= MAST_SENSITIVE_PREV <= MAST_SENSITIVE_PREV_RANGE[1]):
    raise ValueError("MAST_SENSITIVE_PREV must be between 0.005 and 0.010")
if ACTIVE_MAST_MASK not in {"high_confidence", "sensitive"}:
    raise ValueError("ACTIVE_MAST_MASK must be 'high_confidence' or 'sensitive'")

mc_mats = {g: ln(read8(g)) for g in sorted(set(MC_CORE + MC_SUPPORT))}
mast_score = np.zeros((GRID, GRID), dtype=np.float32)
mast_n_pos = np.zeros((GRID, GRID), dtype=np.int16)
for g in MC_CORE:
    mast_score += mc_mats[g]
    mast_n_pos += (mc_mats[g] > 0)
mast_score /= len(MC_CORE)

mast_score_vals = mast_score[umi_mask]
mast_thr_high_min = float(np.quantile(mast_score_vals, 1 - MAST_HIGH_CONF_PREV_RANGE[0]))
mast_thr_high = float(np.quantile(mast_score_vals, 1 - MAST_HIGH_CONF_PREV))
mast_thr_sensitive_min = float(np.quantile(mast_score_vals, 1 - MAST_SENSITIVE_PREV_RANGE[0]))
mast_thr_sensitive = float(np.quantile(mast_score_vals, 1 - MAST_SENSITIVE_PREV))
mc_high_conf = (
    umi_mask &
    (mast_n_pos >= 2) &
    (mast_score >= mast_thr_high)
)
mc_sensitive = umi_mask & (
    ((mast_n_pos >= 2) & (mast_score >= mast_thr_sensitive)) |
    ((mc_mats['CPA3'] > 0) & ((mc_mats['TPSAB1'] > 0) | (mc_mats['TPSB2'] > 0)))
)
mast_mask_defs = {
    "high_confidence": {
        "mask": mc_high_conf,
        "label": f"high-confidence top {MAST_HIGH_CONF_PREV*100:.1f}%",
        "rule": f"top {MAST_HIGH_CONF_PREV*100:.1f}% mast score and >=2 TPSAB1/TPSB2/CPA3 markers",
    },
    "sensitive": {
        "mask": mc_sensitive,
        "label": f"sensitive top {MAST_SENSITIVE_PREV*100:.1f}% / co-expression",
        "rule": f"top {MAST_SENSITIVE_PREV*100:.1f}% mast score with >=2 core markers, or CPA3 plus TPSAB1/TPSB2",
    },
}
mc_high = mast_mask_defs[ACTIVE_MAST_MASK]["mask"]
active_mast_label = mast_mask_defs[ACTIVE_MAST_MASK]["label"]
active_mast_rule = mast_mask_defs[ACTIVE_MAST_MASK]["rule"]

def cancer_quantile_positive(mat, q=LIPID_POS_QUANTILE):
    vals = mat[ca_high]
    thr = float(np.quantile(vals, q)) if vals.size else np.inf
    mask = ca_high & (mat > 0) & (mat >= thr)
    return mask, thr

alox5_mat = ln(read8('ALOX5'))
al_ca, alox5_thr = cancer_quantile_positive(alox5_mat)

print(f"\n  ca_high (Leiden) : {ca_high.sum():,} ({ca_high.sum()/cluster_mask.sum()*100:.1f}% of non-stromal tissue)")
print(f"  mc_high_conf     : {mc_high_conf.sum():,} ({mc_high_conf.sum()/umi_mask.sum()*100:.2f}% of UMI-filtered tissue; "
      f"default top {MAST_HIGH_CONF_PREV*100:.1f}%, threshold={mast_thr_high:.3f}; "
      f"top {MAST_HIGH_CONF_PREV_RANGE[0]*100:.1f}% threshold={mast_thr_high_min:.3f})")
print(f"  mc_sensitive     : {mc_sensitive.sum():,} ({mc_sensitive.sum()/umi_mask.sum()*100:.2f}% of UMI-filtered tissue; "
      f"default top {MAST_SENSITIVE_PREV*100:.1f}%, threshold={mast_thr_sensitive:.3f}; "
      f"top {MAST_SENSITIVE_PREV_RANGE[0]*100:.1f}% threshold={mast_thr_sensitive_min:.3f})")
print(f"  mc_high primary  : {ACTIVE_MAST_MASK} ({active_mast_label})")
print(f"  al_ca            : {al_ca.sum():,} (ALOX5 logCPM >= cancer-bin Q{int(LIPID_POS_QUANTILE*100)}={alox5_thr:.3f})")

# ── Color definitions ─────────────────────────────────────────────────────────
HEX = {
    'cancer':  '#2272C3', 'mc':     '#E63232', 'ALOX5':   '#00E5D0',
    'PTGS1':   '#50DC3C', 'PTGS2':  '#FF32AA', 'PLA2G2A': '#FF8C00',
    'PLA2G4A': '#B450FF', 'ELOVL5': '#FFD700', 'ELOVL2':  '#64B4FF',
}
CLUSTER_COLORS = [
    '#E41A1C','#377EB8','#4DAF4A','#984EA3','#FF7F00',
    '#A65628','#F781BF','#999999','#66C2A5','#FC8D62',
    '#8DA0CB','#E78AC3','#A6D854','#FFD92F','#E5C494',
    '#B3B3B3','#8DD3C7','#FFFFB3','#BEBADA','#FB8072',
]

def make_cmap(hex_color, n=256):
    r, g, b = mc_colors.to_rgb(hex_color)
    c = [(r, g, b, 0)] + [(r, g, b, i / (n - 1)) for i in range(1, n)]
    return LinearSegmentedColormap.from_list('', c, N=n)

CMAPS = {k: make_cmap(v) for k, v in HEX.items()}

def draw_bg(ax, alpha=0.18):
    ax.imshow(fl(np.where(tissue_mask, alpha, 0.)), cmap='gray', vmin=0, vmax=1,
              interpolation='nearest')

def draw_mask(ax, mask, hex_color, alpha=0.88):
    r, g, b = mc_colors.to_rgb(hex_color)
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask, 0] = r; rgba[mask, 1] = g; rgba[mask, 2] = b; rgba[mask, 3] = alpha
    ax.imshow(fl(rgba), interpolation='nearest')

def annotate_ax(ax, title, n, OR, p, title_color='white'):
    ax.axis('off')
    ax.set_title(title, fontsize=8, fontweight='bold', pad=3, color=title_color)
    sig = ' *' if p < 0.05 and OR > 1 else ''
    ax.text(0.97, 0.03, f"n = {n:,}  OR = {OR:.2f}  p = {p:.1e}{sig}",
            transform=ax.transAxes, fontsize=7.5, color='white',
            ha='right', va='bottom', fontfamily='monospace',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 0: Leiden cluster spatial map + UMAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FIGURE 0: Leiden cluster visualization ...")

TYPE_COLORS = {
    'Cancer':       '#2272C3',
    'Mast':         '#E63232',
    'Fibroblast':   '#FF8C00',
    'Endothelial':  '#00CC88',
    'T/NK cell':    '#AA44FF',
    'Macrophage':   '#FFDD00',
    'B/Plasma':     '#FF66CC',
    'Myoepithelial':'#88DDFF',
    'Adipocyte':    '#AAFFAA',
    'Other':        '#888888',
}

fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor='black')

# Panel A: spatial map colored by cluster ID
ax = axes[0]
draw_bg(ax, alpha=0.10)
cluster_ids = sorted(set(cluster_labels), key=int)
# Build single RGBA image (avoid allocating one array per cluster)
rgba_a = np.zeros((GRID, GRID, 4), dtype=np.float32)
for i, c in enumerate(cluster_ids):
    mask_c = (cluster_grid == int(c)) & tissue_mask
    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
    r, g, b = mc_colors.to_rgb(color)
    rgba_a[mask_c] = [r, g, b, 0.88]
ax.imshow(fl(rgba_a), interpolation='nearest')
del rgba_a
ax.axis('off')
ax.set_title(f"Leiden Clusters (n={n_clusters})\nresolution=0.0003", fontsize=10,
             fontweight='bold', color='white', pad=4)
if len(cluster_ids) <= 25:
    handles = [Patch(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                     label=f"C{c} {cluster_to_type[c]} ({(cluster_labels==c).sum():,})")
               for i, c in enumerate(cluster_ids)]
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              facecolor='#111', edgecolor='none', labelcolor='white', framealpha=0.9)
else:
    ax.text(0.98, 0.02, f"n = {len(cluster_ids)} clusters (too many to list)",
            transform=ax.transAxes, color='white', fontsize=8,
            ha='right', va='bottom', fontfamily='monospace',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))

# Panel B: spatial map colored by annotated cell type
ax = axes[1]
draw_bg(ax, alpha=0.10)
rgba_b = np.zeros((GRID, GRID, 4), dtype=np.float32)
for c in cluster_ids:
    mask_c = (cluster_grid == int(c)) & tissue_mask
    ct = cluster_to_type.get(c, 'Other')
    color = TYPE_COLORS.get(ct, TYPE_COLORS['Other'])
    r, g, b = mc_colors.to_rgb(color)
    rgba_b[mask_c] = [r, g, b, 0.88]
ax.imshow(fl(rgba_b), interpolation='nearest')
del rgba_b
ax.axis('off')
ax.set_title("Annotated Leiden Clusters\n(mast cells handled by targeted marker mask)", fontsize=10,
             fontweight='bold', color='white', pad=4)
type_counts = {}
for c in cluster_ids:
    ct = cluster_to_type.get(c, 'Other')
    type_counts[ct] = type_counts.get(ct, 0) + (cluster_labels == c).sum()
handles2 = [Patch(color=TYPE_COLORS.get(ct, '#888'), label=f"{ct} ({n:,})")
            for ct, n in sorted(type_counts.items(), key=lambda x: -x[1])]
ax.legend(handles=handles2, loc='lower right', fontsize=7,
          facecolor='#111', edgecolor='none', labelcolor='white', framealpha=0.9)

# Panel C: UMAP colored by cell type
ax = axes[2]
ax.set_facecolor('black')
umap1 = adata.obsm['X_umap'][:, 0]
umap2 = adata.obsm['X_umap'][:, 1]
type_labels = np.array([cluster_to_type.get(c, 'Other') for c in cluster_labels])
for ct, color in TYPE_COLORS.items():
    mask_t = type_labels == ct
    if mask_t.sum() > 0:
        ax.scatter(umap1[mask_t], umap2[mask_t], s=0.3, c=color, alpha=0.4,
                   linewidths=0, rasterized=True, label=ct)
ax.set_xlabel('UMAP 1', color='white', fontsize=9)
ax.set_ylabel('UMAP 2', color='white', fontsize=9)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.set_title("UMAP (colored by cell type annotation)", fontsize=10,
             fontweight='bold', color='white', pad=4)
ax.legend(loc='upper right', fontsize=7, facecolor='#111',
          edgecolor='none', labelcolor='white', framealpha=0.9,
          markerscale=8)

fig.suptitle(
    "Leiden Clustering  |  Visium HD FF Human Breast Cancer (DCIS)  |  Space Ranger v4.0.1\n"
    f"Curated {len(CLUSTER_GENES)}-gene panel  |  PCA 30 comps  |  k=15 neighbors  |  resolution=0.0003",
    color='white', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/VisiumHD_FF_leiden_clusters.png", dpi=200,
            bbox_inches='tight', facecolor='black')
plt.close()
print("  Saved: VisiumHD_FF_leiden_clusters.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 0b: Marker dotplot per cluster (for annotation validation)
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating cluster marker dotplot ...")

DOT_GENES = (CA_MARKERS[:5] + MC_MARKERS + FIBRO_MARKERS[:3] +
             ENDO_MARKERS[:3] + TCELL_MARKERS[:3] + MACRO_MARKERS[:3] +
             BCELL_MARKERS[:3] + MYOEPI_MARK[:3] + ADIPO_MARKERS[:2])

cluster_ids_sorted = sorted(set(cluster_labels), key=int)
pct_expr  = np.zeros((len(cluster_ids_sorted), len(DOT_GENES)))
mean_expr = np.zeros((len(cluster_ids_sorted), len(DOT_GENES)))

for ci, c in enumerate(cluster_ids_sorted):
    idx = cluster_labels == c
    for gi, gene in enumerate(DOT_GENES):
        if gene in gene_to_j:
            vals = X_log[idx, gene_to_j[gene]]
            pct_expr[ci, gi]  = (vals > 0).mean() * 100
            mean_expr[ci, gi] = vals.mean()

fig, ax = plt.subplots(figsize=(max(14, len(DOT_GENES) * 0.55),
                                max(5,  len(cluster_ids_sorted) * 0.55)),
                        facecolor='white')
vmax = np.percentile(mean_expr[mean_expr > 0], 95) if (mean_expr > 0).any() else 1.

for ci, c in enumerate(cluster_ids_sorted):
    for gi, gene in enumerate(DOT_GENES):
        pct  = pct_expr[ci, gi]
        mval = mean_expr[ci, gi]
        color_val = min(mval / vmax, 1.0)
        color = plt.cm.YlOrRd(color_val)
        size  = (pct / 100) ** 0.5 * 500
        ax.scatter(gi, ci, s=size, c=[color], linewidths=0.4,
                   edgecolors='gray', zorder=3)

ax.set_xticks(range(len(DOT_GENES)))
ax.set_xticklabels(DOT_GENES, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(cluster_ids_sorted)))
ax.set_yticklabels([f"C{c} — {cluster_to_type.get(c,'?')} ({(cluster_labels==c).sum():,})"
                    for c in cluster_ids_sorted], fontsize=8)
ax.set_xlim(-0.5, len(DOT_GENES) - 0.5)
ax.set_ylim(-0.5, len(cluster_ids_sorted) - 0.5)
ax.grid(True, lw=0.3, color='#ccc', zorder=0)
ax.set_title("Marker expression per Leiden cluster\n"
             "Dot size = % expressing bins; Color = mean log-CPM",
             fontsize=11, fontweight='bold')

# Colorbar
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, vmax))
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01, label='Mean log-CPM')

plt.tight_layout()
plt.savefig(f"{OUT}/VisiumHD_FF_leiden_dotplot.png", dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: VisiumHD_FF_leiden_dotplot.png")

# ── Lipid gene masks ──────────────────────────────────────────────────────────
LIPID_GENES = ['PTGS1', 'PTGS2', 'PLA2G2A', 'PLA2G4A', 'ELOVL5', 'ELOVL2']
GENE_LABELS = {
    'ALOX5':   'ALOX5 (5-LOX)', 'PTGS1':   'PTGS1 (COX-1)',
    'PTGS2':   'PTGS2 (COX-2)', 'PLA2G2A': 'PLA2G2A (sPLA2-IIA)',
    'PLA2G4A': 'PLA2G4A (cPLA2)', 'ELOVL5': 'ELOVL5', 'ELOVL2': 'ELOVL2',
}
lipid_mats  = {g: ln(read8(g)) for g in LIPID_GENES}
gene_thresholds = {'ALOX5': alox5_thr}
gene_masks = {}
for g in LIPID_GENES:
    gene_masks[g], gene_thresholds[g] = cancer_quantile_positive(lipid_mats[g])
for g, mk in gene_masks.items():
    print(f"  {g}: {mk.sum():,} cancer bins (logCPM >= cancer-bin Q{int(LIPID_POS_QUANTILE*100)}={gene_thresholds[g]:.3f})")

hf.close()

# ── Proximity helper ──────────────────────────────────────────────────────────
mc_r, mc_c = np.where(mc_high)
mc_mx, mc_my = to_mic(mc_r, mc_c)
tree = cKDTree(np.stack([mc_mx, mc_my], axis=1)) if len(mc_r) else None
PROX_PX = 50.0 / MPP

def prox_stat(pos_mask, ca_mask):
    neg_mask = ca_mask & ~pos_mask
    n_pos = int(pos_mask.sum()); n_neg = int(neg_mask.sum())
    if n_pos == 0 or tree is None:
        return {'n': 0, 'OR': np.nan, 'p': np.nan}
    pr, pc = np.where(pos_mask)
    pmx, pmy = to_mic(pr, pc)
    pdist, _ = tree.query(np.stack([pmx, pmy], axis=1))
    pos_near = int((pdist <= PROX_PX).sum())
    if n_neg > 0:
        nr, nc = np.where(neg_mask)
        nmx, nmy = to_mic(nr, nc)
        ndist, _ = tree.query(np.stack([nmx, nmy], axis=1))
        neg_near = int((ndist <= PROX_PX).sum())
    else:
        neg_near = 0
    ct = np.array([[pos_near, n_pos - pos_near], [neg_near, n_neg - neg_near]])
    OR, p = fisher_exact(ct, alternative='greater')
    return {'n': n_pos, 'OR': round(float(OR), 2), 'p': float(p)}

print("\nComputing proximity stats ...")
panel_stats = {}
panel_stats['ALOX5'] = prox_stat(al_ca, ca_high)
for g in LIPID_GENES:
    panel_stats[g] = prox_stat(gene_masks[g], ca_high)
    s = panel_stats[g]
    print(f"  {g:10s}  n={s['n']:,}  OR={s['OR']:.2f}  p={s['p']:.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: PIK3CA-GS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FIGURE 1: PIK3CA-GS ...")

sig     = pd.read_csv(SIGF)
sig_up  = sig[sig['coefficient'] ==  1]['SYMBOL'].tolist()
sig_dn  = sig[sig['coefficient'] == -1]['SYMBOL'].tolist()

with h5py.File(H5, 'r') as hf2:
    avail_up = [(g, name2idx[g]) for g in sig_up if g in name2idx]
    avail_dn = [(g, name2idx[g]) for g in sig_dn if g in name2idx]
    print(f"  Signature: up={len(avail_up)}/{len(sig_up)}, down={len(avail_dn)}/{len(sig_dn)}")

    def pb_all(idx):
        key = str(idx)
        if key not in hf2['feature_slices']:
            return 0.
        return float(np.log1p(float(hf2['feature_slices'][key]['data'][:].sum()) / total_umi_all * 1e6))

    up_A    = np.array([pb_all(i) for _, i in avail_up])
    dn_A    = np.array([pb_all(i) for _, i in avail_dn])
    score_A = float(up_A.mean() - dn_A.mean())
    print(f"  PIK3CA-GS (all-tissue)  = {score_A:+.4f}")

    ca_high_flat = ca_high.ravel()
    spot_bin_idx = (ur2 >> 2) * GRID + (uc2 >> 2)
    in_ca2 = ca_high_flat[spot_bin_idx]
    ca_total_umi = float(ud2[in_ca2].sum())

    def pb_cancer(idx):
        key = str(idx)
        if key not in hf2['feature_slices']:
            return 0.
        grp = hf2['feature_slices'][key]
        r2, c2, d = grp['row'][:], grp['col'][:], grp['data'][:]
        bidx  = (r2 >> 2) * GRID + (c2 >> 2)
        valid = (r2 >> 2 < GRID) & (c2 >> 2 < GRID)
        in_ca = valid & ca_high_flat[np.where(valid, bidx, 0)]
        cnt = float(d[in_ca].sum())
        return float(np.log1p(cnt / ca_total_umi * 1e6)) if ca_total_umi > 0 else 0.

    up_B    = np.array([pb_cancer(i) for _, i in avail_up])
    dn_B    = np.array([pb_cancer(i) for _, i in avail_dn])
    score_B = float(up_B.mean() - dn_B.mean())
    print(f"  PIK3CA-GS (cancer-only) = {score_B:+.4f}")

fig = plt.figure(figsize=(14, 6), facecolor='white')
gs1 = fig.add_gridspec(1, 3, wspace=0.38, left=0.07, right=0.97)

ax = fig.add_subplot(gs1[0])
methods = [f'All-tissue\npseudo-bulk\n(n = {tissue_mask.sum():,} bins)',
           f'Cancer-enriched\npseudo-bulk\n(Leiden ca, n = {ca_high.sum():,} bins)']
colors  = ['#4A90D9', '#E05A5A']
bars = ax.bar(methods, [score_A, score_B], color=colors, width=0.45,
              edgecolor='black', linewidth=0.8)
ax.axhline(0, color='black', lw=1.0, ls='--')
for bar, sc in zip(bars, [score_A, score_B]):
    ax.text(bar.get_x() + bar.get_width()/2, sc + 0.02,
            f'{sc:+.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('PIK3CA-GS Score', fontsize=11)
ax.set_title('A   PIK3CA-GS Score', fontsize=11, fontweight='bold', loc='left')
ax.set_ylim(0, max(score_A, score_B) * 1.35)
ax.text(0.5, 0.06, 'Both > 0  ->  MUTANT-like',
        transform=ax.transAxes, ha='center', fontsize=9.5, color='#cc0000', style='italic')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2 = fig.add_subplot(gs1[1])
x = np.array([0, 1]); w = 0.3
ax2.bar(x - w/2, [up_A.mean(), dn_A.mean()], w, color='#4A90D9',
        edgecolor='black', lw=0.7, label='All-tissue')
ax2.bar(x + w/2, [up_B.mean(), dn_B.mean()], w, color='#E05A5A',
        edgecolor='black', lw=0.7, label='Cancer-enriched (Leiden)')
ax2.set_xticks(x)
ax2.set_xticklabels([f'UP (n={len(avail_up)})', f'DOWN (n={len(avail_dn)})'], fontsize=10)
ax2.set_ylabel('Mean log-CPM', fontsize=11)
ax2.set_title('B   Score Decomposition', fontsize=10, fontweight='bold', loc='left')
ax2.legend(fontsize=9, framealpha=0.7)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

ax3 = fig.add_subplot(gs1[2])
ax3.scatter(up_A, up_B, alpha=0.55, s=22, color='#4477CC',
            label=f'UP (n={len(up_A)})', linewidths=0, zorder=3)
ax3.scatter(dn_A, dn_B, alpha=0.55, s=22, color='#CC4444',
            label=f'DOWN (n={len(dn_A)})', linewidths=0, zorder=3)
lmax = max(up_A.max(), dn_A.max(), up_B.max(), dn_B.max()) * 1.05
ax3.plot([0, lmax], [0, lmax], 'k--', lw=0.9, label='y = x', zorder=2)
ax3.set_xlabel('log-CPM (all-tissue)', fontsize=10)
ax3.set_ylabel('log-CPM (cancer-cell, Leiden)', fontsize=10)
ax3.set_title('C   Per-gene log-CPM', fontsize=10, fontweight='bold', loc='left')
ax3.legend(fontsize=8.5, framealpha=0.7)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

fig.suptitle(
    f'PIK3CA-GS Score  |  Visium HD FF DCIS  |  Space Ranger v4.0.1  |  '
    f'Cancer-enriched bins: Leiden cluster annotation',
    fontsize=11, fontweight='bold', y=1.02)
fig.savefig(f"{OUT}/VisiumHD_FF_PIK3CA_GS_comparison.png", dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: VisiumHD_FF_PIK3CA_GS_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: 4-panel
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FIGURE 2: 4-panel ...")

fig, axes = plt.subplots(2, 2, figsize=(20, 18), facecolor='black')
axes = axes.ravel()

ax = axes[0]
draw_bg(ax)
draw_mask(ax, ca_high, HEX['cancer'], 0.88)
cancer_cluster_str = '/'.join(f'C{c}' for c in cancer_clusters)
annotate_ax(ax,
    f"Cancer Cells (Leiden cluster {cancer_cluster_str})\n"
    f"Defined by: Leiden cluster with highest mean Cancer marker score (EPCAM/KRT8/18/19/7 etc.)\n"
    f"ca_high = {ca_high.sum():,} ({ca_high.sum()/cluster_mask.sum()*100:.1f}% of non-stromal tissue)",
    ca_high.sum(), 1.0, 1.0)

ax = axes[1]
draw_bg(ax)
draw_mask(ax, mc_high, HEX['mc'], 0.88)
annotate_ax(ax,
    f"Mast-cell-enriched bins ({active_mast_label})\n"
    f"Defined by: {active_mast_rule}\n"
    f"mc_high = {mc_high.sum():,} ({mc_high.sum()/umi_mask.sum()*100:.2f}% of UMI-filtered tissue)",
    mc_high.sum(), 1.0, 1.0)

ax = axes[2]
draw_bg(ax)
draw_mask(ax, ca_high, HEX['cancer'], 0.28)
draw_mask(ax, al_ca,   HEX['ALOX5'],  0.92)
s = panel_stats['ALOX5']
annotate_ax(ax,
    f"ALOX5+ Cancer Cells (al_ca)\n"
    f"Dim blue = ca_high (Leiden cancer clusters)  |  "
    f"Cyan = ca_high AND ALOX5 log-CPM >= cancer-bin Q{int(LIPID_POS_QUANTILE*100)}\n"
    f"al_ca = {al_ca.sum():,} ({al_ca.sum()/ca_high.sum()*100:.1f}% of ca_high)",
    s['n'], s['OR'], s['p'], title_color=HEX['ALOX5'])

ax = axes[3]
draw_bg(ax)
draw_mask(ax, ca_high, HEX['cancer'], 0.65)
draw_mask(ax, mc_high, HEX['mc'],     0.90)
_alox5 = alox5_mat  # already computed earlier
alox5_high_tissue = tissue_mask & (_alox5 > 0) & (_alox5 >= alox5_thr)
draw_mask(ax, alox5_high_tissue, '#FFDD00', 0.75)
draw_mask(ax, al_ca, HEX['ALOX5'], 0.95)
ax.legend(handles=[
    Patch(color=HEX['cancer'], label=f"Cancer ({ca_high.sum():,})"),
    Patch(color=HEX['mc'],     label=f"Mast-enriched ({mc_high.sum():,})"),
    Patch(color='#FFDD00',     label=f"ALOX5-high tissue ({alox5_high_tissue.sum():,})"),
    Patch(color=HEX['ALOX5'], label=f"ALOX5-high Cancer ({al_ca.sum():,})"),
], loc='lower right', fontsize=7.5, framealpha=0.85,
   facecolor='#111', edgecolor='none', labelcolor='white')
ax.set_title("Merged Overlay", fontsize=9, fontweight='bold', pad=3, color='white')
ax.axis('off')

fig.suptitle(
    f"Visium HD FF Human Breast Cancer (DCIS)  |  8 umm bins  |  Space Ranger v4.0.1\n"
    f"Cancer by Leiden; mast by targeted marker mask  |  PIK3CA-GS = {score_B:+.4f}  ->  "
    f"{'MUTANT-like' if score_B > 0 else 'WT-like'}",
    color='white', fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout(pad=1.5)
plt.savefig(f"{OUT}/VisiumHD_FF_4panel_fixed.png", dpi=250,
            bbox_inches='tight', facecolor='black')
plt.close()
print("  Saved: VisiumHD_FF_4panel_fixed.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Lipid 7-panel
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FIGURE 3: Lipid 7-panel ...")

fig, axes = plt.subplots(2, 4, figsize=(32, 18), facecolor='black',
                         gridspec_kw={'wspace': 0.04, 'hspace': 0.10})
axes = axes.ravel()

ax = axes[0]
s_al = panel_stats['ALOX5']
draw_bg(ax)
draw_mask(ax, ca_high, HEX['cancer'], 0.25)
draw_mask(ax, al_ca,   HEX['ALOX5'],  0.90)
draw_mask(ax, mc_high, HEX['mc'],     0.80)
annotate_ax(ax, f"ALOX5 (5-LOX)  [cancer-bin Q{int(LIPID_POS_QUANTILE*100)} high]\nn = {s_al['n']:,}",
            s_al['n'], s_al['OR'], s_al['p'], title_color=HEX['ALOX5'])

for i, gene in enumerate(LIPID_GENES):
    ax = axes[i + 1]
    s = panel_stats[gene]
    draw_bg(ax)
    draw_mask(ax, ca_high,          HEX['cancer'], 0.25)
    draw_mask(ax, gene_masks[gene], HEX[gene],     0.90)
    draw_mask(ax, mc_high,          HEX['mc'],     0.80)
    annotate_ax(ax, f"{GENE_LABELS[gene]}  [cancer-bin Q{int(LIPID_POS_QUANTILE*100)} high]\nn = {s['n']:,}",
                s['n'], s['OR'], s['p'], title_color=HEX[gene])

ax = axes[7]
draw_bg(ax)
draw_mask(ax, ca_high, HEX['cancer'], 0.20)
draw_mask(ax, al_ca,   HEX['ALOX5'],  0.80)
for gene in LIPID_GENES:
    draw_mask(ax, gene_masks[gene], HEX[gene], 0.75)
draw_mask(ax, mc_high, HEX['mc'], 0.85)
ax.legend(
    handles=[Patch(color=HEX['mc'],    label=f"Mast-enriched ({mc_high.sum():,})")] +
            [Patch(color=HEX['ALOX5'], label=f"ALOX5 ({al_ca.sum():,})")] +
            [Patch(color=HEX[g], label=f"{GENE_LABELS[g]} ({gene_masks[g].sum():,})")
             for g in LIPID_GENES],
    loc='lower right', fontsize=7, framealpha=0.88,
    facecolor='#111', edgecolor='none', labelcolor='white')
ax.set_title("Merged (lipid-high cancer-enriched bins)", fontsize=9,
             fontweight='bold', pad=3, color='white')
ax.axis('off')

fig.suptitle(
    "Lipid Metabolism Pathway Genes  |  Cancer-enriched bins, quantile-thresholded  |  "
    "Visium HD FF Human Breast Cancer (DCIS)  |  8 umm bins\n"
    "Red = Mast-cell-enriched bins  |  * = OR > 1 & p < 0.05 vs gene-low/negative cancer bins  |  Space Ranger v4.0.1",
    color='white', fontsize=10, fontweight='bold', y=1.01)
plt.savefig(f"{OUT}/VisiumHD_FF_lipid7_panel.png", dpi=250,
            bbox_inches='tight', facecolor='black', pad_inches=0.06)
plt.close()
print("  Saved: VisiumHD_FF_lipid7_panel.png")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE + FIGURE 4: Full proximity analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE + FIGURE 4: Proximity analysis ...")

GENES     = ['ALOX5'] + LIPID_GENES
RADII_UM  = [25, 50, 100]
COLORS_G  = {
    'ALOX5':   '#00E5D0', 'PTGS1':   '#50DC3C', 'PTGS2':   '#FF32AA',
    'PLA2G2A': '#FF8C00', 'PLA2G4A': '#B450FF', 'ELOVL5':  '#FFD700',
    'ELOVL2':  '#64B4FF',
}

ca_rows2, ca_cols2 = np.where(ca_high)
ca_mx2, ca_my2     = to_mic(ca_rows2, ca_cols2)
if tree is None:
    ca_dist2 = np.full(len(ca_rows2), np.nan)
else:
    ca_dist2, _ = tree.query(np.stack([ca_mx2, ca_my2], axis=1))
ca_dist_grid = np.full((GRID, GRID), np.nan)
ca_dist_grid[ca_rows2, ca_cols2] = ca_dist2

alox5_full = alox5_mat   # already computed earlier
lipid_full = lipid_mats  # already computed earlier
all_gene_mats = {'ALOX5': alox5_full}
all_gene_mats.update(lipid_full)

rows_out = []
for gene in GENES:
    mat      = all_gene_mats[gene]
    if gene == 'ALOX5':
        pos_mask = al_ca
    else:
        pos_mask = gene_masks[gene]
    neg_mask = ca_high & ~pos_mask
    n_pos = int(pos_mask.sum()); n_neg = int(neg_mask.sum())
    if n_pos == 0:
        continue

    d_pos = ca_dist_grid[pos_mask]
    d_neg = ca_dist_grid[neg_mask]
    d_pos = d_pos[~np.isnan(d_pos)] * MPP
    d_neg = d_neg[~np.isnan(d_neg)] * MPP

    med_pos = float(np.median(d_pos)) if len(d_pos) else np.nan
    med_neg = float(np.median(d_neg)) if len(d_neg) else np.nan
    _, p_mwu = mannwhitneyu(d_pos, d_neg, alternative='less') if (len(d_pos) and len(d_neg)) else (None, np.nan)

    fisher_res = {}
    for r_um in RADII_UM:
        r_px = r_um / MPP
        if tree is None:
            pos_near = 0
            neg_near = 0
            OR, p_fish = np.nan, np.nan
        else:
            pr, pc = np.where(pos_mask)
            pmx, pmy = to_mic(pr, pc)
            pdist, _ = tree.query(np.stack([pmx, pmy], axis=1))
            pos_near = int((pdist <= r_px).sum())
            nr, nc = np.where(neg_mask)
            if len(nr) > 0:
                nmx, nmy = to_mic(nr, nc)
                ndist, _ = tree.query(np.stack([nmx, nmy], axis=1))
                neg_near = int((ndist <= r_px).sum())
            else:
                neg_near = 0
            ct = np.array([[pos_near, n_pos - pos_near], [neg_near, n_neg - neg_near]])
            OR, p_fish = fisher_exact(ct, alternative='greater')
        fisher_res[r_um] = (round(float(OR), 3), float(p_fish), pos_near)

    row = {
        'gene': gene, 'label': GENE_LABELS.get(gene, gene),
        'threshold_logcpm': round(float(gene_thresholds.get(gene, np.nan)), 4),
        'n_pos_cancer': n_pos, 'n_neg_cancer': n_neg,
        'median_dist_pos_um': round(med_pos, 1),
        'median_dist_neg_um': round(med_neg, 1),
        'MWU_p': p_mwu,
    }
    for r_um in RADII_UM:
        OR_r, p_r, n_near_r = fisher_res[r_um]
        row[f'OR_{r_um}um']       = OR_r
        row[f'p_{r_um}um']        = p_r
        row[f'pct_near_{r_um}um'] = round(n_near_r / n_pos * 100, 2)
    rows_out.append(row)
    print(f"  {gene:10s}  n={n_pos:6,}  OR50={fisher_res[50][0]:.2f}(p={fisher_res[50][1]:.2e})  MWU_p={p_mwu:.2e}")

cols = ['gene', 'label', 'threshold_logcpm', 'n_pos_cancer', 'n_neg_cancer',
        'median_dist_pos_um', 'median_dist_neg_um', 'MWU_p',
        'OR_25um', 'p_25um', 'pct_near_25um',
        'OR_50um', 'p_50um', 'pct_near_50um',
        'OR_100um', 'p_100um', 'pct_near_100um']
with open(f"{OUT}/lipid_proximity_stats_full.tsv", 'w') as f:
    f.write('\t'.join(cols) + '\n')
    for row in rows_out:
        f.write('\t'.join(str(row.get(c, '')) for c in cols) + '\n')
print("  Saved: lipid_proximity_stats_full.tsv")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor='white')

ax = axes[0]
y_pos = list(range(len(rows_out) - 1, -1, -1))
for row, yp in zip(rows_out, y_pos):
    gene = row['gene']; OR_v = row['OR_50um']; p_v = row['p_50um']
    n    = row['n_pos_cancer']
    color = COLORS_G.get(gene, 'gray')
    a  = int(row['pct_near_50um'] / 100 * n)
    b  = n - a
    cc = int(row['pct_near_50um'] / 100 * row['n_neg_cancer'])
    dd = row['n_neg_cancer'] - cc
    if OR_v > 0:
        log_or = np.log(OR_v)
        se     = np.sqrt(1/max(a,1)+1/max(b,1)+1/max(cc,1)+1/max(dd,1))
        ci_lo  = np.exp(log_or - 1.96*se); ci_hi = np.exp(log_or + 1.96*se)
    else:
        ci_lo = ci_hi = OR_v
    sig = p_v < 0.05 and OR_v > 1
    ax.scatter([OR_v], [yp], color=color, s=120 if sig else 60,
               marker='D' if sig else 'o', edgecolors='black', lw=0.6, zorder=5)
    ax.plot([ci_lo, ci_hi], [yp, yp], color=color, lw=2, zorder=4)
    ax.text(-0.08, yp, f"{GENE_LABELS.get(gene,gene)} (n={n:,})",
            ha='right', va='center', fontsize=9, transform=ax.get_yaxis_transform())
    ax.text(max(OR_v, ci_hi)+0.1, yp, f"p={p_v:.1e}", va='center', fontsize=8,
            color='#cc0000' if sig else '#666')
ax.axvline(1.0, color='gray', lw=1.2, ls='--')
ax.set_xlabel("Odds Ratio (95% CI)  within 50 umm of mast cell", fontsize=10)
ax.set_yticks([]); ax.set_xlim(-0.05, None)
ax.set_title(f"A   Forest Plot: Gene-high vs Gene-low Cancer Bins\n(Fisher's Exact, 50 umm, Q{int(LIPID_POS_QUANTILE*100)} threshold)",
             fontsize=10, fontweight='bold')
for sp in ['left','top','right']: ax.spines[sp].set_visible(False)
ax.set_ylim(-0.5, len(rows_out)-0.5)

ax2 = axes[1]
gene_lbls = [r['gene'] for r in rows_out]
med_pos_v = [r['median_dist_pos_um'] for r in rows_out]
med_neg_v = [r['median_dist_neg_um'] for r in rows_out]
mwu_ps    = [r['MWU_p'] for r in rows_out]
x = np.arange(len(rows_out)); w = 0.35
ax2.bar(x-w/2, med_pos_v, w, label='Gene+ cancer',
        color=[COLORS_G.get(g,'gray') for g in gene_lbls], edgecolor='black', lw=0.6)
ax2.bar(x+w/2, med_neg_v, w, label='Gene- cancer', color='#aaa', edgecolor='black', lw=0.6, alpha=0.7)
for xi, (pm, pv, nv) in enumerate(zip(mwu_ps, med_pos_v, med_neg_v)):
    if pm < 0.05:
        ymax = max(pv, nv) + 8
        ax2.text(xi, ymax, '*', ha='center', va='bottom', fontsize=13, color='#cc0000')
        ax2.plot([xi-w/2, xi+w/2], [ymax-3, ymax-3], color='#cc0000', lw=1)
ax2.set_xticks(x)
ax2.set_xticklabels(gene_lbls, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel("Median nearest mast cell distance (umm)", fontsize=10)
ax2.set_title("B   Nearest Mast-Enriched Bin Distance\n(Mann-Whitney U, * p<0.05)",
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=9)
for sp in ['top','right']: ax2.spines[sp].set_visible(False)
for tick, gene in zip(ax2.get_xticklabels(), gene_lbls):
    tick.set_color(COLORS_G.get(gene, 'black'))

fig.suptitle(
    "Spatial Proximity to Mast-Cell-Enriched Bins: Lipid Pathway Genes in Cancer-Enriched Bins\n"
    "Visium HD FF DCIS  |  Space Ranger v4.0.1  |  Mast: targeted TPSAB1/TPSB2/CPA3 mask",
    fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/VisiumHD_FF_proximity_summary.png", dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: VisiumHD_FF_proximity_summary.png")

print("\nGenerating lipid proximity cutoff sensitivity outputs ...")

def lipid_cutoff_suffix(q):
    return f"top{int(round((1 - q) * 100)):02d}"

def lipid_masks_for_cutoff(q):
    masks = {}
    thresholds = {}
    for gene in GENES:
        mask, thr = cancer_quantile_positive(all_gene_mats[gene], q=q)
        masks[gene] = mask
        thresholds[gene] = thr
    return masks, thresholds

def proximity_rows_for_cutoff(q):
    cutoff_masks, cutoff_thresholds = lipid_masks_for_cutoff(q)
    cutoff_rows = []
    for gene in GENES:
        pos_mask = cutoff_masks[gene]
        neg_mask = ca_high & ~pos_mask
        n_pos = int(pos_mask.sum()); n_neg = int(neg_mask.sum())
        if n_pos == 0:
            continue

        d_pos = ca_dist_grid[pos_mask]
        d_neg = ca_dist_grid[neg_mask]
        d_pos = d_pos[~np.isnan(d_pos)] * MPP
        d_neg = d_neg[~np.isnan(d_neg)] * MPP

        med_pos = float(np.median(d_pos)) if len(d_pos) else np.nan
        med_neg = float(np.median(d_neg)) if len(d_neg) else np.nan
        _, p_mwu = mannwhitneyu(d_pos, d_neg, alternative='less') if (len(d_pos) and len(d_neg)) else (None, np.nan)

        fisher_res = {}
        for r_um in RADII_UM:
            r_px = r_um / MPP
            if tree is None:
                pos_near = 0
                OR, p_fish = np.nan, np.nan
            else:
                pr, pc = np.where(pos_mask)
                pmx, pmy = to_mic(pr, pc)
                pdist, _ = tree.query(np.stack([pmx, pmy], axis=1))
                pos_near = int((pdist <= r_px).sum())
                nr, nc = np.where(neg_mask)
                if len(nr) > 0:
                    nmx, nmy = to_mic(nr, nc)
                    ndist, _ = tree.query(np.stack([nmx, nmy], axis=1))
                    neg_near = int((ndist <= r_px).sum())
                else:
                    neg_near = 0
                ct = np.array([[pos_near, n_pos - pos_near], [neg_near, n_neg - neg_near]])
                OR, p_fish = fisher_exact(ct, alternative='greater')
            fisher_res[r_um] = (round(float(OR), 3), float(p_fish), pos_near)

        row = {
            'gene': gene, 'label': GENE_LABELS.get(gene, gene),
            'cutoff_top_pct': int(round((1 - q) * 100)),
            'threshold_logcpm': round(float(cutoff_thresholds.get(gene, np.nan)), 4),
            'n_pos_cancer': n_pos, 'n_neg_cancer': n_neg,
            'median_dist_pos_um': round(med_pos, 1),
            'median_dist_neg_um': round(med_neg, 1),
            'MWU_p': p_mwu,
        }
        for r_um in RADII_UM:
            OR_r, p_r, n_near_r = fisher_res[r_um]
            row[f'OR_{r_um}um']       = OR_r
            row[f'p_{r_um}um']        = p_r
            row[f'pct_near_{r_um}um'] = round(n_near_r / n_pos * 100, 2)
        cutoff_rows.append(row)
    return cutoff_rows

def save_proximity_summary_for_cutoff(cutoff_rows, q, png_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor='white')
    cutoff_label = f"top {int(round((1 - q) * 100))}%"

    ax = axes[0]
    y_pos = list(range(len(cutoff_rows) - 1, -1, -1))
    for row, yp in zip(cutoff_rows, y_pos):
        gene = row['gene']; OR_v = row['OR_50um']; p_v = row['p_50um']
        n = row['n_pos_cancer']
        color = COLORS_G.get(gene, 'gray')
        a = int(row['pct_near_50um'] / 100 * n)
        b = n - a
        cc = int(row['pct_near_50um'] / 100 * row['n_neg_cancer'])
        dd = row['n_neg_cancer'] - cc
        if np.isfinite(OR_v) and OR_v > 0:
            log_or = np.log(OR_v)
            se = np.sqrt(1/max(a,1)+1/max(b,1)+1/max(cc,1)+1/max(dd,1))
            ci_lo = np.exp(log_or - 1.96*se); ci_hi = np.exp(log_or + 1.96*se)
            x_text = max(OR_v, ci_hi) + 0.1
        else:
            ci_lo = ci_hi = OR_v
            x_text = 1.1
        sig = np.isfinite(p_v) and p_v < 0.05 and np.isfinite(OR_v) and OR_v > 1
        ax.scatter([OR_v], [yp], color=color, s=120 if sig else 60,
                   marker='D' if sig else 'o', edgecolors='black', lw=0.6, zorder=5)
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ax.plot([ci_lo, ci_hi], [yp, yp], color=color, lw=2, zorder=4)
        ax.text(-0.08, yp, f"{GENE_LABELS.get(gene,gene)} (n={n:,})",
                ha='right', va='center', fontsize=9, transform=ax.get_yaxis_transform())
        ax.text(x_text, yp, f"p={p_v:.1e}", va='center', fontsize=8,
                color='#cc0000' if sig else '#666')
    ax.axvline(1.0, color='gray', lw=1.2, ls='--')
    ax.set_xlabel("Odds Ratio (95% CI) within 50 um of mast-enriched bins", fontsize=10)
    ax.set_yticks([]); ax.set_xlim(-0.05, None)
    ax.set_title(f"A   Forest Plot: Gene-high vs Gene-low Cancer Bins\n(Fisher's Exact, 50 um, {cutoff_label} threshold)",
                 fontsize=10, fontweight='bold')
    for spn in ['left','top','right']: ax.spines[spn].set_visible(False)
    ax.set_ylim(-0.5, len(cutoff_rows)-0.5)

    ax2 = axes[1]
    gene_lbls = [r['gene'] for r in cutoff_rows]
    med_pos_v = [r['median_dist_pos_um'] for r in cutoff_rows]
    med_neg_v = [r['median_dist_neg_um'] for r in cutoff_rows]
    mwu_ps = [r['MWU_p'] for r in cutoff_rows]
    x = np.arange(len(cutoff_rows)); w = 0.35
    ax2.bar(x-w/2, med_pos_v, w, label='Gene-high cancer',
            color=[COLORS_G.get(g,'gray') for g in gene_lbls], edgecolor='black', lw=0.6)
    ax2.bar(x+w/2, med_neg_v, w, label='Gene-low/negative cancer', color='#aaa', edgecolor='black', lw=0.6, alpha=0.7)
    for xi, (pm, pv, nv) in enumerate(zip(mwu_ps, med_pos_v, med_neg_v)):
        if np.isfinite(pm) and pm < 0.05:
            ymax = max(pv, nv) + 8
            ax2.text(xi, ymax, '*', ha='center', va='bottom', fontsize=13, color='#cc0000')
            ax2.plot([xi-w/2, xi+w/2], [ymax-3, ymax-3], color='#cc0000', lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(gene_lbls, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel("Median nearest mast-enriched bin distance (um)", fontsize=10)
    ax2.set_title("B   Nearest Mast-Enriched Bin Distance\n(Mann-Whitney U, * p<0.05)",
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    for spn in ['top','right']: ax2.spines[spn].set_visible(False)
    for tick, gene in zip(ax2.get_xticklabels(), gene_lbls):
        tick.set_color(COLORS_G.get(gene, 'black'))

    fig.suptitle(
        f"Spatial Proximity to Mast-Cell-Enriched Bins: Lipid Pathway Genes in Cancer-Enriched Bins ({cutoff_label})\n"
        "Visium HD FF DCIS  |  Space Ranger v4.0.1  |  Mast: targeted TPSAB1/TPSB2/CPA3 mask",
        fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

cutoff_cols = ['gene', 'label', 'cutoff_top_pct', 'threshold_logcpm', 'n_pos_cancer', 'n_neg_cancer',
               'median_dist_pos_um', 'median_dist_neg_um', 'MWU_p',
               'OR_25um', 'p_25um', 'pct_near_25um',
               'OR_50um', 'p_50um', 'pct_near_50um',
               'OR_100um', 'p_100um', 'pct_near_100um']
for q in LIPID_POS_QUANTILES:
    suffix = lipid_cutoff_suffix(q)
    cutoff_rows = proximity_rows_for_cutoff(q)
    tsv_path = f"{OUT}/lipid_proximity_stats_full_{suffix}.tsv"
    png_path = f"{OUT}/VisiumHD_FF_proximity_summary_{suffix}.png"
    with open(tsv_path, 'w') as f:
        f.write('\t'.join(cutoff_cols) + '\n')
        for row in cutoff_rows:
            f.write('\t'.join(str(row.get(c, '')) for c in cutoff_cols) + '\n')
    save_proximity_summary_for_cutoff(cutoff_rows, q, png_path)
    print(f"  Saved cutoff sensitivity: {os.path.basename(tsv_path)}")
    print(f"  Saved cutoff sensitivity: {os.path.basename(png_path)}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All done. Outputs in", OUT)
output_files = ['VisiumHD_FF_leiden_clusters.png', 'VisiumHD_FF_leiden_dotplot.png',
                'VisiumHD_FF_PIK3CA_GS_comparison.png', 'VisiumHD_FF_4panel_fixed.png',
                'VisiumHD_FF_lipid7_panel.png', 'lipid_proximity_stats_full.tsv',
                'VisiumHD_FF_proximity_summary.png']
for q in LIPID_POS_QUANTILES:
    suffix = lipid_cutoff_suffix(q)
    output_files.extend([
        f'lipid_proximity_stats_full_{suffix}.tsv',
        f'VisiumHD_FF_proximity_summary_{suffix}.png',
    ])
for f in output_files:
    print(f"  {f}")
