#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_engine.py
Built directly from attention-model-inferences_only-1.ipynb.

Coordinate strategy for JPG inputs:
Rasters (boundary TIF, skeleton TIF):
- No transform key (identity Affine(1,0,0,0,1,0))
- QGIS renders non-CRS rasters in pixel mode (row 0 at top) → correct ✓
- Source JPG is also identity → rasters and JPG all overlay ✓

Vectors (shapefiles, GeoPackage):
- QGIS renders non-CRS vectors in standard Y-UP cartesian mode
- Identity gives y=row → y=0 at bottom → row 0 at bottom → upside down ✗
- Fix: use from_origin(0, H, 1, 1) for coordinate conversion in _step_vector
- This gives y = H - row → y=H at top → row 0 at top → correct ✓
- No .prj file written (would cause QGIS to reproject pixel coords as real-world)
"""

import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin as _fo
from pathlib import Path
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import random
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Angle helper for T-Junction step ─────────────────────────────────────────
def _angle_between(p1, vertex, p2):
    v1 = np.array(p1) - np.array(vertex)
    v2 = np.array(p2) - np.array(vertex)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

# ============================================================
# Config at MODULE LEVEL — required for torch.load pickle fix
# ============================================================
class Config:
    TRAIN_IMG_DIR    = ''
    TRAIN_MASK_DIR   = ''
    TEST_IMG_DIR     = ''
    TEST_MASK_DIR    = ''
    ENCODER          = 'resnet34'
    ENCODER_WEIGHTS  = 'imagenet'
    NUM_CLASSES      = 2
    EPOCHS           = 200
    BATCH_SIZE       = 8
    LEARNING_RATE    = 3e-4
    WEIGHT_DECAY     = 1e-4
    IMG_SIZE         = 512
    PATIENCE         = 20
    USE_AMP          = True
    DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR         = 'checkpoints'
    LOG_DIR          = 'logs'

_MODEL_PATH = Path(__file__).parent / "model" / "best_model.pth"

CONFIG = {
    "encoder_name"      : "resnet34",
    "decoder_attention" : "scse",
    "num_classes"       : 2,
    "model_path"        : str(_MODEL_PATH),
    "tile_size"         : 512,
    "stride"            : 256,
    "batch_size"        : 2,
    "seed"              : 42,
    "simplify_tolerance": 0.3,
    "snap_tolerance"    : 1.1,
    "tjunction_enable"  : False,
    "tjunction_angle"   : 165.0,
    "tjunction_snap"    : 1.0,
}

SUPPORTED_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}
GEO_EXT       = {".tif", ".tiff"}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ============================================================
# NORMALISE → uint8 (from notebook — global max)
# ============================================================
def normalise_to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img / (img.max() + 1e-8) * 255, 0, 255).astype(np.uint8)

# ============================================================
# TILE POSITION BUILDER (from notebook)
# ============================================================
def build_positions(h: int, w: int, tile_size: int, stride: int):
    rows = list(range(0, max(1, h - tile_size + 1), stride))
    cols = list(range(0, max(1, w - tile_size + 1), stride))
    if rows and rows[-1] + tile_size < h:
        rows.append(h - tile_size)
    if cols and cols[-1] + tile_size < w:
        cols.append(w - tile_size)
    if not rows: rows = [0]
    if not cols: cols = [0]
    return [(r, c) for r in rows for c in cols]

# ============================================================
# CORE INFERENCE LOOP (from notebook)
# ============================================================
def infer_positions(model, positions, read_tile_fn, h, w, aug,
                    device, threshold, progress_callback=None):
    tile_size    = CONFIG["tile_size"]
    bsz          = CONFIG["batch_size"]
    boundary_sum = np.zeros((h, w), dtype=np.float32)
    count_map    = np.zeros((h, w), dtype=np.float32)
    total        = len(positions)

    for i in range(0, total, bsz):
        batch_pos  = positions[i : i + bsz]
        batch_imgs = []
        for r, c in batch_pos:
            patch = read_tile_fn(r, c)
            ph = tile_size - patch.shape[0]
            pw = tile_size - patch.shape[1]
            if ph > 0 or pw > 0:
                patch = np.pad(patch, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            batch_imgs.append(aug(image=patch)["image"])

        with torch.no_grad():
            tensor = torch.stack(batch_imgs).to(device)
            probs  = F.softmax(model(tensor), dim=1)[:, 1, :, :].cpu().numpy()

        for idx, (r, c) in enumerate(batch_pos):
            ah = min(tile_size, h - r)
            aw = min(tile_size, w - c)
            boundary_sum[r:r+ah, c:c+aw] += probs[idx][:ah, :aw]
            count_map   [r:r+ah, c:c+aw] += 1

        if progress_callback:
            progress_callback(10 + int(((i + len(batch_pos)) / total) * 55))

    count_map[count_map == 0] = 1
    return ((boundary_sum / count_map) > threshold).astype(np.uint8) * 255

# ============================================================
# GEOTIFF PATH (from notebook)
# ============================================================
def run_geotiff(model, aug, device, threshold, input_path: Path,
                progress_callback=None):
    with rasterio.open(input_path) as src:
        h, w     = src.height, src.width
        profile  = src.profile.copy()
        positions = build_positions(h, w, CONFIG["tile_size"], CONFIG["stride"])

        def read_tile(r, c):
            data = src.read(
                window     = Window(c, r, CONFIG["tile_size"], CONFIG["tile_size"]),
                boundless  = True,
                fill_value = 0,
            )
            img = data[:3].transpose(1, 2, 0) if data.shape[0] >= 3 \
                  else np.stack([data[0]] * 3, axis=-1)
            if img.dtype != np.uint8:
                img = normalise_to_uint8(img)
            return img

        binary = infer_positions(model, positions, read_tile, h, w, aug,
                                 device, threshold, progress_callback)

    for key in ("photometric", "compress"):
        profile.pop(key, None)
    profile.update({"driver": "GTiff", "dtype": "uint8",
                    "count": 1, "nodata": 0, "compress": "lzw"})
    return binary, profile

# ============================================================
# JPG / PNG PATH (from notebook — NO transform key)
# ============================================================
def run_jpg(model, aug, device, threshold, input_path: Path,
            progress_callback=None):
    image     = np.array(PILImage.open(input_path).convert("RGB"))
    h, w      = image.shape[:2]
    positions = build_positions(h, w, CONFIG["tile_size"], CONFIG["stride"])

    def read_tile(r, c):
        ts = CONFIG["tile_size"]
        return image[r : r + ts, c : c + ts]

    binary = infer_positions(model, positions, read_tile, h, w, aug,
                             device, threshold, progress_callback)

    profile = {
        "driver"  : "GTiff",
        "dtype"   : "uint8",
        "width"   : w,
        "height"  : h,
        "count"   : 1,
        "nodata"  : 0,
        "compress": "lzw",
    }
    return binary, profile

# ============================================================
# POST-PROCESSING STEPS
# ============================================================

def _step_skeleton(boundary_tif, base_out_dir, log):
    from skimage.morphology import skeletonize as ski_skel
    skel_dir = Path(base_out_dir) / "1.skeleton"
    skel_dir.mkdir(parents=True, exist_ok=True)
    skel_tif = skel_dir / (Path(boundary_tif).stem + "_skeleton.tif")
    log("[PP 1/5] Skeletonizing ...")
    with rasterio.open(boundary_tif) as src:
        img     = src.read(1)
        has_crs = src.crs is not None
        if has_crs:
            meta = src.meta.copy()
            meta.update({"dtype": "uint8", "count": 1, "compress": "lzw", "nodata": 0})
            for k in ["jpeg_quality", "jpegtablesmode"]:
                meta.pop(k, None)
        else:
            meta = {
                "driver"  : "GTiff",
                "dtype"   : "uint8",
                "width"   : src.width,
                "height"  : src.height,
                "count"   : 1,
                "nodata"  : 0,
                "compress": "lzw",
            }
    skeleton     = ski_skel(img > 0)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    with rasterio.open(skel_tif, "w", **meta) as dst:
        dst.write(skeleton_img, 1)
    log(f"[PP 1/5] Skeleton saved -> 1.skeleton/{skel_tif.name}")
    return skel_tif


def _step_vector(skel_tif, base_out_dir, log):
    import geopandas as gpd
    from skan import Skeleton
    from shapely.geometry import LineString
    vec_dir  = Path(base_out_dir) / "2.skeleton-vector"
    vec_dir.mkdir(parents=True, exist_ok=True)
    stem     = Path(skel_tif).stem.replace("_skeleton", "")
    vec_path = vec_dir / f"{stem}_vector.shp"
    log("[PP 2/5] Skeleton -> vector lines ...")

    with rasterio.open(skel_tif) as src:
        skel   = src.read(1) > 0
        height = src.height
        crs    = src.crs

    if crs is None:
        vec_transform = _fo(0, height, 1, 1)
    else:
        vec_transform = None

    with rasterio.open(skel_tif) as src:
        real_transform = src.transform

    coord_transform = vec_transform if vec_transform is not None else real_transform

    skel_obj = Skeleton(skel)
    lines    = []
    for i in range(skel_obj.n_paths):
        path_rc = skel_obj.path_coordinates(i)
        coords  = [rasterio.transform.xy(coord_transform, int(r), int(c))
                   for r, c in path_rc]
        if len(coords) > 1:
            lines.append(LineString(coords))

    if not lines:
        raise RuntimeError("No paths found in skeleton raster.")

    gdf = gpd.GeoDataFrame({"id": range(len(lines))}, geometry=lines, crs=crs)
    gdf.to_file(vec_path)
    if crs is None:
        Path(vec_path).with_suffix(".prj").unlink(missing_ok=True)

    log(f"[PP 2/5] Vector saved -> 2.skeleton-vector/{vec_path.name} ({len(lines)} lines)")
    return vec_path, (crs is None)


def _step_simplify(vec_path, base_out_dir, log, no_crs=False):
    import geopandas as gpd
    simp_dir = Path(base_out_dir) / "3.simplify"
    simp_dir.mkdir(parents=True, exist_ok=True)
    gdf  = gpd.read_file(vec_path)
    if no_crs:
        gdf = gdf.set_crs(None, allow_override=True)
    tol       = CONFIG["simplify_tolerance"]
    stem      = Path(vec_path).stem.replace("_vector", "")
    simp_path = simp_dir / f"{stem}_simp{tol}.shp"
    log(f"[PP 3/5] Simplifying (tol={tol}) ...")
    gdf.geometry = gdf.geometry.simplify(tol, preserve_topology=True)
    gdf.to_file(simp_path)
    if no_crs:
        Path(simp_path).with_suffix(".prj").unlink(missing_ok=True)
    log(f"[PP 3/5] Simplified saved -> 3.simplify/{simp_path.name}")
    return simp_path


def _step_tjunction(simp_path, base_out_dir, log, no_crs=False,
                    angle_thresh=165.0, snap_tol=1.0):
    import geopandas as gpd
    from shapely.geometry import LineString
    from collections import defaultdict

    tj_dir  = Path(base_out_dir) / "3b.tjunction"
    tj_dir.mkdir(parents=True, exist_ok=True)
    stem    = Path(simp_path).stem
    tj_path = tj_dir / f"{stem}_tj.shp"
    log(f"[PP 3b] T-Junction regularization (angle≥{angle_thresh:.1f}°) ...")

    gdf = gpd.read_file(simp_path)
    if no_crs:
        gdf = gdf.set_crs(None, allow_override=True)

    node_map = defaultdict(list)
    for idx, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        coords = list(row.geometry.coords)
        for end in [coords[0], coords[-1]]:
            key = (round(end[0], 3), round(end[1], 3))
            node_map[key].append(idx)

    t_nodes = {k: v for k, v in node_map.items() if len(v) == 3}
    log(f"[PP 3b] Found {len(t_nodes)} T-junction nodes")

    new_coords = {
        idx: list(row.geometry.coords)
        for idx, row in gdf.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    }                                       # ← closing brace was missing

    fixed = 0
    for node_key, line_ids in t_nodes.items():
        node_pt   = np.array(node_key)
        edge_dirs = []
        for lid in line_ids:
            if lid not in new_coords:
                continue
            coords = new_coords[lid]
            if np.linalg.norm(np.array(coords[0]) - node_pt) < snap_tol:
                edge_dirs.append((lid, np.array(coords[1]),  'start'))
            elif np.linalg.norm(np.array(coords[-1]) - node_pt) < snap_tol:
                edge_dirs.append((lid, np.array(coords[-2]), 'end'))

        if len(edge_dirs) < 2:
            continue

        best_angle, best_pair = 0, None
        for i in range(len(edge_dirs)):
            for j in range(i + 1, len(edge_dirs)):
                a = _angle_between(edge_dirs[i][1], node_pt, edge_dirs[j][1])
                if a > best_angle:
                    best_angle = a
                    best_pair  = (edge_dirs[i], edge_dirs[j])

        if best_pair is None or best_angle < angle_thresh:
            continue

        p1        = best_pair[0][1]
        p2        = best_pair[1][1]
        line_vec  = p2 - p1
        t         = np.dot(node_pt - p1, line_vec) / (np.dot(line_vec, line_vec) + 1e-12)
        projected = p1 + t * line_vec

        for lid, _, pos in edge_dirs:
            if pos == 'start':
                new_coords[lid][0]  = tuple(projected)
            else:
                new_coords[lid][-1] = tuple(projected)
        fixed += 1

    log(f"[PP 3b] Fixed {fixed} / {len(t_nodes)} T-junctions")

    result = gdf.copy()
    for idx in result.index:
        if idx in new_coords:
            result.at[idx, 'geometry'] = LineString(new_coords[idx])

    result.to_file(tj_path)
    if no_crs:
        Path(tj_path).with_suffix(".prj").unlink(missing_ok=True)
    log(f"[PP 3b] T-Junction saved -> 3b.tjunction/{tj_path.name}")
    return tj_path


def _step_snap(simp_path, base_out_dir, log, no_crs=False):
    import geopandas as gpd
    from shapely.geometry import LineString
    from scipy.spatial import cKDTree
    import re
    snap_dir  = Path(base_out_dir) / "4.snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(simp_path)
    if no_crs:
        gdf = gdf.set_crs(None, allow_override=True)
    tol       = CONFIG["snap_tolerance"]
    stem      = re.sub(r"(_tj|_simp[\d\.]+)$", "", Path(simp_path).stem)
    snap_path = snap_dir / f"{stem}_snap{tol}.shp"
    log(f"[PP 4/5] Snapping (tol={tol} px) ...")
    if gdf.empty:
        raise RuntimeError("Simplified lines empty.")
    endpoints, refs = [], []
    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty: continue
        coords = list(geom.coords)
        endpoints.append(coords[0]);  refs.append((idx,  0))
        endpoints.append(coords[-1]); refs.append((idx, -1))
    endpoints = np.array(endpoints)
    tree, visited, clusters = cKDTree(endpoints), set(), []
    for i, pt in enumerate(endpoints):
        if i in visited: continue
        idxs = tree.query_ball_point(pt, tol)
        for j in idxs: visited.add(j)
        clusters.append(idxs)
    centroids = {}
    for cl in clusters:
        c = endpoints[cl].mean(axis=0)
        for i in cl: centroids[i] = c
    # ── BUG FIX 5: build a direct (row_idx, end) → endpoint_list_index map.
    #    The old refs.index() always returned the FIRST match (O(n²) scan),
    #    which is wrong when geometry indices are non-contiguous after filtering
    #    and can raise ValueError on a reset index mismatch.
    ref_to_ep = {ref: ep_i for ep_i, ref in enumerate(refs)}
    new_geoms = []
    for idx, geom in enumerate(gdf.geometry):
        coords        = list(geom.coords)
        coords[0]     = tuple(centroids[ref_to_ep[(idx,  0)]])
        coords[-1]    = tuple(centroids[ref_to_ep[(idx, -1)]])
        new_geoms.append(LineString(coords))
    out          = gdf.copy()
    out.geometry = new_geoms
    out.to_file(snap_path)
    if no_crs:
        Path(snap_path).with_suffix(".prj").unlink(missing_ok=True)
    log(f"[PP 4/5] Snap saved -> 4.snap/{snap_path.name}")
    return snap_path


def _step_polygon(snap_path, base_out_dir, log, no_crs=False):
    import geopandas as gpd
    from shapely.ops import polygonize, unary_union
    import re
    poly_dir  = Path(base_out_dir) / "5.polygon"
    poly_dir.mkdir(parents=True, exist_ok=True)
    stem      = re.sub(r"_snap[\d\.]+$", "", Path(snap_path).stem)
    # ── BUG FIX 1: save as .gpkg — app.py /download route keys "polygon"
    #    with send_file (not zip), and plugin.py downloads it as .gpkg.
    #    Saving as .shp caused the download to return an incomplete shapefile
    #    (no sidecar zip logic for the polygon key) → 0-byte / corrupt file.
    poly_path = poly_dir / f"{stem}_polygon.shp"
    log("[PP 5/5] Polygonizing ...")
    gdf = gpd.read_file(snap_path)
    if no_crs:
        gdf = gdf.set_crs(None, allow_override=True)
    gdf = gdf[~gdf.is_empty & gdf.is_valid]
    if len(gdf) == 0:
        raise RuntimeError("No valid lines to polygonize.")

    # ── BUG FIX 2: union_all() was introduced in geopandas 0.14; fall back
    #    to unary_union for older installs that silently return None / raise.
    try:
        merged = gdf.geometry.union_all()
    except AttributeError:
        merged = unary_union(gdf.geometry)

    # ── BUG FIX 3: polygonize needs fully-noded lines.
    #    shapely.ops.node() exists only in Shapely ≥ 2.0; fall back to
    #    unary_union on the individual geometries (which internally nodes them)
    #    for older installs.
    try:
        from shapely.ops import node as shapely_node
        merged = shapely_node(merged)
    except ImportError:
        from shapely.ops import unary_union
        merged = unary_union(list(gdf.geometry))  # re-union forces noding

    faces = list(polygonize(merged))
    log(f"[PP 5/5] Polygon faces : {len(faces)}")
    if not faces:
        raise RuntimeError("Polygonize produced 0 faces.")
    out              = gpd.GeoDataFrame(geometry=faces, crs=gdf.crs)
    out["area"]      = out.geometry.area
    out["perimeter"] = out.geometry.length
    # ── BUG FIX 4: driver must match the .gpkg extension
    out.to_file(poly_path, driver="ESRI Shapefile")
    log(f"[PP 5/5] Polygon saved -> 5.polygon/{poly_path.name}")
    return poly_path


def run_postprocessing(boundary_tif, base_out_dir,
                       skeleton_tif=None,
                       progress_callback=None, log_callback=None):
    def log(m):
        if log_callback: log_callback(m)
        else: print(m)
    def progress(v):
        if progress_callback: progress_callback(v)

    t0 = time.time()
    log("[INFO] ====== POST-PROCESSING START ======")
    progress(0)

    if skeleton_tif and Path(skeleton_tif).exists():
        log(f"[PP 1/5] Using provided skeleton: {Path(skeleton_tif).name}")
        skel_tif = Path(skeleton_tif)
    else:
        skel_tif = _step_skeleton(boundary_tif, base_out_dir, log)
    progress(20)

    vec_result       = _step_vector(skel_tif, base_out_dir, log)
    vec_path, no_crs = vec_result if isinstance(vec_result, tuple) else (vec_result, False)
    progress(40)
    simp_path = _step_simplify(vec_path, base_out_dir, log, no_crs=no_crs); progress(55)
    snap_path = _step_snap(simp_path, base_out_dir, log, no_crs=no_crs);    progress(80)
    poly_path = _step_polygon(snap_path, base_out_dir, log, no_crs=no_crs); progress(95)

    total = time.time() - t0
    log(f"[INFO] Post-processing done : {int(total//60)}m {int(total%60)}s")
    log("[INFO] ====== POST-PROCESSING END ========")
    progress(100)

    return {
        "skeleton": str(skel_tif),
        "vector"  : str(vec_path),
        "simplify": str(simp_path),
        "snap"    : str(snap_path),
        "polygon" : str(poly_path),
    }

# ============================================================
# MAIN run_inference — used by app.py / plugin.py
# ============================================================
def run_inference(image_path, confidence, device, output_path,
                  progress_callback=None, log_callback=None):
    def log(m):
        if log_callback: log_callback(m)
        else: print(m)
    def progress(v):
        if progress_callback: progress_callback(v)

    total_start = time.time()

    if not Path(CONFIG["model_path"]).exists():
        raise FileNotFoundError(
            f"Model not found at: {CONFIG['model_path']}\n"
            "Please copy best_model.pth into the model/ folder.")

    log(f"[INFO] Model     : {Path(CONFIG['model_path']).name}")
    log(f"[INFO] Device    : {device}")
    log(f"[INFO] Threshold : {confidence:.2f}")

    import __main__
    __main__.Config = Config

    model = smp.Unet(
        encoder_name         = CONFIG["encoder_name"],
        encoder_weights      = None,
        in_channels          = 3,
        classes              = CONFIG["num_classes"],
        decoder_attention_type = CONFIG["decoder_attention"],
        activation           = None,
    ).to(device)
    checkpoint = torch.load(CONFIG["model_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log(f"[INFO] Model loaded | Epoch: {checkpoint.get('epoch','?')} "
        f"| Val IoU: {checkpoint.get('val_iou', float('nan')):.4f}")
    progress(5)

    aug = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    ext    = Path(image_path).suffix.lower()
    is_geo = ext in GEO_EXT

    log("[INFO] Loading raster ...")
    if is_geo:
        log("[INFO] Type : GeoTIFF")
        binary, profile = run_geotiff(model, aug, device, confidence,
                                      Path(image_path), progress_callback)
    else:
        log("[INFO] Type : JPG/PNG (no georeference)")
        binary, profile = run_jpg(model, aug, device, confidence,
                                  Path(image_path), progress_callback)
    progress(68)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(binary, 1)

    n_bound = int((binary == 255).sum())
    n_total = binary.size
    log(f"[INFO] Output written -> {output_path}")
    log(f"[INFO] Boundary: {n_bound:,} px ({100*n_bound/n_total:.1f}%) | "
        f"Parcel: {n_total-n_bound:,} px ({100*(n_total-n_bound)/n_total:.1f}%)")

    elapsed = time.time() - total_start
    log(f"[INFO] Inference time : {int(elapsed//60)}m {int(elapsed%60)}s")
    log(f"[SUCCESS] Complete -> {output_path}")
    progress(72)

    return np.stack([binary] * 3, axis=-1), None


if __name__ == "__main__":
    TEST_INPUT  = r""
    TEST_OUTPUT = r""
    if TEST_INPUT:
        run_inference(TEST_INPUT, 0.1,
                      "cuda" if torch.cuda.is_available() else "cpu",
                      TEST_OUTPUT)
    else:
        print("Set TEST_INPUT / TEST_OUTPUT to run standalone.")
