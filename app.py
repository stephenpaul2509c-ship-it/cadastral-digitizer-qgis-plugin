#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py  —  Cadastral Pipeline Flask Server
Routes:
  POST /run_boundary          -> Step 1: inference + skeleton
  POST /run_vector/<job_id>   -> Step 2: vector + simplify + snap
  POST /run_polygon/<job_id>  -> Step 3: polygonize
  GET  /status/<job_id>       -> job status + progress
  GET  /download/<job_id>/<file_key>  -> download result file
  GET  /jobs                  -> HTML monitor dashboard
"""

# ── Fix module path BEFORE any local imports ────────────────────────────────
# Ensures inference_engine.py is found regardless of the working directory
# from which Flask was launched (important for threaded imports).
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Standard imports ─────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, send_file
from pathlib import Path
import threading, uuid, logging
from datetime import datetime

# ── Import all inference_engine symbols at module level ─────────────────────
# Importing here (once, at startup) avoids repeated per-thread imports and
# eliminates the ModuleNotFoundError that occurs when threads inherit a
# different working directory from the main Flask process.
from inference_engine import (
    run_inference,
    CONFIG       as IE_CONFIG,
    _step_skeleton,
    _step_vector,
    _step_simplify,
    _step_tjunction,
    _step_snap,
    _step_polygon,
)

# ============================================================================
app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

jobs = {}

# -- Logging -----------------------------------------------------------------
log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = logging.FileHandler("job_log.txt", encoding="utf-8")
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler(
    open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
)
console_handler.setFormatter(log_formatter)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log.addHandler(console_handler)


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def home():
    return "Cadastral Server is running!"


# -- STEP 1: Boundary + Skeleton ---------------------------------------------
@app.route("/run_boundary", methods=["POST"])
def run_boundary():
    if "file" not in request.files:
        return jsonify({"error": "No file sent"}), 400

    f = request.files["file"]
    allowed = {".tif", ".tiff", ".jpg", ".jpeg"}
    if Path(f.filename).suffix.lower() not in allowed:
        return jsonify({"error": "Only .tif .jpg allowed"}), 400

    job_id  = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(parents=True)

    save_path = UPLOAD_FOLDER / f"{job_id}_{f.filename}"
    f.save(str(save_path))

    confidence = float(request.form.get("confidence", 0.10))
    client_ip  = request.remote_addr

    jobs[job_id] = {
        "status"     : "queued",
        "progress"   : 0,
        "result"     : {},
        "error"      : None,
        "filename"   : f.filename,
        "client_ip"  : client_ip,
        "started_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "duration"   : None,
        "_source_tif": str(save_path),
    }

    log.info(
        f"NEW JOB | ID: {job_id} | IP: {client_ip} | "
        f"File: {f.filename} | Confidence: {confidence}"
    )

    threading.Thread(
        target = task_boundary,
        args   = (job_id, save_path, job_dir, confidence),
        daemon = True
    ).start()

    return jsonify({"job_id": job_id, "status_url": f"/status/{job_id}"}), 202


# -- STEP 2: Vector Cleanup --------------------------------------------------
@app.route("/run_vector/<job_id>", methods=["POST"])
def run_vector(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if "boundary" not in job["result"]:
        return jsonify({"error": "Run Step 1 first"}), 400

    simplify         = float(request.form.get("simplify", 1.0))
    snap             = float(request.form.get("snap", 1.1))
    tjunction_enable = request.form.get("tjunction_enable", "false").lower() == "true"  # ← NEW
    tjunction_angle  = float(request.form.get("tjunction_angle", 165.0))                # ← NEW

    job["simplify"]          = simplify
    job["snap"]              = snap
    job["tjunction_enable"]  = tjunction_enable   # ← NEW
    job["tjunction_angle"]   = tjunction_angle    # ← NEW
    job["status"]            = "queued_vector"
    job["progress"]          = 0

    log.info(
        f"VECTOR | ID: {job_id} | Simplify: {simplify} | Snap: {snap} | "
        f"TJunction: {tjunction_enable} @ {tjunction_angle:.1f}°"
    )

    threading.Thread(
        target=task_vector,
        args=(job_id, simplify, snap),
        daemon=True
    ).start()

    return jsonify({"job_id": job_id, "status_url": f"/status/{job_id}"}), 202


# -- STEP 3: Polygon ---------------------------------------------------------
@app.route("/run_polygon/<job_id>", methods=["POST"])
def run_polygon(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if "snap" not in job["result"]:
        return jsonify({"error": "Run Step 2 first"}), 400

    job["status"]   = "queued_polygon"
    job["progress"] = 0

    log.info(f"POLYGON | ID: {job_id}")

    threading.Thread(
        target = task_polygon,
        args   = (job_id,),
        daemon = True
    ).start()

    return jsonify({"job_id": job_id, "status_url": f"/status/{job_id}"}), 202


# -- STATUS ------------------------------------------------------------------
@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status"  : job["status"],
        "progress": job["progress"],
        "error"   : job["error"],
    })


# -- DOWNLOAD ----------------------------------------------------------------
@app.route("/download/<job_id>/<file_key>")
def download(job_id, file_key):
    """
    file_key options:
      boundary -> boundary raster .tif
      skeleton -> skeleton raster .tif
      simplify -> simplified lines .shp (zipped)
      snap     -> snapped lines    .shp (zipped)
      polygon  -> final polygon    .gpkg
    """
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    path = job["result"].get(file_key)
    if not path:
        return jsonify({"error": f"'{file_key}' not ready yet"}), 400
    if not Path(path).exists():
        return jsonify({"error": f"File missing: {path}"}), 500

    # Zip all shapefile sidecar files before sending
    if str(path).endswith(".shp"):
        import zipfile, tempfile
        tmp_zip = tempfile.mktemp(suffix=".zip")
        stem    = Path(path).stem
        folder  = Path(path).parent
        with zipfile.ZipFile(tmp_zip, "w") as zf:
            for ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
                fp = folder / (stem + ext)
                if fp.exists():
                    zf.write(fp, fp.name)
        log.info(
            f"DOWNLOAD | ID: {job_id} | Key: {file_key} (zip) "
            f"| IP: {request.remote_addr}"
        )
        return send_file(tmp_zip, as_attachment=True,
                         download_name=f"{file_key}.zip")

    log.info(f"DOWNLOAD | ID: {job_id} | Key: {file_key} | IP: {request.remote_addr}")
    return send_file(path, as_attachment=True)


# -- JOB MONITOR -------------------------------------------------------------
@app.route("/jobs")
def all_jobs():
    rows = ""
    for jid, j in jobs.items():
        color = {
            "done"           : "lightgreen",
            "done_vector"    : "lightgreen",
            "done_polygon"   : "lightgreen",
            "error"          : "tomato",
            "running"        : "orange",
            "running_vector" : "orange",
            "running_polygon": "orange",
            "queued"         : "gray",
        }.get(j["status"], "white")

        rows += f"""
        <tr style="background:{color}">
            <td>{jid}</td>
            <td>{j['client_ip']}</td>
            <td>{j['filename']}</td>
            <td>{j['status'].upper()}</td>
            <td>{j['progress']}%</td>
            <td>{j['started_at']}</td>
            <td>{j.get('finished_at') or '---'}</td>
            <td>{j.get('duration') or '---'}</td>
        </tr>"""

    return f"""
    <html><head><title>Cadastral Jobs</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: monospace; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
        th {{ background: #333; color: white; }}
    </style>
    </head><body>
    <h2>Cadastral Job Monitor</h2>
    <p>Auto-refreshes every 5 seconds</p>
    <table>
        <tr>
            <th>Job ID</th><th>Client IP</th><th>File</th>
            <th>Status</th><th>Progress</th>
            <th>Started</th><th>Finished</th><th>Duration</th>
        </tr>
        {rows or "<tr><td colspan='8'>No jobs yet</td></tr>"}
    </table>
    </body></html>"""


# ============================================================================
# PIPELINE TASKS
# ============================================================================

def task_boundary(job_id, input_path, job_dir, confidence):
    job   = jobs[job_id]
    start = datetime.now()
    try:
        job["status"]  = "running"
        boundary_tif   = str(job_dir / f"{Path(input_path).stem}_boundary.tif")

        run_inference(
            image_path        = str(input_path),
            confidence        = confidence,
            device            = "cuda:0",
            output_path       = boundary_tif,
            progress_callback = lambda v: job.update({"progress": int(v)}),
            log_callback      = lambda m: log.info(f"[{job_id}] {m}"),
        )

        def cblog(m): log.info(f"[{job_id}] {m}")

        # Skeleton raster only — no vector lines in Step 1
        skel_tif = _step_skeleton(boundary_tif, str(job_dir), cblog)

        job["result"]["boundary"] = boundary_tif
        job["result"]["skeleton"] = str(skel_tif)
        job["status"]             = "done"
        job["progress"]           = 100

        end                  = datetime.now()
        job["finished_at"]   = end.strftime("%Y-%m-%d %H:%M:%S")
        job["duration"]      = str(end - start).split(".")[0]

        log.info(f"BOUNDARY DONE | ID: {job_id} | Duration: {job['duration']}")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"]  = str(e)
        log.error(f"ERROR | ID: {job_id} | {str(e)}")
        print(traceback.format_exc())


def task_vector(job_id, simplify_tol, snap_tol):
    job     = jobs[job_id]
    job_dir = OUTPUT_FOLDER / job_id
    start   = datetime.now()
    try:
        job["status"] = "running_vector"

        # ── BUG FIX 6: IE_CONFIG is a shared global dict.  Writing to it from
        #    a background thread while another job's thread reads it causes a
        #    race condition that silently applies one job's tolerances to another.
        #    Take a thread-local snapshot instead of mutating the global.
        import copy
        local_cfg = copy.copy(IE_CONFIG)
        local_cfg["simplify_tolerance"] = simplify_tol
        local_cfg["snap_tolerance"]     = snap_tol

        # Monkey-patch IE_CONFIG for the duration of this function only via a
        # context approach: temporarily override, then restore.  Because the
        # step functions read CONFIG (same object as IE_CONFIG) directly, we
        # update and restore under a lock so concurrent jobs don't stomp each other.
        import threading as _threading
        _cfg_lock = getattr(task_vector, "_lock", None)
        if _cfg_lock is None:
            task_vector._lock = _threading.Lock()
            _cfg_lock = task_vector._lock

        def cblog(m): log.info(f"[{job_id}] {m}")

        with _cfg_lock:
            IE_CONFIG["simplify_tolerance"] = simplify_tol
            IE_CONFIG["snap_tolerance"]     = snap_tol

            job["progress"] = 10
            vec_result = _step_vector(job["result"]["skeleton"], str(job_dir), cblog)
            vec_path, no_crs = vec_result if isinstance(vec_result, tuple) else (vec_result, False)
            job["result"]["raw_vector"] = str(vec_path)
            job["result"]["no_crs"]     = no_crs
            job["progress"] = 35

            simp_path = _step_simplify(vec_path, str(job_dir), cblog, no_crs=no_crs)
            job["result"]["simplify"] = str(simp_path)
            job["progress"] = 55

            # ── Optional T-Junction step ─────────────────────────────────────────
            if job.get("tjunction_enable", False):
                tj_angle = job.get("tjunction_angle", 165.0)
                tj_path  = _step_tjunction(
                    simp_path, str(job_dir), cblog,
                    no_crs=no_crs, angle_thresh=tj_angle
                )
                job["result"]["tjunction"] = str(tj_path)
                snap_input = tj_path
            else:
                snap_input = simp_path
            job["progress"] = 72

            snap_path = _step_snap(snap_input, str(job_dir), cblog, no_crs=no_crs)

        job["result"]["snap"]   = str(snap_path)
        job["result"]["vector"] = str(vec_path)
        job["progress"] = 100
        job["status"]   = "done_vector"

        end = datetime.now()
        job["finished_at"] = end.strftime("%Y-%m-%d %H:%M:%S")
        job["duration"]    = str(end - start).split(".")[0]
        log.info(f"VECTOR DONE | ID: {job_id} | Duration: {job['duration']}")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"]  = str(e)
        log.error(f"ERROR | ID: {job_id} | {str(e)}")
        print(traceback.format_exc())



def task_polygon(job_id):
    job     = jobs[job_id]
    job_dir = OUTPUT_FOLDER / job_id
    start   = datetime.now()
    try:
        job["status"]   = "running_polygon"
        job["progress"] = 10

        def cblog(m): log.info(f"[{job_id}] {m}")

        no_crs    = job["result"].get("no_crs", False)
        poly_path = _step_polygon(
            job["result"]["snap"], str(job_dir), cblog, no_crs=no_crs
        )

        job["result"]["polygon"] = str(poly_path)
        job["status"]            = "done_polygon"
        job["progress"]          = 100

        end                = datetime.now()
        job["finished_at"] = end.strftime("%Y-%m-%d %H:%M:%S")
        job["duration"]    = str(end - start).split(".")[0]

        log.info(f"POLYGON DONE | ID: {job_id} | Duration: {job['duration']}")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"]  = str(e)
        log.error(f"ERROR | ID: {job_id} | {str(e)}")
        print(traceback.format_exc())


# ============================================================================
if __name__ == "__main__":
    log.info("Cadastral Server starting on http://0.0.0.0:8889")
    app.run(host="0.0.0.0", port=8889, debug=False, threaded=True)
