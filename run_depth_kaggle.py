"""Optional Kaggle launcher. All model training, scoring and plotting remain in R.

One native-R process per visible GPU, with separate job folders to avoid concurrent
writes. Public outputs are two tables and two figures; detailed records go in ZIP.
Run from the repository root: python run_depth_kaggle.py
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import zipfile


def main():
    root = Path.cwd()
    for name in ["timekan.R", "run_depth_experiment.R"]:
        if not (root / name).exists():
            raise FileNotFoundError(f"Run from the repository root; missing {name}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], text=True)
    ids = [line.split(",")[0].strip() for line in gpu.strip().splitlines()]
    if not ids:
        raise RuntimeError("A GPU is required for this launcher")
    ids = ids[:2]  # independent experiments, not distributed training of a model
    datasets = ["ETTh1", "ETTh2", "ETTm1"]
    depths = [1, 2, 3, 4, 5, 10, 20]
    out = root / "results/depth"
    (out / "records").mkdir(parents=True, exist_ok=True)
    (root / "results/depth_jobs").mkdir(parents=True, exist_ok=True)
    jobs = [(ds, L) for ds in datasets for L in depths]
    jobs.sort(key=lambda j: j[1] * (4 if j[0].startswith("ETTm") else 1), reverse=True)
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ["timekan.R", "run_depth_experiment.R", "run_depth_kaggle.py"]}
    provenance = out / "environment.json"
    if provenance.exists() and json.loads(provenance.read_text())["sources_sha256"] != sources:
        raise RuntimeError("Sources changed since this sweep began; preserve records and use a fresh checkout")
    provenance.write_text(json.dumps({"gpu": gpu, "sources_sha256": sources,
        "seed": 0, "datasets": datasets, "depths": depths,
        "max_epochs": 10, "patience": 3, "history": 96, "horizon": 96,
        "parallelism": "one independent R training process per GPU"}, indent=2))
    # Download before parallel workers start, avoiding competing partial downloads.
    import urllib.request
    (root / "data").mkdir(exist_ok=True)
    for ds in datasets:
        path = root / "data" / (ds + ".csv")
        if not path.exists():
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/" + ds + ".csv", path)
    subprocess.run(["Rscript", "run_depth_experiment.R", "--task", "self_test"], check=True)
    devices = queue.Queue()
    for device in ids:
        devices.put(device)
    console = threading.Lock()

    def run(job):
        ds, L = job
        device = devices.get()
        name = f"{ds}_L{L}_seed0"
        folder = f"results/depth_jobs/{name}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        args = ["Rscript", "run_depth_experiment.R", "--datasets", ds,
            "--depths", str(L), "--device", "cuda", "--results_dir", folder,
            "--figures_dir", folder + "/figures"]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        Path(folder, "command.json").write_text(json.dumps({"command": args,
            "CUDA_VISIBLE_DEVICES": device}, indent=2))
        try:
            with Path(folder, "run.log").open("w") as log:
                process = subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(("START", "L=", "COMPLETE", "Error")):
                        with console:
                            print(f"[GPU {device}] {ds} {line}", end="", flush=True)
                if process.wait():
                    raise RuntimeError(f"{name} failed; see {folder}/run.log")
            record = Path(folder, "records", name + "_complete.rds")
            if not record.exists():
                raise RuntimeError(f"{name} returned without a completed record")
            return record
        finally:
            devices.put(device)

    def bundle():
        # Write an atomic, complete archive after each finished run.
        path = root / "TimeKAN_depth_results.zip"
        tmp = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for folder in ["results/depth", "figures/depth"]:
                for f in (root / folder).rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(root))
            for name in ["timekan.R", "run_depth_experiment.R", "run_depth_kaggle.py", "docs/DEPTH_EXPERIMENT.md"]:
                if (root / name).exists():
                    z.write(root / name, name)
        tmp.replace(path)

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ids)) as pool:
        pending = {pool.submit(run, job): job for job in jobs}
        for done in concurrent.futures.as_completed(pending):
            try:
                record = done.result()
                shutil.copy2(record, out / "records" / record.name)
                subprocess.run(["Rscript", "run_depth_experiment.R", "--task", "plots"], check=True)
                bundle()
                count = len(list((out / "records").glob("*_complete.rds")))
                print(f"SAVED {count}/21 depth runs; compact tables, plots and ZIP updated", flush=True)
            except Exception as e:
                failures.append(str(e))
                print("FAILED:", e, flush=True)
    if failures:
        (out / "failures.txt").write_text("\n".join(failures))
        raise RuntimeError("Some depth jobs failed; successful runs remain saved")
    if len(list((out / "records").glob("*_complete.rds"))) != 21:
        raise RuntimeError("Expected 21 unique completed records")
    print("DEPTH_SWEEP_VERIFIED_COMPLETE: 21 runs, seed 0 only", flush=True)


if __name__ == "__main__":
    main()
