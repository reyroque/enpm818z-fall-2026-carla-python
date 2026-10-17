"""Time one detector call, many times, and save every time to detector_times.json.

L13 uses these times as the detector node's processing time in pipeline_sim.py.
Optional: the hands-on reads the saved file, so you only need this script to
measure your own laptop. It needs ultralytics and the two models from GP2
(YOLOv8s and RT-DETR-L).

  python3 measure_detector.py --image frame.png --weights-dir <folder with the .pt files>

Three runs:
  yolov8s_gpu    YOLOv8s on the GPU
  rtdetr_l_gpu   RT-DETR-L on the GPU
  yolov8s_cpu    YOLOv8s on the CPU, limited to --cpu-threads threads (default 4,
                 chosen to stand for a student laptop)
Each run: 10 warm-up calls, then --n timed calls of model(image), wall time in ms.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def time_calls(model, image, n, device):
    for _ in range(10):
        model(image, device=device, verbose=False)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(image, device=device, verbose=False)
        out.append(1000.0 * (time.perf_counter() - t0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="one camera frame (png or jpg)")
    ap.add_argument("--weights-dir", default=".", help="folder with yolov8s.pt and rtdetr-l.pt")
    ap.add_argument("--n", type=int, default=600, help="timed calls per run")
    ap.add_argument("--n-cpu", type=int, default=200, help="timed calls for the CPU run")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "detector_times.json"))
    a = ap.parse_args()

    import torch
    from ultralytics import RTDETR, YOLO

    import numpy as np
    from PIL import Image
    # Decode the frame once, as a detector node gets it: an array, not a file.
    # Ultralytics reads a numpy array as BGR, as OpenCV does.
    image = np.ascontiguousarray(np.asarray(Image.open(a.image).convert("RGB"))[:, :, ::-1])
    wd = Path(a.weights_dir).expanduser()
    runs = {}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if gpu:
        runs["yolov8s_gpu"] = time_calls(YOLO(str(wd / "yolov8s.pt")), image, a.n, 0)
        runs["rtdetr_l_gpu"] = time_calls(RTDETR(str(wd / "rtdetr-l.pt")), image, a.n, 0)
    torch.set_num_threads(a.cpu_threads)
    runs["yolov8s_cpu"] = time_calls(YOLO(str(wd / "yolov8s.pt")), image, a.n_cpu, "cpu")

    load = os.getloadavg()
    meta = {"gpu": gpu, "cpu": platform.processor() or platform.machine(),
            "cpu_threads": a.cpu_threads, "image": Path(a.image).name, "image_shape": list(image.shape),
            "load_average_1min_at_end": round(load[0], 2),
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "note": "wall time of one model(image) call, in ms, after 10 warm-up calls"}
    Path(a.out).write_text(json.dumps({"meta": meta, "runs_ms": {
        k: [round(v, 3) for v in t] for k, t in runs.items()}}, indent=1))
    for k, t in runs.items():
        t = sorted(t)
        print(f"{k:14s} n={len(t)} median={t[len(t) // 2]:.1f} ms  "
              f"p99={t[int(0.99 * (len(t) - 1))]:.1f} ms  max={t[-1]:.1f} ms")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
