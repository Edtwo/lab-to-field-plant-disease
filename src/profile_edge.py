"""Edge cost profile: params, GPU latency (bf16, batch 1), peak GPU memory, ONNX size, CPU ONNX Runtime latency.
Usage: python src/profile_edge.py   -> results/edge/edge_profile.json
"""
import json, os, platform, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import build_model, dump

KEYS = ["mnv3", "effb0", "dinov2s_ft"]  # dinov2s_lp has identical inference cost to dinov2s_ft


def cpu_name():
    try:
        return subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                        "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
    except Exception:
        return platform.processor()


def main():
    import onnxruntime as ort
    torch.manual_seed(0)
    os.makedirs("results/edge", exist_ok=True)
    res = {"cpu": cpu_name(), "gpu": torch.cuda.get_device_name(0), "onnxruntime": ort.__version__,
           "torch": torch.__version__, "protocol": "batch 1, 224x224, 50 warmup + 300 timed iters, median ms"}
    for k in KEYS:
        m = build_model(k, pretrained=False).eval()
        r = {"params_M": sum(p.numel() for p in m.parameters()) / 1e6}
        x = torch.randn(1, 3, 224, 224)
        # GPU
        g = m.cuda().to(memory_format=torch.channels_last); xg = x.cuda().to(memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(50): g(xg)
            torch.cuda.synchronize(); ts = []
            for _ in range(300):
                t = time.perf_counter(); g(xg); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
        r["gpu_ms"] = float(np.median(ts)); r["gpu_peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
        # ONNX CPU
        m = m.cpu().float().to(memory_format=torch.contiguous_format)
        path = f"results/edge/{k}.onnx"
        torch.onnx.export(m, x, path, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
        r["onnx_MB"] = os.path.getsize(path) / 2**20
        for th in [1, 4]:
            so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
            s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            xn = x.numpy()
            for _ in range(50): s.run(None, {"x": xn})
            ts = []
            for _ in range(300):
                t = time.perf_counter(); s.run(None, {"x": xn}); ts.append((time.perf_counter() - t) * 1e3)
            r[f"cpu_ms_{th}thr"] = float(np.median(ts))
        os.remove(path)
        res[k] = r
        print(k, json.dumps(r), flush=True)
    dump("results/edge/edge_profile.json", res)


if __name__ == "__main__":
    main()
