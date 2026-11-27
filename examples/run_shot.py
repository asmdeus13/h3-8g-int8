#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_shot.py —— 跑一镜的最小例子（2/3/4 针通用）

用法：
    python run_shot.py \
        --prompt ../prompts/shot_template.txt \
        --frames a.png m.png c.png \
        --w 896 --h 672 --len 124 --steps 2

    # 2 针（放飞）
    python run_shot.py --prompt p.txt --frames a.png c.png

    # 4 针（打桩）
    python run_shot.py --prompt p.txt --frames a.png m1.png m2.png c.png

锚点帧位置按张数自动均分：2 张 → 0/123；3 张 → 0/61/123；4 张 → 0/41/83/123

⚠️ 模型文件名在下面的 MODELS 里改；本脚本不含任何密钥。
"""
import argparse
import json
import os
import sys
import time
import urllib.request

# --- 按你的实际文件名改这三组 ---
MODELS = {
    "unet":    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "adapter2": "pi_h3_2step_dwd_adapter_int8_convrot.safetensors",
    "adapter4": "pi_h3_4step_dwd_adapter_int8_convrot.safetensors",
    "clip":    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae_v":   "minimax_h3_video_vae_int8_convrot.safetensors",
    "vae_a":   "minimax_h3_audio_vae_fp32.safetensors",
}

SERVER = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


def build(w, h, frames, steps, prompt, imgs, seed=42,
          max_raw_t=None, final_scale=0.6, fps=24, prefix="video/pih3kf4"):
    """构造 ComfyUI API 格式的工作流。

    ⚠️ 为什么走 UNETLoader + Load LakonLab Adapter 两段式：
       单节点加载器（Load pi-Flow Model）对 int8_convrot 探测失败，
       报 `'NoneType' object has no attribute 'quant_config'`。
    ⚠️ 为什么底模必须是 ref2va：
       fl2va 结构上只有 first_frame / last_frame 两个口，喂不了第 3、第 4 张。
    """
    if max_raw_t is None:
        max_raw_t = 0.93 if steps <= 2 else 0.96

    adapter = MODELS["adapter2"] if steps <= 2 else MODELS["adapter4"]

    # 按张数均分引导帧位置（2/3/4 针通用）
    n_img = len(imgs)
    if n_img <= 1:
        idxs = [0]
    else:
        idxs = [round(k * (frames - 1) / (n_img - 1)) for k in range(n_img)]
    idxs = sorted(set(int(x) for x in idxs))

    wf, nid = {}, [1]

    def add(t, inputs, title=None):
        i = str(nid[0]); nid[0] += 1
        wf[i] = {"class_type": t, "inputs": inputs}
        if title:
            wf[i]["_meta"] = {"title": title}
        return i

    # 模型链
    unet = add("UNETLoader", {"unet_name": MODELS["unet"], "weight_dtype": "default"},
               "ref2va int8_convrot")
    m = add("Load LakonLab Adapter", {
        "model": [unet, 0], "adapter_name": adapter, "adapter_strength": 1.0,
    }, "pi-H3 %dstep" % steps)
    chunk = add("MiniMaxChunkFeedForward",
                {"model": [m, 0], "chunks": 4, "seq_threshold": 4096}, "ChunkFF(lowvram)")
    low = add("MiniMaxLowVRAMAttention",
              {"model": [chunk, 0], "head_chunks": 4}, "LowVRAMAttention")
    shift = add("MiniMaxH3SigmaShift",
                {"model": [low, 0], "shift_video": 12.0, "shift_audio": 3.0}, "SigmaShift 12/3")

    clip = add("CLIPLoader", {"clip_name": MODELS["clip"], "type": "minimax", "device": "default"})
    vv = add("VAELoader", {"vae_name": MODELS["vae_v"]}, "video vae")
    va = add("VAELoader", {"vae_name": MODELS["vae_a"]}, "audio vae")

    # 参考图（文本里可用 <Picture N> 引用）
    refs = {}
    for k, img in enumerate(imgs):
        refs["ref_image_%d" % k] = [add("LoadImage", {"image": img}, "ref %d" % k), 0]

    r2v = add("MiniMaxH3ReferenceToVideo", {
        "clip": [clip, 0], "vae": [vv, 0], "audio_vae": [va, 0],
        "prompt": prompt, "width": w, "height": h, "length": frames,
        "ref_image_size": "match", "ref_images": refs,
    }, "Ref2V")

    # AddGuide 串联（链式：每加一个就把它接成新的 positive）
    cond = [r2v, 0]
    for n, (idx, img) in enumerate(zip(idxs, imgs)):
        ld = add("LoadImage", {"image": img}, "KF %d" % n)
        ag = add("MiniMaxH3AddGuide", {
            "positive": cond, "latent": [r2v, 1], "vae": [vv, 0],
            "image": [ld, 0], "frame_idx": int(idx),
        }, "AddGuide f%d" % int(idx))
        cond = [ag, 0]

    guider = add("BasicGuider", {"model": [shift, 0], "conditioning": cond})
    noise = add("RandomNoise", {"noise_seed": seed})
    ksel = add("KSamplerSelect", {"sampler_name": "euler"})
    sched = add("pi-Flow Scheduler", {
        "model": [shift, 0], "steps": steps, "min_raw_t": 0.0,
        "max_raw_t": max_raw_t, "final_step_size_scale": final_scale,
    }, "pi-Flow Scheduler %dstep" % steps)
    smp = add("SamplerCustomAdvanced", {
        "noise": [noise, 0], "guider": [guider, 0], "sampler": [ksel, 0],
        "sigmas": [sched, 0], "latent_image": [r2v, 1],
    }, "sample")

    dwd = add("DWD Decode", {"samples": [smp, 0]}, "DWD decode")
    aud = add("VAEDecodeAudio", {"samples": [smp, 0], "vae": [va, 0]}, "audio decode")
    cv = add("CreateVideo", {"images": [dwd, 0], "audio": [aud, 0], "fps": fps})
    add("SaveVideo", {"video": [cv, 0], "filename_prefix": prefix,
                      "format": "auto", "codec": "auto"})
    return wf


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="视频提示词文件")
    ap.add_argument("--frames", nargs="+", required=True,
                    help="锚点图（2/3/4 张），按时间顺序")
    ap.add_argument("--w", type=int, default=896, help="宽（必须 32 的倍数）")
    ap.add_argument("--h", type=int, default=672, help="高（必须 32 的倍数）")
    ap.add_argument("--len", type=int, default=124, help="帧数（会吸附到 17k+5）")
    ap.add_argument("--steps", type=int, default=2, choices=[2, 4])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--prefix", default="video/pih3kf4")
    args = ap.parse_args()

    if args.w % 32 or args.h % 32:
        print("  ❌ width/height 必须是 32 的倍数，否则报 "
              "`shape ... is invalid for input of size N`", file=sys.stderr)
        return 2

    prompt = open(args.prompt, encoding="utf-8").read().strip()
    wf = build(args.w, args.h, args.len, args.steps, prompt, args.frames,
               seed=args.seed, fps=args.fps, prefix=args.prefix)

    print("  %d 针 / %d 步 / %dx%d / %d 帧 / seed %d"
          % (len(args.frames), args.steps, args.w, args.h, args.len, args.seed))
    t0 = time.time()
    res = post("/prompt", {"prompt": wf})
    pid = res.get("prompt_id")
    print("  queued: %s" % pid)

    # 轮询
    while True:
        time.sleep(3)
        with urllib.request.urlopen(SERVER + "/history/" + pid, timeout=30) as r:
            hist = json.loads(r.read().decode("utf-8"))
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("completed"):
                outs = hist[pid].get("outputs", {})
                print("  ✅ %.0fs 完成" % (time.time() - t0))
                for node, out in outs.items():
                    for k, v in out.items():
                        if isinstance(v, list):
                            for item in v:
                                print("    %s" % item)
                return 0
            if st.get("status_str") == "error":
                print("  ❌ %s" % json.dumps(st, ensure_ascii=False)[:800], file=sys.stderr)
                return 1
        if time.time() - t0 > 1800:
            print("  ❌ 超时 30 分钟", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
