"""Train a dot detector on synthetic crops; select checkpoint and threshold on val pages.

    # quick M2 run
    python scripts/train_detector.py --model resnet_heatmap --steps 1500 --name heatmap_quick
    python scripts/train_detector.py --model plain_unet     --steps 1500 --name plainunet_quick

Validation runs full-page inference on --val-pages val pages, sweeps the
post-processing threshold, and keeps the checkpoint with the best dot F1
(at --tol px). The chosen threshold is stored in the checkpoint, so test
evaluation never tunes on test data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brailleocr.detection.data import CropDataset, collate_crops, load_index, read_rgb  # noqa: E402
from brailleocr.detection.inference import NeuralDetector, pick_device, save_checkpoint  # noqa: E402
from brailleocr.detection.losses import LOSSES  # noqa: E402
from brailleocr.detection.models import build_model  # noqa: E402
from brailleocr.evaluation import match_dots  # noqa: E402

THRESHOLDS = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05), [0.97, 0.98, 0.99]]), 2)


def validate(detector: NeuralDetector, pages: list[tuple[np.ndarray, np.ndarray]], tol: float) -> dict:
    counts = {t: [0, 0, 0] for t in THRESHOLDS}
    for img, gt in pages:
        prob = detector.prob(img)
        for t in THRESHOLDS:
            xy, _ = detector.points(prob, float(t))
            tp, fp, fn, _ = match_dots(xy, gt, tol)
            c = counts[t]
            c[0] += tp; c[1] += fp; c[2] += fn
    f1 = {t: 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1) for t, c in counts.items()}
    best = max(f1, key=f1.get)
    tp, fp, fn = counts[best]
    return {"threshold": float(best), "f1": f1[best], "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["resnet_heatmap", "plain_unet"])
    ap.add_argument("--backbone", default="resnet34", help="resnet_heatmap encoder")
    ap.add_argument("--no-pretrained", action="store_true", help="random encoder init (ablation)")
    ap.add_argument("--data", default=ROOT / "data" / "synth", type=Path)
    ap.add_argument("--name", default=None)
    ap.add_argument("--runs", default=ROOT / "runs", type=Path)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16, help="crops per step")
    ap.add_argument("--crops-per-page", type=int, default=4)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-train-pages", type=int, default=None)
    ap.add_argument("--val-pages", type=int, default=24)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--infer-tile", type=int, default=1024, help="full-page inference tile (lower on small GPUs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <runs>/<name>/last.pt (e.g. after a Colab disconnect)")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision (used by default on CUDA)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    name = args.name or f"{args.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = args.runs / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    model_kwargs = {"backbone": args.backbone, "pretrained": not args.no_pretrained} if args.model == "resnet_heatmap" else {}
    model, kind = build_model(args.model, **model_kwargs)
    model.to(device)
    loss_fn = LOSSES[kind]
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    print(f"{name}: {args.model} ({kind}) on {device}{gpu}, amp={use_amp}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    train = load_index(args.data / "train", max_pages=args.max_train_pages)
    ds = CropDataset(train, target=kind, crop=args.crop, crops_per_page=args.crops_per_page)
    loader = DataLoader(
        ds, batch_size=max(1, args.batch // args.crops_per_page), shuffle=True, num_workers=args.workers,
        collate_fn=collate_crops, drop_last=True, persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    val_pages = []
    for rec in load_index(args.data / "val", max_pages=args.val_pages):
        label = json.loads(rec.label.read_text())
        val_pages.append((read_rgb(rec.image), np.array([d[:2] for d in label["dots"]], float)))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, step = -1.0, 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        step, best_f1 = ck["step"], ck.get("best_f1", -1.0)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            if "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
        else:  # older checkpoint with weights only: fresh optimizer, schedule fast-forwarded
            for _ in range(step):
                sched.step()
        print(f"resumed from step {step} (best val F1 {best_f1:.4f})")
    start_step, t0, running = step, time.time(), []
    log = open(out / "log.jsonl", "a")
    model.train()
    while step < args.steps:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)  # loss in fp32 for the focal log terms
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running.append(loss.item())

            if step % 50 == 0:
                rate = (step - start_step) / (time.time() - t0)
                print(f"step {step:6d}  loss {np.mean(running):.4f}  lr {sched.get_last_lr()[0]:.2e}  {rate:.2f} it/s", flush=True)
                log.write(json.dumps({"step": step, "loss": float(np.mean(running))}) + "\n")
                running = []

            if step % args.val_every == 0 or step == args.steps:
                det = NeuralDetector(model, kind, 0.5, device, tile=args.infer_tile)
                v = validate(det, val_pages, args.tol)
                model.train()
                print(f"  val step {step}: F1 {v['f1']:.4f}  P {v['precision']:.4f}  R {v['recall']:.4f}  thr {v['threshold']}", flush=True)
                log.write(json.dumps({"step": step, "val": v}) + "\n")
                log.flush()
                extra = {"model_kwargs": {k: v_ for k, v_ in model_kwargs.items() if k != "pretrained"}, "step": step, "val": v}
                if v["f1"] > best_f1:
                    best_f1 = v["f1"]
                    save_checkpoint(out / "best.pt", model, args.model, kind, v["threshold"], **extra)
                save_checkpoint(
                    out / "last.pt", model, args.model, kind, v["threshold"], **extra,
                    optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(), best_f1=best_f1,
                )
            if step >= args.steps:
                break
    log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min; best val F1 {best_f1:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
