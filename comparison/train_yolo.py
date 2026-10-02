"""The same Ultralytics training entrypoint for v8n, 11n and 26n."""
import argparse
import json
from pathlib import Path


def train(model_name, data, output_root, run_name, epochs=200, imgsz=640, batch=16, device=None, pretrained=True):
    from ultralytics import YOLO
    if imgsz != 640:
        raise ValueError("Comparison experiments require imgsz=640")
    run_dir = Path(output_root).resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing training run: {run_dir}")
    source = model_name if pretrained else model_name.removesuffix(".pt") + ".yaml"
    model = YOLO(source)
    kwargs = dict(data=str(Path(data).resolve()), epochs=epochs, imgsz=imgsz, batch=batch,
                  pretrained=bool(pretrained),
                  project=str(Path(output_root).resolve()), name=run_name, exist_ok=False)
    if device is not None:
        kwargs["device"] = device
    model.train(**kwargs)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    (run_dir / "training_metadata.json").write_text(json.dumps({
        "pretrained": bool(pretrained), "epochs": int(epochs), "batch_size": int(batch),
        "source_model": model_name, "data": str(Path(data).resolve()), "image_size": imgsz
    }, indent=2), encoding="utf-8")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", default="comparison_outputs")
    parser.add_argument("--device")
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    print(train(args.model, args.data, args.output_root, args.name, args.epochs, args.imgsz,
                args.batch, args.device, not args.from_scratch))
