"""Command line for the defect-detection pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from visionguard.data.preview import write_preview
from visionguard.data.splits import split_pool
from visionguard.data.synthesize import synthesize_dataset
from visionguard.eval.metrics import format_summary
from visionguard.eval.run import class_names_from_yaml, evaluate_dataset
from visionguard.export import export_onnx
from visionguard.infer.detector import OnnxDetector
from visionguard.train import train


def _cmd_synthesize(args: argparse.Namespace) -> None:
    yaml_path = synthesize_dataset(
        Path(args.out),
        count=args.count,
        image_size=args.size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clean_ratio=args.clean_ratio,
    )
    logging.info("wrote %s", yaml_path)


def _cmd_split(args: argparse.Namespace) -> None:
    yaml_path = split_pool(
        Path(args.source),
        Path(args.out),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    logging.info("wrote %s", yaml_path)


def _cmd_preview(args: argparse.Namespace) -> None:
    out = write_preview(Path(args.data), Path(args.out), seed=args.seed)
    logging.info("wrote previews to %s", out)


def _cmd_train(args: argparse.Namespace) -> None:
    best = train(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
    )
    logging.info("best checkpoint %s", best)


def _cmd_export(args: argparse.Namespace) -> None:
    path = export_onnx(args.weights, out=args.out, imgsz=args.imgsz, opset=args.opset)
    logging.info("onnx model %s", path)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    names = class_names_from_yaml(Path(args.data))
    detector = OnnxDetector(
        args.model,
        names,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    summary = evaluate_dataset(
        detector,
        Path(args.data),
        split=args.split,
        iou_threshold=args.iou_match,
        out_dir=Path(args.out),
    )
    print(format_summary(summary))
    logging.info("wrote %s", args.out)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "visionguard.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionguard",
        description="Visual defect detection pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser(
        "synthesize",
        help="draw a synthetic defect dataset and hold out a val split",
    )
    synth.add_argument("--out", default="data/defects")
    synth.add_argument("--count", type=int, default=80)
    synth.add_argument("--size", type=int, default=640)
    synth.add_argument("--val-ratio", type=float, default=0.2)
    synth.add_argument("--clean-ratio", type=float, default=0.15)
    synth.add_argument("--seed", type=int, default=7)
    synth.set_defaults(func=_cmd_synthesize)

    split = sub.add_parser("split", help="split an existing YOLO pool into train and val")
    split.add_argument("--source", required=True, help="directory with images/ and labels/")
    split.add_argument("--out", required=True)
    split.add_argument("--val-ratio", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=7)
    split.set_defaults(func=_cmd_split)

    preview = sub.add_parser("preview-aug", help="save mosaic, flip, and colour-jitter examples")
    preview.add_argument("--data", default="data/defects")
    preview.add_argument("--out", default="runs/preview")
    preview.add_argument("--seed", type=int, default=7)
    preview.set_defaults(func=_cmd_preview)

    trainer = sub.add_parser("train", help="fine-tune YOLOv8")
    trainer.add_argument("--data", default="data/defects/data.yaml")
    trainer.add_argument("--model", default="yolov8n.pt")
    trainer.add_argument("--epochs", type=int, default=50)
    trainer.add_argument("--imgsz", type=int, default=640)
    trainer.add_argument("--batch", type=int, default=8)
    trainer.add_argument("--device", default="cpu")
    trainer.add_argument("--project", default="runs/detect")
    trainer.add_argument("--name", default="visionguard")
    trainer.add_argument("--workers", type=int, default=0)
    trainer.add_argument("--seed", type=int, default=7)
    trainer.set_defaults(func=_cmd_train)

    export = sub.add_parser("export", help="export a checkpoint to ONNX")
    export.add_argument("--weights", required=True)
    export.add_argument("--out", default="models/visionguard.onnx")
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--opset", type=int, default=12)
    export.set_defaults(func=_cmd_export)

    evaluate = sub.add_parser(
        "evaluate",
        help="mAP, precision/recall, and a confusion matrix on ONNX outputs",
    )
    evaluate.add_argument("--model", default="models/visionguard.onnx")
    evaluate.add_argument("--data", default="data/defects/data.yaml")
    evaluate.add_argument("--split", default="val")
    evaluate.add_argument("--conf", type=float, default=0.25)
    evaluate.add_argument("--iou", type=float, default=0.45)
    evaluate.add_argument("--iou-match", type=float, default=0.5)
    evaluate.add_argument("--out", default="runs/eval")
    evaluate.set_defaults(func=_cmd_evaluate)

    serve = sub.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
