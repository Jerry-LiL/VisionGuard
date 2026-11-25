.PHONY: install install-train test lint synth preview train export eval serve docker

install:
	python -m pip install -e ".[dev]"

install-train:
	python -m pip install -r requirements-train.txt
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check visionguard tests

synth:
	python -m visionguard synthesize --count 80 --out data/defects

preview:
	python -m visionguard preview-aug --data data/defects --out runs/preview

train:
	python -m visionguard train --data data/defects/data.yaml --epochs 50 --imgsz 640 --batch 8 --device cpu

export:
	python -m visionguard export --weights runs/detect/visionguard/weights/best.pt --out models/visionguard.onnx

eval:
	python -m visionguard evaluate --model models/visionguard.onnx --data data/defects/data.yaml

serve:
	uvicorn visionguard.api.app:app --host 0.0.0.0 --port 8000

docker:
	docker build -t visionguard:latest .
