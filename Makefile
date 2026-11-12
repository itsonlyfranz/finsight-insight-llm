PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: setup data test train export eval all
setup:
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -e ".[dev]" torch transformers sentencepiece protobuf
data:
	$(PY) -m finsight_llm.dataset --out data
test:
	$(PY) -m pytest -q
train:
	scripts/train.sh
export:
	scripts/export.sh
eval:
	scripts/eval.sh
all: data test train export eval
