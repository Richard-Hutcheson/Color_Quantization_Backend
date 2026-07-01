.PHONY: setup dev

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

dev:
	.venv/bin/uvicorn src.main:app --reload
