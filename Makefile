.PHONY: install server example test bench fmt clean

install:
	pip install -e ./server -e ./sdk-python
	pip install pytest httpx scikit-learn fastapi 'uvicorn[standard]' pydantic

server:
	uvicorn langpred_server.main:app --host 0.0.0.0 --port 7187 --reload

example:
	python examples/01_migrate_from_langfuse.py

test:
	pytest -q tests/

bench:
	python benchmarks/eval_predictions.py

fmt:
	python -m black server sdk-python examples benchmarks tests

clean:
	rm -f langpred.db
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
