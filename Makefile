# ZX-2026-0303 Dishwasher Project Makefile
# Type `make` or `make help` for this list

.PHONY: help install test-scene test-perception test-grasp run-l1 run-l2 run-l3 clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install          Install dishwasher package in editable mode"
	@echo "  test-scene       Verify scene loads correctly"
	@echo "  test-perception  Run perception module tests"
	@echo "  test-grasp       Run grasping module tests"
	@echo "  run-l1           Level 1 evaluation (10 episodes, quick)"
	@echo "  run-l1-full      Level 1 evaluation (100 episodes)"
	@echo "  run-l2           Level 2 evaluation (10 episodes, quick)"
	@echo "  run-l2-full      Level 2 evaluation (100 episodes)"
	@echo "  run-l3           Level 3 evaluation (10 episodes, quick)"
	@echo "  run-l3-full      Level 3 evaluation (100 episodes)"
	@echo "  clean            Remove __pycache__ and .pyc files"

install:
	pip install -e src/

test-scene:
	python scripts/test_scene.py

test-perception:
	python scripts/test_perception.py

test-grasp:
	python scripts/test_grasp.py

run-l1:
	python scripts/run_level1.py --episodes 10

run-l1-full:
	python scripts/run_level1.py --episodes 100

run-l2:
	python scripts/run_level2.py --episodes 10

run-l2-full:
	python scripts/run_level2.py --episodes 100

run-l3:
	python scripts/run_level3.py --episodes 10

run-l3-full:
	python scripts/run_level3.py --episodes 100

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
