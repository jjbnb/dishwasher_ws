# ZX-2026-0303 Dishwasher Project Makefile
# Type `make` or `make help` for this list

.PHONY: help install validate-all-usd test-scene test-m0 test-piper test-plates run-l1 run-l2 run-l3 clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install          Install dishwasher package in editable mode"
	@echo "  validate-all-usd Validate native competition all.usd without launching Isaac"
	@echo "  test-scene       Inspect native all.usd prim structure"
	@echo "  test-m0          Verify native all.usd simulation baseline"
	@echo "  test-piper       Test native Piper articulation wrapper"
	@echo "  test-plates      Test native plate rigid-body wrappers"
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

validate-all-usd:
	python scripts/validate_all_usd.py

test-m0:
	python scripts/test_m0_verify.py --headless

test-piper:
	python scripts/test_piper_articulation.py --headless

test-plates:
	python scripts/test_plate_physics.py --headless

run-l1:
	python scripts/run_level1_native.py --headless --num_plates 1

run-l1-full:
	python scripts/run_level1_native.py --headless --num_plates 3

run-l2:
	@echo "Level 2 runner is not implemented yet."

run-l2-full:
	@echo "Level 2 runner is not implemented yet."

run-l3:
	@echo "Level 3 runner is not implemented yet."

run-l3-full:
	@echo "Level 3 runner is not implemented yet."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
