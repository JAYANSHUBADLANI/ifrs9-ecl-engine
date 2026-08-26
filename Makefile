PYTHON ?= python3.12
CONFIG ?= config/project.yaml
PYTHONPATH ?= src

export PYTHONPATH

.PHONY: phase0 phase1 phase2 phase3 phase4 phase5 run-all test

phase0:
	$(PYTHON) -m ifrs9_ecl phase0 --config $(CONFIG)

phase1:
	$(PYTHON) -m ifrs9_ecl phase1 --config $(CONFIG)

phase2:
	$(PYTHON) -m ifrs9_ecl phase2 --config $(CONFIG)

phase3:
	$(PYTHON) -m ifrs9_ecl phase3 --config $(CONFIG)

phase4:
	$(PYTHON) -m ifrs9_ecl phase4 --config $(CONFIG)

phase5:
	$(PYTHON) -m ifrs9_ecl phase5 --config $(CONFIG)

run-all:
	$(PYTHON) -m ifrs9_ecl run-all --config $(CONFIG)

test:
	$(PYTHON) -m pytest
