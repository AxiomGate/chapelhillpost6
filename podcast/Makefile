.PHONY: help setup models test lint research script review approve finish status doctor gc clean

VENV := .venv
PY := $(VENV)/bin/python
PIPE := $(VENV)/bin/podcastpipe

help:
	@echo "Setup"
	@echo "  make setup      install the orchestrator and system packages"
	@echo "  make models     install the TTS and avatar environments (large)"
	@echo "  make doctor     check that everything is present"
	@echo ""
	@echo "Daily"
	@echo "  make research   gather sources, build the brief"
	@echo "  make script     draft the episode"
	@echo "  make review     open the approval UI at http://127.0.0.1:8420"
	@echo "  make approve    approve today's script from the CLI"
	@echo "  make finish     voice, avatar, captions, assembly, publish"
	@echo "  make status     recent episodes"
	@echo ""
	@echo "Maintenance"
	@echo "  make test       run the test suite"
	@echo "  make gc         prune intermediates older than 14 days"

setup:
	bash scripts/setup_ubuntu.sh

models:
	bash scripts/install_models.sh

test:
	$(PY) -m pytest tests/ -q || python -m pytest tests/ -q

doctor:
	$(PIPE) doctor

research:
	$(PIPE) research

script:
	$(PIPE) script

review:
	$(PIPE) review

approve:
	$(PIPE) approve

finish:
	$(PIPE) finish

status:
	$(PIPE) status

gc:
	$(PIPE) gc --days 14

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
