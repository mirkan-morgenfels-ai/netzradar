.PHONY: all data baseline export verify test lint

DATA_ARGS = $(if $(DATA_DIR),--data-dir "$(DATA_DIR)")
DATASET_ARGS = $(if $(DATASET),--dataset $(DATASET))

all: lint test verify
	uv run k3-train data --synth $(DATA_ARGS)
	uv run k3-train baseline $(DATA_ARGS)
	uv run k3-train export $(DATA_ARGS)
	uv run k3-train verify

data:
ifeq ($(SYNTH),1)
	uv run k3-train data --synth $(DATA_ARGS)
else
	uv run k3-train data $(DATASET_ARGS) $(if $(RAW_DIR),--raw-dir "$(RAW_DIR)") $(DATA_ARGS)
endif

baseline:
	uv run k3-train baseline $(DATASET_ARGS) $(DATA_ARGS)

export:
	uv run k3-train export $(DATASET_ARGS) $(DATA_ARGS)

verify:
	uv run k3-train verify

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
