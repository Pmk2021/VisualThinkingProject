WAYMO_DATASET ?= waymo/dataset/rgb_trajectory_dataset
WAYMO_VIEWER ?= $(WAYMO_DATASET)/viewer
WAYMO_PORT ?= 8001
PYTHON ?= python

.PHONY: waymo-smoke waymo-schema waymo-serve

waymo-smoke:
	$(PYTHON) waymo/scripts/waymo_smoke.py --dataset-root $(WAYMO_DATASET)

waymo-schema:
	$(PYTHON) waymo/scripts/generate_rgb_trajectory_schema_doc.py \
		--converter waymo/scripts/build_waymo_rgb_trajectory_dataset.py \
		--output waymo/docs/rgb_trajectory_parquet_interface.md

waymo-serve:
	bash waymo/scripts/serve_waymo_viewer.sh \
		--local $(WAYMO_VIEWER) \
		--local-port $(WAYMO_PORT)
