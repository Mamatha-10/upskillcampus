SHELL := /bin/bash
.PHONY: all train report clean
all: train report

train:
	python3 train_models.py

report:
	python3 scripts/generate_report.py

clean:
	rm -f data/submission.csv
	rm -f models/traffic_prediction_model.pkl
	rm -rf reports/*
	echo "Cleaned generated outputs"
