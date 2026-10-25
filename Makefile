PYTHON ?= python

.PHONY: data db test dashboard

## Download the Kaggle dataset into data/raw/ and the data dictionary into data/reference/
data:
	$(PYTHON) scripts/download_data.py

## Build data/lc.duckdb (tables accepted, rejected; views v_accepted_asof, v_target, v_split)
db:
	$(PYTHON) scripts/build_db.py

## Run the test suite
test:
	$(PYTHON) -m pytest -q

## Launch the Streamlit PSI/CSI dashboard
dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py
