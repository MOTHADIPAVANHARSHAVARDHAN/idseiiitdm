---
## GAME : The Outlier Astronaut
Play the live data science educational game here: "https://outlier-astronaut.onrender.com"

---
# Next-Day Air Quality (AQI) Forecasting for Hyderabad

## Goal
Predict tomorrow's Air Quality Index (AQI) in Hyderabad from today's pollutant readings and recent AQI history, and test whether machine learning can beat the simple "tomorrow = today" baseline.

## Data
- **Source:** "Air Quality Data in India (2015-2020)" on Kaggle (https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india), compiled from Central Pollution Control Board (CPCB) records.
- **File used:** `city_day.csv`, filtered to Hyderabad: 2,006 daily rows from 2015-01-04 to 2020-07-01.
- **Not included in this repository.** Download `city_day.csv` from the link above and place it in a `data/` folder. Check the dataset page for its license.

## Setup
Tested with Python 3.14 on macOS.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On a Mac, XGBoost also needs the OpenMP library: `brew install libomp`.

## How to run
1. Put `city_day.csv` in `data/`.
2. Open `notebook.ipynb` in VS Code or Jupyter and select the `venv` kernel.
3. Run all cells from top to bottom. The tuning cell takes 1-3 minutes.

All random seeds are fixed (`random_state=42`), so results should repeat exactly.

## Method
1. **Cleaning:** no duplicates; all calendar days present; AQI (the target) not imputed; 3-day forward fill on pollutant inputs; remaining gaps median-imputed inside the model pipeline (fitted on training data only).
2. **Features:** today's pollutants and AQI, AQI lags (1, 2, 7 days), 7-day rolling mean, month, day of week.
3. **Split (chronological, no shuffling):** train 2015-2018, validation 2019, test 2020 (Jan-Jun).
4. **Baselines:** persistence (tomorrow = today) and 7-day average.
5. **Models:** Ridge, Random Forest, XGBoost, with time-series cross-validation for tuning.

## Results
Validation (2019), MAE:

| Model | MAE |
|---|---|
| Random Forest (trained on 2017+) | 8.16 |
| XGBoost (trained on 2017+) | 8.62 |
| Persistence baseline | 10.18 |

Final test (2020, Jan-Jun), Random Forest trained on 2017-2019:

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Persistence baseline | 9.00 | 12.38 | 0.78 |
| Random Forest | 8.94 | 10.84 | 0.83 |

The Random Forest matched persistence on MAE but reduced RMSE by about 12%, mainly helping on days with sharp changes in AQI (strongest in January).

## Limitations
- Small gain over the baseline on the test set.
- Training window (2017+) was chosen using validation scores, so validation results are optimistic.
- Short test period that includes the start of the COVID lockdown.
- No weather data (rain, wind, humidity).
- Single city.

## Repository structure
```
notebook.ipynb      analysis and models
requirements.txt    library versions
data/               put city_day.csv here
README.md
```