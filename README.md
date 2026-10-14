# Madrid Weather Lakehouse & Forecasting Pipeline (PySpark)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/madrid-weather-pyspark/blob/main/notebooks/madrid_weather_pipeline.ipynb)

An end-to-end Big Data Engineering and Machine Learning pipeline built with **Apache Spark (PySpark)** following the **Medallion Architecture (Bronze ➔ Silver ➔ Gold)**. The project processes 103,128 hourly meteorological observations from Madrid, Spain (2015–2026), performs automated extreme weather anomaly detection, and trains a distributed ML model for temperature forecasting.

---

## Data Source & Metadata

* **Source:** Historical Meteorological Reanalysis Data from the Madrid station via [Open-Meteo API](https://open-meteo.com/).
* **Location:** Madrid, Spain (`Latitude: 40.3866° N`, `Longitude: -3.6761° W`, `Elevation: 651 m`).
* **Timeframe:** January 1, 2015 – October 6, 2026 (Hourly resolution, `GMT`).
* **Volume:** 103,128 records across 14 variables (2m temperature, apparent temperature, precipitation, rain, snowfall, snow depth, relative humidity, sea-level pressure, WMO weather code, and cloud cover layers).

---

## Project Architecture (Medallion Pattern)

```text
[Raw CSV Data] 
       │
       ▼ (Metadata header stripping & strict StructType enforcement)
[Bronze Layer]  ──► Stored as Partitioned Parquet (by year)
       │
       ▼ (Physical quality filters, normalization & temporal features)
[Silver Layer]  ──► Stored as Partitioned Parquet (by year and month)
       │
       ▼ (Daily aggregations & Window Functions: 24h & 7d moving averages)
[Gold Layer]    ──► Analytical tables & automated anomaly detection
       │
       ▼ (Lag features, VectorAssembler, StandardScaler)
[MLlib Pipeline]──► Distributed Regression Model (LinearRegression)
```

1. **Bronze Layer (Ingestion):** Cleans initial metadata rows from the raw CSV, applies a strict schema (`StructType`) to avoid inference overhead, and converts the dataset to columnar **Apache Parquet** format partitioned by `year`.
2. **Silver Layer (Quality & Feature Engineering):** Normalizes column names to `snake_case`, enforces physical boundary rules (filtering out-of-range humidity and temperature sensor anomalies), extracts calendar attributes (`month`, `day`, `hour`, `day_of_week`, `season`), and creates binary flags for rain and frost. Partitioned by `year` and `month`.
3. **Gold Layer (Analytics & Time-Series):** 
   * Daily aggregated summary table with extreme temperatures ($T_{\min}$, $T_{\max}$, $T_{\text{avg}}$), daily thermal amplitude, and cumulative rainfall/snowfall.
   * Time-series smoothing using **Window Functions** (`Window.orderBy("timestamp").rowsBetween(...)`) to compute 24-hour and 7-day rolling averages.
   * Data-driven anomaly detection without hardcoded dates.

---

## Analytical Insights: Storm Filomena Detection

Through statistical anomaly filtering on snowfall accumulation and sub-zero temperatures, the pipeline automatically isolates the most extreme winter event in the historical series (January 2021):
* **Single-Day Snowfall Record:** **20.16 cm** accumulated on `2021-01-09` (with **12.25 cm** the previous day and a snow depth peak of **0.24 m**).
* **All-Time Minimum Temperature:** **-11.1 °C** recorded on `2021-01-14` following the storm.

![Storm Filomena Impact](docs/img/filomena_impact.png)

---

## Predictive Modeling with Spark MLlib

Distributed regression pipeline designed to forecast next-hour temperature ($t$):
* **Feature Engineering (Lag Features):** Autoregressive lag variables built using `F.lag()` ($t-1$, $t-2$, and $t-24$ temperatures, alongside previous-hour relative humidity and atmospheric pressure).
* **Strict Temporal Split:** Chronological split to prevent data leakage:
  * **Training Set (Train):** 2015 – 2023
  * **Testing Set (Test):** 2024 – 2026
* **MLlib Pipeline:** `VectorAssembler` ➔ `StandardScaler` ➔ `LinearRegression`.

### Test Set Evaluation Metrics
| Metric | Value |
| :--- | :--- |
| **MAE (Mean Absolute Error)** | **0.52 °C** |
| **RMSE (Root Mean Squared Error)** | **0.75 °C** |
| **$R^2$ (Coefficient of Determination)** | **0.994** |

![PySpark MLlib Predictions](docs/img/mllib_predictions.png)

---

## Unit Testing (`pytest`)

Core Silver-layer transformations are validated via automated unit tests using `pytest` on synthetic PySpark DataFrames:
* `test_enriquecimiento_estaciones`: Verifies accurate seasonal mapping across all months.
* `test_reglas_calidad_limites`: Confirms quality filters block sensor readings outside realistic physical boundaries.

---

## Repository Structure

```text
madrid-weather-pyspark/
├── docs/
│   └── img/
│       ├── filomena_impact.png
│       └── mllib_predictions.png
├── notebooks/
│   └── madrid_weather_pipeline.ipynb
├── tests/
│   └── test_transforms.py
├── requirements.txt
└── README.md
```

---

## How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/jorgefgin/madrid-weather-pyspark.git](https://github.com/jorgefgin/madrid-weather-pyspark.git)
   cd madrid-weather-pyspark
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the unit test suite:**
   ```bash
   pytest tests/test_transforms.py -v
   ```
