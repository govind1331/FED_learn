# FED_learn

A lightweight **Federated Logistic Regression** service built with Flask and scikit-learn. It exposes a simple REST API for training a logistic regression model across multiple local datasets using federated averaging, running predictions, and persisting/loading trained models.

## Overview

FED_learn simulates federated learning by training a `LogisticRegression` model locally on each supplied dataset over several rounds, then averaging the model weights and intercepts across rounds (a simplified FedAvg-style approach). The resulting model can be saved to disk, reloaded, and used to serve predictions on new data via a Flask API.

The included example model is trained on transaction-style data (columns such as `date`, `credit_amt`, `debit_amt`, `balance`, `transaction_type`) to predict a binary target column (`is_split`).

## Features

- **Federated averaging** of logistic regression weights across multiple local datasets and training rounds
- **REST API** built with Flask for training, prediction, saving, and loading models
- **Preprocessing pipeline** using `StandardScaler` for numeric features and `OneHotEncoder` for categorical features
- **Model persistence** via `joblib`
- Optional **weight reporting** — trained weights can be POSTed to an external aggregation endpoint

## Project Structure

```
FED_learn/
├── app.py                          # Flask API (train, predict, save, load endpoints)
├── logistic_reg.py                 # FederatedLogisticRegression class and training/prediction services
├── federated_logistic_model.joblib # Pre-trained example model
├── requirements.txt                # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- Flask >=2.0, <3.0
- pandas >=1.2, <2.1
- numpy >=1.20, <2.0
- scikit-learn >=1.0, <2.0
- joblib >=1.0, <2.0

## Installation

```bash
# Clone the repository
git clone https://github.com/govind1331/FED_learn.git
cd FED_learn

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the API

```bash
python app.py
```

The Flask server starts on `http://127.0.0.1:5000` with debug mode enabled.

## API Reference

### `POST /train`
Trains the federated model on an uploaded CSV file.

- **Body:** `multipart/form-data` with a `file` field containing a CSV
- **CSV columns expected:** `date`, `credit_amt`, `debit_amt`, `balance`, `transaction_type`, `is_split` (target)

```bash
curl -X POST http://127.0.0.1:5000/train \
  -F "file=@data/transactions.csv"
```

**Response**
```json
{ "message": "Model trained successfully" }
```

### `POST /predict`
Generates predictions for an uploaded CSV and returns a downloadable CSV of results.

- **Body:** `multipart/form-data` with a `file` field containing a CSV (same feature columns as training, minus the target)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@data/new_transactions.csv" \
  -o predictions.csv
```

**Response:** `predictions.csv` file containing a `prediction` column.

### `POST /save_model`
Saves the current in-memory model to disk.

- **Body:** JSON, optional `filename` (defaults to `federated_logistic_model.joblib`)

```bash
curl -X POST http://127.0.0.1:5000/save_model \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_model.joblib"}'
```

### `POST /load_model`
Loads a previously saved model from disk.

- **Body:** JSON, optional `filename` (defaults to `federated_logistic_model.joblib`)

```bash
curl -X POST http://127.0.0.1:5000/load_model \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_model.joblib"}'
```

## How It Works

1. **Preprocessing** — Dates are converted to Unix timestamps; numeric columns (`date`, `credit_amt`, `debit_amt`, `balance`) are standardized; the categorical `transaction_type` column is one-hot encoded.
2. **Local training** — For each dataset (client), a `LogisticRegression` model is trained locally with `warm_start=True`.
3. **Federated averaging** — Over `num_rounds` (default 5), weights and intercepts from each local model are averaged to produce a global model.
4. **Evaluation** — Accuracy is computed against the final dataset in the list.
5. **Optional weight reporting** — Trained weights can be sent to an external endpoint (`send_weights_to_api`) for further aggregation, e.g. a central coordination service.

## Notes & Limitations

- This is a simplified, single-process simulation of federated learning rather than a distributed multi-client system — all "clients" are datasets processed sequentially in one run.
- The weight-reporting step posts to a hardcoded local URL (`http://127.0.0.1:5005/receive_weights`) and will fail silently (logged, not raised) if that endpoint isn't running.
- `debug=True` is enabled in `app.py`; disable this before any production deployment.
- No authentication is implemented on the API endpoints — add appropriate access controls before exposing this publicly.

## Roadmap Ideas

- [ ] Add proper multi-client/distributed federated training support
- [ ] Add configuration for target column, feature columns, and number of rounds via API/config file
- [ ] Add unit tests and CI
- [ ] Add authentication and input validation on API endpoints
- [ ] Containerize with Docker

## License

This project is licensed under the [MIT License](LICENSE).

## Author

[govind1331](https://github.com/govind1331)
