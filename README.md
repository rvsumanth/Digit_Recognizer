# Handwritten Digit Recognizer

A CNN trained on MNIST, served through a FastAPI backend with a single-page
web UI. Draw a digit in the browser, get a prediction and full probability
breakdown back instantly.

## Project structure

```
digit-recognizer/
├── app/
│   ├── main.py              # FastAPI app: loads the model, exposes POST /predict
│   └── static/
│       └── index.html       # Single-page UI (canvas + live probability bars)
│
├── model/
│   ├── train.py             # Builds, trains, and saves the CNN
│   └── saved_models/        # Trained .h5 files land here (gitignored)
│       ├── best_mnist_model.h5
│       └── mnist_cnn_model.h5
│
├── tests/                   # Add API/model tests here
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Train the model

```bash
python model/train.py
```

This downloads MNIST, trains the CNN with data augmentation, and saves
`best_mnist_model.h5` and `mnist_cnn_model.h5` into `model/saved_models/`.

## Run the web app

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. `main.py` looks for the model at
`model/saved_models/mnist_cnn_model.h5` by default — override with the
`MODEL_PATH` environment variable if you keep it somewhere else.

## API

`POST /predict`

```json
{ "image": "data:image/png;base64,...." }
```

Response:

```json
{ "digit": 7, "confidence": 0.98, "probabilities": [0.0, 0.0, ...] }
```
