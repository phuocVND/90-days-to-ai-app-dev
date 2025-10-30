mkdir -p crypto_predictor/app/models
mkdir -p crypto_predictor/app/services
mkdir -p crypto_predictor/app/controllers
mkdir -p crypto_predictor/app/schemas

touch crypto_predictor/app/main.py
touch crypto_predictor/app/__init__.py
touch crypto_predictor/app/models/price_model.py
touch crypto_predictor/app/models/__init__.py
touch crypto_predictor/app/services/data_fetcher.py
touch crypto_predictor/app/services/predictor.py
touch crypto_predictor/app/services/__init__.py
touch crypto_predictor/app/controllers/predict_controller.py
touch crypto_predictor/app/controllers/__init__.py
touch crypto_predictor/app/schemas/price_schema.py
touch crypto_predictor/app/schemas/__init__.py
touch crypto_predictor/requirements.txt
touch crypto_predictor/README.md



pip3 install -r requirements.txt


uvicorn app.main:app --reload
