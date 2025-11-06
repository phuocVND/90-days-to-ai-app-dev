admin@admins-Mac-mini-2 rag_backend % tree
.
├── README.md
├── api_test.rest
├── app
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   └── main.cpython-310.pyc
│   ├── api
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-310.pyc
│   │   └── routes
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-310.pyc
│   │       │   └── ask_router.cpython-310.pyc
│   │       └── ask_router.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── config.cpython-310.pyc
│   │   │   ├── controller.cpython-310.pyc
│   │   │   └── predict_controller.cpython-310.pyc
│   │   └── config.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── model.cpython-310.pyc
│   │   │   └── price_model.cpython-310.pyc
│   │   └── model.py
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── price_schema.cpython-310.pyc
│   │   │   └── schema.cpython-310.pyc
│   │   └── schema.py
│   └── services
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   ├── data_fetcher.cpython-310.pyc
│       │   ├── predictor.cpython-310.pyc
│       │   └── service.cpython-310.pyc
│       └── service.py
├── data
│   ├── Photovoltaic_systems.pdf
│   ├── Solar-Basics.pdf
│   ├── WK3-GE-MC3-PVintro.pdf
│   └── handbook_for_solar_pv_systems.pdf
├── main.py
└── requirements.txt


pip3 install -r requirements.txt


uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
