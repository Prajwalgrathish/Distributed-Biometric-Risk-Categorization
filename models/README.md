# Models

Trained models are written here by `python -m src.train` as `random_forest.joblib` (demo runs write `demo_random_forest.joblib`). They are git-ignored because they can be large for a 200K+ row dataset and may encode information from sensitive data.

Each `.joblib` file is a dict containing the fitted scikit-learn `Pipeline` (imputation + encoding + Random Forest) and metadata needed for inference: target column, feature columns, classes, positive label, training data path, and Random Forest settings. Load it with `src.model_io.load_model`.

Only load model files you trust: joblib uses pickle, which can execute code when loading.
