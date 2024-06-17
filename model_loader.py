import joblib

def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Load the model once, so it can be reused across requests
model = load_model('trained_model5_inceptionv3old.joblib')