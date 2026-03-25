from src.feature_engineering import build_features, load_config 
import yaml
from loguru import logger
import shap
import numpy as np
import joblib




def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()




def predict_rul(df, model_name): 
    
    model_pipeline = joblib.load(filename=f"models/{model_name}.pkl")

    features = model_pipeline.feature_names_in_
    X_test = df[features]  
    
    predicted_rul = model_pipeline.predict(X_test)
    
    logger.success(f"RUL is predicted")
    return predicted_rul  



def predict(df, engine_id, model_name):
    logger.info(f"\nRUL prediction initiating using '{model_name}'")
    
    df = df[df["engine_id"]==engine_id]
    df = build_features(df, config)

    prediction = predict_rul(df, model_name)

    return prediction 
    


def get_shap_values(df, model_name):

    pipeline = joblib.load(filename=f"models/{model_name}.pkl")
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps['scaler']
    df_processed = build_features(df, config)
    X = df_processed[scaler.feature_names_in_]

    explainer = shap.Explainer(model)
    shap_values = explainer(scaler.transform(X))

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        if len(expected) == 1:
            base_value = float(expected[0])   
        else:
            base_value = expected.tolist()
    else:
        base_value = float(expected)
    
    return {"shap_values": shap_values.values.tolist(), "base_value": base_value, "feature_names": X.columns.tolist(), "X": X.values.tolist()}



 


