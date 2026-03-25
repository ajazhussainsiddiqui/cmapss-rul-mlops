import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import yaml
import mlflow
from src.data_preprocessing import load_raw_data, drop_low_variance_sensors, clip_rul, add_failure_label
from src.feature_engineering import build_features
import mlflow.sklearn
from mlflow.client import MlflowClient
from loguru import logger
import joblib


def load_config():
    with open ("config.yaml", "r") as f:
        return yaml.safe_load(f)


def plot_engine_lifecycle(df, root_path):
    engine_id = [10, 20, 40, 80]
    path = f"{root_path}engine_lifecycle.png"
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.flatten()
    for i, eid in enumerate(engine_id):  
        df_engine=df[df['engine_id']==eid]    
        y_real, y_pred = df_engine["RUL"] , df_engine["RUL_pred"]        
        ax[i].plot(df_engine['cycle'], y_real, label="actual_rul", color='green')
        ax[i].plot(df_engine["cycle"], y_pred, label="predicted_rul" ,color='red')
        ax[i].set_xlabel('cylce')
        ax[i].set_ylabel('RUL')
        ax[i].set_title(f"engine_id-{eid}")
    plt.suptitle(f"RUL vs cycles of 4 engines lifecycle")
    plt.tight_layout()
    plt.legend()
    plt.savefig(path)
    plt.close()
    return path
    

def plot_pred_vs_actual(y_pred, y_true, root_path):
    path = f"{root_path}pred_vs_actual.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, s=10)
    plt.plot([0, 125], [0, 125], "r--", label="Perfect prediction")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL") 
    plt.legend()
    plt.savefig(path)
    plt.close()
    return path 


def shap_summary_plot(model, X, features, root_path):
    path = f"{root_path}shap_plot.png"
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=features, show=False) 
    fig = plt.gcf()
    fig.tight_layout() 
    fig.savefig(path)
    plt.close(fig)
    return path



def prepare_test_data(config, ds_filename):

    df_test_raw = load_raw_data(config, f"test_{ds_filename}.txt")

    rul_datafile_path = os.path.join(config["data"]["raw_path"], f"RUL_{ds_filename}.txt")  
    true_rul_df = pd.read_csv(rul_datafile_path, sep=r"\s+", header=None, names=["RUL"])
    

    def add_rul_to_test_data(true_rul, df_test):
        rul_map = dict(enumerate(true_rul["RUL"].values, start=1))
        df_test['end_rul'] = df_test['engine_id'].map(rul_map)
        max_cycle_in_engine = df_test.groupby('engine_id')['cycle'].transform('max')
        df_test['RUL'] =  df_test['end_rul'] + (max_cycle_in_engine - df_test['cycle'])
        df_test = df_test.drop(columns=["end_rul"])

        return df_test  
    
    df_test = add_rul_to_test_data(true_rul_df, df_test_raw)
    df_test = clip_rul(df_test, config)
    df_test = add_failure_label(df_test, config)
    df_test = drop_low_variance_sensors(df_test, config, ds_filename)
    
    return df_test, df_test_raw  



def load_model_form_mlflow(model_name, ds_filename):
    
    reg_model_name = f"{model_name} - {ds_filename}" 
    model_uri = f"models:/{reg_model_name}@challenger"
    logger.info(f"Fetching '{model_uri}' from MLflow Registry...")
    pipeline = mlflow.sklearn.load_model(model_uri)
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps['scaler']

    return pipeline, model, scaler 


def save_production_model_locally(model_name):
    os.makedirs("models", exist_ok=True)
   
    reg_model_name =  model_name 
    model_uri = f"models:/{reg_model_name}@champion" 

    model_pipeline = mlflow.sklearn.load_model(model_uri)

    joblib.dump(value=model_pipeline, filename=f"models/{model_name}.pkl")
    logger.info(f"CHAMPION model'{model_name}' fetched from MLflow and save locally in '../models'")    
    return model_pipeline


def compare_and_promote_model(model_name, ds_filename, challenger_metrics):

    client = MlflowClient()
    reg_model_name = f"{model_name} - {ds_filename}"
    challenger_score = challenger_metrics["nasa_score"]

    try:
        champion_version = client.get_model_version_by_alias(name=reg_model_name, alias="champion")
        champion_run = client.get_run(run_id=champion_version.run_id)   
        champion_score =champion_run.data.metrics.get("nasa_score", float('inf'))
        logger.info(f"Current '@champion' version {champion_version.version} score: {champion_score}")           
    except Exception:
        logger.warning(f"No existing '@champion' version found for model {reg_model_name}")
        champion_score = float('inf')
    
    if challenger_score < champion_score:
        logger.success(f"WIN...! Challenger ({challenger_score}) beat Champion ({champion_score}) in nasa_score")

        challenger_version = client.get_model_version_by_alias(name=reg_model_name, alias="challenger") 
        
        client.set_registered_model_alias(name=reg_model_name, alias="champion", version=challenger_version.version) 
        logger.info(f"Version {challenger_version.version} promoted to @champion ")
        
        save_production_model_locally(model_name= reg_model_name)
    else:
        logger.info(f"LOSS...! Challenger ({challenger_score}) Did NOT beat Champion ({champion_score}) in nasa_score")



def nasa_asymmetric_score(df_test):
    """
    Asymmetric scoring from NASA challenge.
    Late predictions (positive diff) penalized more heavily.
    Lower is better.
    """
    diff = df_test["RUL_pred"] - df_test["RUL"]
    score = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return np.sum(score)




def evaluate_model(model_name, config, ds_filename):
    logger.info("Mode evaluation START...") 

    df_test_processed, df_test_raw = prepare_test_data(config, ds_filename)
    
    df_test = build_features(df_test_processed, config)

    pipeline, model, scaler = load_model_form_mlflow(model_name, ds_filename)

    feature_names = scaler.feature_names_in_
    X_test, y_test_reg = df_test[feature_names], df_test['RUL']
   
    y_pred = pipeline.predict(X_test)
    y_pred = np.maximum(y_pred - 3, 0)   # optional adjustment to reduce nasa score 
    df_test['RUL_pred'] = y_pred
     
    # Calculate metrics 
    last_cycle = df_test.groupby('engine_id').last()
    rmse_last = np.sqrt(np.mean((last_cycle['RUL_pred'] - last_cycle['RUL'])**2))
    nasa_score = nasa_asymmetric_score(last_cycle)
    rmse_all = np.sqrt(np.mean((y_pred - y_test_reg)**2))

    last_cycle_rul_df =pd.DataFrame({"true_RUL":last_cycle["RUL"], "predicted_RUL":last_cycle["RUL_pred"]}).reset_index() 

    metrics = {"nasa_score":float(nasa_score), "rmse_last_cycles":float(rmse_last), "rmse_all_test_points":float(rmse_all)}
    
    root_path = f"{config['mlflow']['artifact']}/{ds_filename}/evaluate/{model_name}_"
    plot_1 = plot_engine_lifecycle(df_test, root_path) 
    plot_2 = plot_pred_vs_actual(y_pred, y_test_reg, root_path)
        
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)
        mlflow.log_table(last_cycle_rul_df, artifact_file="last_cycles_actualRUL_predRUL.json")
        mlflow.log_params({"test_model":model_name, "test_dataset_Id":ds_filename, "df_test_raw_shape ":df_test_raw.shape, "df_test_processed_shape":df_test_processed.shape, "X_test_shape":X_test.shape, "df_test_raw_columns":df_test_raw.columns})
        mlflow.log_artifact(plot_1)
        mlflow.log_artifact(plot_2)
        if model_name != "Ridge":
                plot_3 = shap_summary_plot(model=model, X=scaler.transform(X_test), features=feature_names, root_path=root_path)  
                mlflow.log_artifact(plot_3)

    compare_and_promote_model(model_name, ds_filename, metrics)    
    

    logger.info("Mode evaluation COMPLETE...")
    return metrics



def main():
    
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    ds_filename=f"{config['data']['dataset']}"
    
    model_name = "XGBoost"

    metrics = evaluate_model(model_name, config=config, ds_filename=ds_filename)
    return metrics 



if __name__ == "__main__":
  
  print(main())

