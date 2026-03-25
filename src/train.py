import pandas as pd
import yaml
import os
import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import optuna.visualization as vis
from evaluate import evaluate_model
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
from loguru import logger
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger.add("pipeline_execution.log", rotation="10 MB")


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
    

def setup_mlflow(config, ds_filename):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    root_path = f"{config['mlflow']['artifact']}/{ds_filename}"
    os.makedirs(f"{root_path}/train", exist_ok=True)
    os.makedirs(f"{root_path}/evaluate", exist_ok=True)
    logger.info("MLflow tracking setup DONE")
     

def load_data(config, ds_filename):
    
    df_path = os.path.join(config["data"]["processed_path"], f"train_{ds_filename}_processed.csv")
    df = pd.read_csv(df_path)                        
    return df 

def prepare_data(df, config):

    X = df.drop(columns=config['training']['non_feature_columns'])
    y_reg = df["RUL"]
    y_cls = df["will_fail_soon"]
    groups = df["engine_id"]

    logger.info("Data is prepared")
    return X, y_reg, y_cls, groups



def cv_score(model, X, y, groups, cv_folds=5):
    """
    Use GroupKFold so same engine never appears in both train and val.
    Returns mean RMSE across folds.
    """
    gkf = GroupKFold(n_splits=cv_folds)

    pipeline = make_pipeline(StandardScaler(), model)
    
    scores = cross_val_score(pipeline, X, y, cv=gkf, groups=groups, scoring="neg_root_mean_squared_error")
    
    return -scores.mean()



def objective_ridge(trial, X, y, groups, cv_folds):
    params = {
        'alpha': trial.suggest_float('alpha', 1e-2, 20.0, log=True)
    }

    model = Ridge(**params, random_state=42)
    return cv_score(model, X, y, groups, cv_folds)


def objective_rf(trial, X, y, groups, cv_folds):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', None]),
        'n_jobs': -1
    }
     
    model = RandomForestRegressor(**params, random_state=42)
    return cv_score(model, X, y, groups, cv_folds)
 

def objective_xgb(trial, X, y, groups, cv_folds):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 350, step=50),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        "tree_method": "hist",  
        "device": "cuda",
    }
    
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    score = cv_score(model, X, y, groups, cv_folds)
    return score


def objective_lgbm(trial, X, y, groups, cv_folds):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 350, step=50),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
        "device": "gpu"            
    }

    model = LGBMRegressor(**params, random_state=42, verbose=-1)
    return cv_score(model, X, y, groups, cv_folds)



def tune_with_optuna(X_train, y_train, groups, config, model_name, ds_filename, n_trials=50):
    
    # trim tuning data
    X = X_train.iloc[-30000:]
    y = y_train.to_frame()
    y = y.iloc[-30000:] 
    groups = groups.iloc[-30000:]

    objectives ={
        "Ridge": objective_ridge,
        "RandomForest": objective_rf,
        "XGBoost": objective_xgb,
        "LightGBM": objective_lgbm
    }
    model_name_pref =  model_name.split('_')[0]
    if model_name_pref not in objectives:
        raise ValueError(f"Unknown model: {model_name_pref}. Choose from: {list(objectives.keys())}")
    logger.info(f"Tuning {model_name} with {n_trials} trials...")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    
        
    study.optimize(lambda trial: objectives[model_name_pref](trial=trial, X=X, y=y, groups=groups, cv_folds=config['training']['cv_folds']),
                n_trials=n_trials, show_progress_bar=True, n_jobs=config['training']['n_jobs'])

    with mlflow.start_run(run_name=f"optuna_tuning_{model_name}_{ds_filename}", nested=True):
        best_params = study.best_params
        best_rmse = study.best_value
        trials_df = study.trials_dataframe()  # History of trials 
        
        root_path = f"{config['mlflow']['artifact']}/{ds_filename}/train/{model_name}_"
        
        fig = vis.plot_optimization_history(study)    # trails vs rmse  
        fig.write_html(f"{root_path}optimization_history.html")
        fig = vis.plot_param_importances(study)           # parameters importance  
        fig.write_html(f"{root_path}param_importances.html")
        fig = vis.plot_slice(study)                        # rmse vs each parameter             
        fig.write_html(f"{root_path}slice_plot.html")
        fig = vis.plot_timeline(study)                  # tilmeline of all trials
        fig.write_image(f"{root_path}timeline_plot.png")
        
        mlflow.log_artifact(f"{root_path}timeline_plot.png")
        mlflow.log_artifact(f"{root_path}slice_plot.html")
        mlflow.log_artifact(f"{root_path}param_importances.html")
        mlflow.log_artifact(f"{root_path}optimization_history.html")
        mlflow.log_params({"model_name": model_name, "dataset_Id":ds_filename, "X_train_shape":X_train.shape, "X_optuna_shape":X.shape, "best_params": best_params, "data_features": ",".join(X.columns)})
        mlflow.log_metrics({"best_rmse": best_rmse, "best_trial_number": study.best_trial.number})
        mlflow.log_table(trials_df , artifact_file="cv_trial_history.json")        
        
        return {"model_name": model_name, "best_rmse": best_rmse, "best_params": best_params}



def train_and_save_model(X, y, params, additional_params, model_name, ds_filename):
    logger.info(f"\nModel '{model_name}' training started on dataset '{ds_filename}'")
    
    models = {"LightGBM":LGBMRegressor, "XGBoost":XGBRegressor, "Ridge":Ridge, "RandomForest":RandomForestRegressor}
    model = models[model_name.split('_')[0]]

    full_params = {**params, **additional_params}
    base_model = model(**full_params)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])

    pipeline.fit(X, y)

    reg_model_name = f"{model_name} - {ds_filename}"
    model_info = mlflow.sklearn.log_model(sk_model = pipeline, name = model_name, registered_model_name=reg_model_name,
                                           input_example = X.iloc[20:21, :], params=full_params)

    client = MlflowClient()
    client.set_registered_model_alias(name=reg_model_name, alias="challenger", version= model_info.registered_model_version)
    

    logger.info(f"Model '{model_name}' trained and Registered to Production\n") 



def main(model_name, ds_filename, additional_params={"random_state":42}):
    logger.info(f"\nModel {model_name} ({ds_filename}) ---------- TUNING AND TRAINING STARTED\n")
    
    config = load_config()

    setup_mlflow(config, ds_filename)
    
    with mlflow.start_run(run_name=f"pipeline_{model_name}_{ds_filename}"):
        
        df = load_data(config, ds_filename)
        
        X, y_reg, y_cls, groups = prepare_data(df, config)
        
        result = tune_with_optuna(X_train=X, y_train=y_reg, groups=groups, config=config, model_name=model_name,
                 ds_filename=ds_filename, n_trials=config['training']['n_trials'])
        
        train_and_save_model(X=X, y=y_reg, params=result["best_params"], 
                            additional_params=additional_params, model_name=model_name, ds_filename=ds_filename)

        evaluate_model(model_name=model_name, ds_filename=ds_filename, config=config)
    
    logger.success(f"\nModel {model_name} ({ds_filename}) ---------- TUNING, TRAINING and EVALUATION COMPLETED\n")                  




if __name__ == "__main__":
    
    config = load_config()
    
    ds_filename = config['data']['dataset']

  # data = ['FD001', 'FD002', 'FD003', 'FD004']  
    models = ["Ridge", "XGBoost", "LightGBM", "RandomForest"]
      
    model = models[1]
     
    main(model_name = model, ds_filename=ds_filename, additional_params={"random_state":42})                                                                                                               

