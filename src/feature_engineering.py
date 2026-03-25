import pandas as pd
import yaml
import os
from src.data_preprocessing import preprocess
from loguru import logger


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
    


def get_sensor_cols(df_processed, config):

    exclude = config['data']['non_sensor_cols']
    sensors_list = [col for col in df_processed.columns if col not in exclude]
    
    return sensors_list    



def add_rolling_window_features(df, config, sensor_cols) :

    windows = config["feature_engineering"]["rolling_windows"]
    new_cols = {}
    for sensor in sensor_cols:
        for w in windows:
            grp = df.groupby("engine_id")[sensor]

            new_cols[f"{sensor}_roll_mean_{w}"] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            
            new_cols[f"{sensor}_roll_std_{w}"] = grp.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info(f"Added rolling features, shape: {df.shape}")
    return df




def add_lag_features(df, config, sensor_cols):

    lag_steps = config["feature_engineering"]["lag_steps"]
    
    new_cols = {}
    for sensor in sensor_cols:
        for lag in lag_steps:
            new_cols[f"{sensor}_lag_{lag}"] = df.groupby("engine_id")[sensor].shift(lag)
    
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info(f"Added lag features, shape: {df.shape}")
    return df



def add_rate_of_change(df, sensor_cols):

    new_cols = {}
    for sensor in sensor_cols:
        new_cols[f"{sensor}_diff"] = df.groupby("engine_id")[sensor].diff().fillna(0)
    
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info(f"Added rate of change features, shape: {df.shape}")
    return df




def build_features(df, config):
    sensor_cols = get_sensor_cols(df, config) 
    df = add_rolling_window_features(df, config, sensor_cols)
    df = add_lag_features(df, config, sensor_cols)
    df = add_rate_of_change(df, sensor_cols)
    old_df=len(df)
    df = df.dropna()
    logger.info(f"Data's length change after droping NaN rows : ({old_df} to {len(df)})")    
    
    return df




def save_features_df(df, config, ds_filename):
    processed_path = config["data"]["processed_path"]
    filepath = os.path.join(processed_path, f"train_{ds_filename}_processed.csv")
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved: {filepath}")



def feature_engineering(config, ds_filename):
    logger.info(f"Feature engineering START ({ds_filename})...")     
    
    df_processed = preprocess(ds_filename, config)
    logger.info(f"Before features engineering, df shape: {df_processed.shape}")
    df = build_features(df_processed, config)
    save_features_df(df, config, ds_filename)
    logger.info(f"After features engineering, dataset shape: {df.shape}")
    
    logger.info(f"Feature engineering COMPLETE ({ds_filename})...")


def main():
    
    config = load_config()
    ds_filename = config['data']['dataset']

    feature_engineering(config, ds_filename)



if __name__ == "__main__":
    
    main()
