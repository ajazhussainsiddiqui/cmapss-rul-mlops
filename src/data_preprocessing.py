import pandas as pd
import yaml
import os
from loguru import logger




def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)



def load_raw_data(config, ds_filename):

    cols_names = config["data"]["col_names"]

    raw_path = config["data"]["raw_path"]
    filepath = os.path.join(raw_path, f"{ds_filename}")
    
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=cols_names)
    logger.info(f"Load raw data's shape: {df.shape}")
    return df



def compute_rul(df):

    df['RUL'] = df.groupby('engine_id')['cycle'].transform('max') - df['cycle']  

    return df



def clip_rul(df, config) :
    
    rul_clip = config["feature_engineering"]["rul_clip"]
    df["RUL"] = df["RUL"].clip(upper=rul_clip)
    return df



def save_const_cols(df, config, ds_filename):
     
     df_variance = df.var()
     const_cols = df_variance[df_variance <= 1e-10].index.to_list()
     
     if const_cols:
        processed_path = config["data"]["processed_path"]
        os.makedirs(processed_path, exist_ok=True)
        with open(f"{processed_path}constant_cols_{ds_filename}.txt", 'w') as f:
            f.write('\n'.join(const_cols))
            



def drop_low_variance_sensors(df, config, ds_filename):
    processed_path = config["data"]["processed_path"]
    full_filepath = f"{processed_path}constant_cols_{ds_filename}.txt"

    try:
        with open(full_filepath, 'r') as f:
            const_cols = f.read()
        const_cols = const_cols.split('\n')        
        df = df.drop(columns=const_cols)
        logger.info(f"Dropped low-variance sensors: {const_cols}")
        return df
    except FileNotFoundError:
        logger.warning("NO low variance columns existing")
    return df     



def add_failure_label(df, config):

    threshold = config["training"]["early_failure_threshold"]
    df["will_fail_soon"] = (df["RUL"] <= threshold).astype(int)
    return df





def preprocess(ds_filename, config):
    logger.info("Data preprocessing START...")
    
    df = load_raw_data(config, f"train_{ds_filename}.txt")
    df = compute_rul(df)
    df = clip_rul(df, config)
    save_const_cols(df, config, ds_filename)
    df = drop_low_variance_sensors(df, config, ds_filename)
    df = add_failure_label(df, config)

    logger.info("Data preprocessing COMPLETED...")
    return df

   

def main():
    
    config = load_config()
    ds_filename = f"{config['data']['dataset']}"

    df_processed = preprocess(ds_filename=ds_filename, config=config)
        
    logger.info(f"processed data shape: {df_processed.shape}")
    



if __name__ == "__main__":

    main() 


    