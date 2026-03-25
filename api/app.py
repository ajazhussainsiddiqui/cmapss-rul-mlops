from fastapi import APIRouter, UploadFile, HTTPException
from pydantic import BaseModel
from src.predict import predict, load_config, get_shap_values
from typing import List
import pandas as pd
import io
import os



router = APIRouter()

class SensorBatchInput(BaseModel):
    engine_id: List[int]
    cycle: List[int]
    op_setting_1: List[float]
    op_setting_2: List[float]
    op_setting_3: List[float]
    s1: List[float]
    s2: List[float]
    s3: List[float]
    s4: List[float]
    s5: List[float]
    s6:List[float]
    s7: List[float]
    s8: List[float]
    s9: List[float]
    s10: List[float]
    s11: List[float]
    s12: List[float]
    s13: List[float]
    s14: List[float]
    s15: List[float]
    s16: List[float]
    s17: List[float]
    s18: List[float]
    s19: List[float]
    s20: List[float]
    s21: List[float]


    class Config:
        json_schema_extra = {
            "example": 
            {'engine_id': [1, 1, 1, 1, 1, 1, 1, 1], 'cycle': [132, 133, 134, 135, 136, 137, 138, 139], 'op_setting_1': [0.0016, 0.0007, 0.0006, -0.0001, -0.0022, -0.0005, 0.0015, -0.0026], 'op_setting_2': [0.0004, 0.0001, 0.0002, -0.0002, 0.0004, -0.0003, -0.0001, 0.0004], 'op_setting_3': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 's1': [518.67, 518.67, 518.67, 518.67, 518.67, 518.67, 518.67, 518.67], 's2': [642.96, 642.67, 642.93, 642.74, 643.06, 643.4, 643.02, 642.43], 's3': [1594.74, 1591.92, 1593.23, 1594.17, 1593.9, 1597.69, 1591.62, 1596.31], 's4': [1408.37, 1416.33, 1416.01, 1417.15, 1416.76, 1415.35, 1418.24, 1415.06], 's5': [14.62, 14.62, 14.62, 14.62, 14.62, 14.62, 14.62, 14.62], 's6': [21.61, 21.61, 21.61, 21.61, 21.61, 21.61, 21.61, 21.61], 's7': [552.69, 553.38, 553.83, 553.24, 552.51, 553.33, 552.77, 552.47], 's8': [2388.13, 2388.16, 2388.18, 2388.17, 2388.13, 2388.16, 2388.2, 2388.15], 's9': [9056.22, 9059.65, 9051.88, 9054.01, 9052.19, 9058.81, 9046.78, 9045.46], 's10': [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], 's11': [47.57, 47.74, 47.74, 47.58, 47.77, 47.85, 47.82, 47.8], 's12': [520.65, 520.86, 520.75, 521.09, 521.13, 521.17, 520.63, 520.86], 's13': [2388.12, 2388.14, 2388.16, 2388.09, 2388.15, 2388.21, 2388.19, 2388.21], 's14': [8129.08, 8128.1, 8137.09, 8140.16, 8136.0, 8129.42, 8127.21, 8126.65], 's15': [8.4422, 8.4802, 8.4662, 8.4844, 8.4085, 8.4866, 8.4579, 8.4827], 's16': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03], 's17': [394, 394, 394, 394, 395, 395, 394, 394], 's18': [2388, 2388, 2388, 2388, 2388, 2388, 2388, 2388], 's19': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 's20': [38.73, 38.73, 38.7, 38.92, 38.74, 38.85, 38.51, 38.57], 's21': [23.2956, 23.204, 23.2741, 23.1984, 23.3566, 23.2295, 23.2438, 23.2745]}
        }



config = load_config()

@router.post("/predict")
def predict_from_json( data: SensorBatchInput, engine_id:int=1, model_name:str="LightGBM - FD001",  include_shap: bool = False):
            
        try:
            df = pd.DataFrame(data.dict())
            if len(df)<6:
                raise HTTPException(status_code=400, detail=f"Insuffient data to process, in engine_id-{engine_id} current data sample {len(df)} and minimum 6 required for prediction")
            prediction = predict(df, engine_id ,model_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{e}")
        df_pred = df[df['engine_id']==engine_id]
        df_pred = df_pred.iloc[-len(prediction):, :]
        df_pred["predicted_RUL"] = prediction
        response = {"df_pred":df_pred.to_dict()}
        
        if include_shap:
            try:
                response["shap_data"] = get_shap_values(df, model_name)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"SHAP plots not supported for '{model_name}' (Potential cause).\n\n{e}")
        return response



@router.post("/predict_from_uploaded_file")
async def predict_from_file(file:UploadFile, engine_id:int=20, model_name:str="LightGBM", include_shap:bool=False):
        
    contents = await file.read()
    
    try:
        df = pd.read_csv(io.BytesIO(contents), sep=r"\s+", header=None, names=config["data"]["col_names"])
        if len(df)<6:
                raise HTTPException(status_code=400, detail=f"Insuffient data to process, in engine_id-{engine_id} current data sample and {len(df)} minimum 6 required for prediction")
        prediction = predict(df, engine_id ,model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")    
    df_pred = df[df['engine_id']==engine_id]
    df_pred = df_pred.iloc[-len(prediction):, :]
    df_pred["predicted_RUL"] = prediction
    response =  {"df_pred":df_pred.to_dict()}
    
    if include_shap:
        try:
            response["shap_data"] = get_shap_values(df, model_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SHAP plots not supported for '{model_name}' (Potential cause).\n\n{e}")    
    return response






@router.get("/models")
def load_models_names():
    
    os.makedirs("models", exist_ok=True)
    
    return  [model.split('.pkl')[0] for model in os.listdir("models")]

