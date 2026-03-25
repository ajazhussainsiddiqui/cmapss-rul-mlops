import streamlit as st 
import requests  
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np


# root_url = "http://127.0.0.1:8000"  
root_url = "https://ajazhussainsiddiqui-predictive-maintenance-cmapss.hf.space" 



def plot_scatter(prediction):
    fig, ax = plt.subplots() 
    ax.scatter(prediction['cycle'], prediction['predicted_RUL'])
    ax.plot([min(prediction["cycle"]),max(prediction["cycle"])], [max(prediction["predicted_RUL"]),min(prediction["predicted_RUL"])], "r--", label="Perfect prediction")
    ax.set_title(f"RUL prediction over cycles [engine_id: {prediction['engine_id'][0]}]")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Predicted RUL")
    ax.legend()
    plt.close()
    return fig


def shap_plot(shap_data):
    shap_values = np.array(shap_data['shap_values'])       
    base_value = np.array([shap_data['base_value']])       
    X = np.array(shap_data['X'])                          
    feature_names = shap_data['feature_names']

        
    force_description = """
                        <details>
                            <summary><b>Force Plot</b></summary>
                            Visualizes how each feature contributes to pushing the model output from the base value (expected prediction) to the final prediction for a single instance(last row of inputs here). 
                            Features in red push the prediction higher, blue pushes it lower.
                        </details>
                        """
    st.markdown(force_description, unsafe_allow_html=True)
    shap.force_plot(base_value=base_value, shap_values=np.round(shap_values[-1], 3), features=np.round(X[-1], 3), feature_names=feature_names, matplotlib=True) 
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    force_description = """
                        <details>
                            <summary><b>Waterfall Plot</b></summary>
                            Decomposes the prediction for a single instance (last row here) into contributions from each feature. 
                            It starts at the base value and adds/subtracts feature effects to arrive at the final prediction.(Almost similar to force plot)
                        </details>
                        """
    st.markdown(force_description, unsafe_allow_html=True)
    shap.plots.waterfall(shap.Explanation(values=shap_values[-1], base_values=base_value, data=X[-1], feature_names=feature_names))
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    force_description = """
                        <details>
                            <summary><b>SHAP Summary Plot</b></summary>
                            Shows the distribution of SHAP values for each feature across all data. Features are ordered by importance. 
                            Each point represents a single prediction; color indicates feature value (red high, blue low). It gives a global view of feature impact.
                        </details>
                        """
    st.markdown(force_description, unsafe_allow_html=True)        
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    force_description = """
                        <details>
                            <summary><b>SHAP Summary Plot(bar plot)</b></summary>
                            A bar chart of mean absolute SHAP values per feature. Shows the global feature importance.
                        </details>
                        """
    st.markdown(force_description, unsafe_allow_html=True) 
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()    

    force_description = """
                        <details>
                            <summary><b>SHAP Heatmap</b></summary>
                           Displays the SHAP values of all features for every sample (here we use last 500 rows max) in a heatmap. Rows are features, columns are samples. 
                           It reveals patterns of feature impact across the dataset.
                        </details>
                        """
    st.markdown(force_description, unsafe_allow_html=True)
    shap.plots.heatmap(shap.Explanation(values=shap_values[-500:], feature_names=feature_names), show=False)
    st.pyplot(plt.gcf())
    plt.close()

    




st.set_page_config(
    page_title="ML Algo - Jet Engine RUL Predictor",
    page_icon="🤔"
)

st.title("Predictive Maintenance: RUL Estimation for Aircraft Engines ")
st.write('''End‑to‑end ML system that forecasts Remaining Useful Life (RUL) of turbofan engines using sensor data. Built with a FastAPI backend, Streamlit frontend, and MLflow model registry – complete with SHAP explanations for transparent predictions. *([GitHub](https://github.com/ajazhussainsiddiqui/cmapss-rul-mlops))*''')
st.markdown("Data source: [NASA CMAPSS Turbofan Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)")
st.divider()
response = requests.get(url=f"{root_url}/models")
models_list = response.json()


model_name = st.selectbox("Models", options=models_list, index=3)
engine_id = st.number_input(f"Engine ID", value=50)
include_shap = st.checkbox(label="Include SHAP plots")

st.info("Upload a sensor data file or manually edit the table to get RUL predictions.")


tab1, tab2 = st.tabs(["Manual Input", "Upload File"])


with tab1:
    st.caption("Manual Data Entry edit the table below. Add or delete rows as needed.")
    
    sample_data = {'engine_id': [50, 50, 50, 50, 50, 50, 50, 50], 'cycle': [132, 133, 134, 135, 136, 137, 138, 139], 'op_setting_1': [0.0016, 0.0007, 0.0006, -0.0001, -0.0022, -0.0005, 0.0015, -0.0026], 'op_setting_2': [0.0004, 0.0001, 0.0002, -0.0002, 0.0004, -0.0003, -0.0001, 0.0004], 'op_setting_3': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 's1': [518.67, 518.67, 518.67, 518.67, 518.67, 518.67, 518.67, 518.67], 's2': [642.96, 642.67, 642.93, 642.74, 643.06, 643.4, 643.02, 642.43], 's3': [1594.74, 1591.92, 1593.23, 1594.17, 1593.9, 1597.69, 1591.62, 1596.31], 's4': [1408.37, 1416.33, 1416.01, 1417.15, 1416.76, 1415.35, 1418.24, 1415.06], 's5': [14.62, 14.62, 14.62, 14.62, 14.62, 14.62, 14.62, 14.62], 's6': [21.61, 21.61, 21.61, 21.61, 21.61, 21.61, 21.61, 21.61], 's7': [552.69, 553.38, 553.83, 553.24, 552.51, 553.33, 552.77, 552.47], 's8': [2388.13, 2388.16, 2388.18, 2388.17, 2388.13, 2388.16, 2388.2, 2388.15], 's9': [9056.22, 9059.65, 9051.88, 9054.01, 9052.19, 9058.81, 9046.78, 9045.46], 's10': [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], 's11': [47.57, 47.74, 47.74, 47.58, 47.77, 47.85, 47.82, 47.8], 's12': [520.65, 520.86, 520.75, 521.09, 521.13, 521.17, 520.63, 520.86], 's13': [2388.12, 2388.14, 2388.16, 2388.09, 2388.15, 2388.21, 2388.19, 2388.21], 's14': [8129.08, 8128.1, 8137.09, 8140.16, 8136.0, 8129.42, 8127.21, 8126.65], 's15': [8.4422, 8.4802, 8.4662, 8.4844, 8.4085, 8.4866, 8.4579, 8.4827], 's16': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03], 's17': [394, 394, 394, 394, 395, 395, 394, 394], 's18': [2388, 2388, 2388, 2388, 2388, 2388, 2388, 2388], 's19': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 's20': [38.73, 38.73, 38.7, 38.92, 38.74, 38.85, 38.51, 38.57], 's21': [23.2956, 23.204, 23.2741, 23.1984, 23.3566, 23.2295, 23.2438, 23.2745]}


    edited_data = st.data_editor(sample_data, num_rows="dynamic")

    if st.button("Predict_RUL"):
        with st.spinner("Processing..."): 
            try:
                response = requests.post(url=f"{root_url}/predict", json=edited_data, params={"engine_id":engine_id, "model_name":model_name, "include_shap":include_shap})
                data = response.json()
                df_pred = pd.DataFrame(data['df_pred'])
            except Exception as e:
                st.error(f"Oops! API request fail {e}")
                st.write(data)
                st.stop()   
                        
            prediction = df_pred[['predicted_RUL', 'cycle','engine_id']]
            col1, col2, col3 = st.columns([0.4, 0.1, 0.5])
            with col1:
                st.write(f"RUL prediction for last {len(prediction)} cycle of engine_id- {engine_id}")
                st.dataframe(prediction, hide_index=True)

            with col3:
                with st.expander("Show full Dataframe output"):    
                    st.write(df_pred) 
                st.divider()
                
                st.pyplot(plot_scatter(prediction))
            if include_shap:
                with st.expander("Show SHAP plots of input data", expanded=True):
                    shap_plot(data["shap_data"])   



with tab2:
    st.caption("Upload Sensor Data File")

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'], help="Upload a file with sensor readings. The file should contain the required columns.")
    if st.button("predict_RUL"):
        with st.spinner("Processing..."): 
            try:
                response = requests.post(url = f"{root_url}/predict_from_uploaded_file", files={"file": uploaded_file}, 
                                        params={"engine_id": engine_id, "model_name":model_name, "include_shap":include_shap})
                data = response.json()
                df_pred = pd.DataFrame(data["df_pred"])          
            except Exception as e:
                st.error(f"Oops! API request fail {e}")
                st.write(data)
                st.stop() 
            
            prediction = df_pred[['predicted_RUL', 'cycle','engine_id']]
            total_prediction = len(prediction)
            col1, col2, col3 = st.columns([0.4, 0.1, 0.5])
            with col1:
                st.write(f"RUL prediction for last {total_prediction} cycle of engine_id - {engine_id}")
                st.dataframe(prediction, hide_index=True)
                 
            with col3:
                with st.expander("Show full Dataframe output"):    
                    st.write(df_pred) 
                st.divider()

                st.pyplot(plot_scatter(prediction))
            if include_shap: 
                with st.expander("Show SHAP plots of input data", expanded=True):
                    shap_plot(data["shap_data"])     
   
           
    