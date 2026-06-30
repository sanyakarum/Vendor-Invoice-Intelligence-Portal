import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\91948\Desktop\Data Analytics\Invoice ML project\freight_cost_prediction\models\predict_freight_model.pkl"

def load_model(model_path : str = MODEL_PATH):
    """
    LOAD TRAINED FREIGHT COST PREDICTION MODEL.
    """

    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    PREDICT FREIGHT COST FOR NEW VENDOR INVOICES.

    parameters
    -----------
    input_data: dict


    Returns
    ---------
    pd.DataFrame with predicted freight cost
    """

    model= load_model()
    input_df = pd.DataFrame(input_data)
    input_df = input_df[['Quantity', 'Dollars']]
    input_df['Predicted_Freight'] = model.predict(input_df).round(2)
    return input_df

if __name__ == "__main__":
    # example inference rin (local testing)
    sample_data = {
        "Quantity": [1200, 800, 250, 50],
        "Dollars" : [18500,9000, 3000, 200]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)
    
    