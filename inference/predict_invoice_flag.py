import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\91948\Desktop\Data Analytics\Invoice ML project\invoice_flagging\models\predict_flag_invoice.pkl"

SCALER_PATH = r"C:\Users\91948\Desktop\Data Analytics\Invoice ML project\invoice_flagging\models\scaler.pkl"


def load_model(model_path=MODEL_PATH):
    """
    Load trained Random Forest classifier.
    """
    return joblib.load(model_path)


def load_scaler(scaler_path=SCALER_PATH):
    """
    Load saved StandardScaler.
    """
    return joblib.load(scaler_path)


def predict_invoice_flag(input_data):
    """
    Predict invoice risk flag.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    DataFrame with prediction
    """

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    # Scale input features
    input_scaled = scaler.transform(input_df)

    # Predict
    input_df["Predicted_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [120],
        "invoice_dollars": [15000],
        "Freight": [250],
        "total_item_quantity": [120],
        "total_item_dollars": [15000]
    }

    prediction = predict_invoice_flag(sample_data)

    print("\nPrediction Result:")
    print(prediction)