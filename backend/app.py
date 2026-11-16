
import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify

# --------------------------------------------------
# 1. Initialize Flask application
# --------------------------------------------------

superkart_api = Flask("SuperKart")

# Load model using the path configured for deployment
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "superkart_model.joblib")
)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. Define MRP Tier boundaries
# --------------------------------------------------

MRP_BINS = [30.0, 117.73, 147.63, 177.84, 267.0]

TIER_LABELS = [
    "Tier_1_Budget",
    "Tier_2_Economy",
    "Tier_3_MidRange",
    "Tier_4_Premium"
]


def add_mrp_tier(dataframe):

    dataframe = dataframe.copy()

    dataframe["MRP_Tier"] = pd.cut(
        dataframe["Product_MRP"],
        bins=MRP_BINS,
        labels=TIER_LABELS,
        include_lowest=True
    )

    if dataframe["MRP_Tier"].isna().any():
        raise ValueError(
            "Product_MRP must be within the range 30.0 to 267.0"
        )

    return dataframe


# --------------------------------------------------
# 3. Home endpoint
# --------------------------------------------------

@superkart_api.get("/")
def home():

    return "Welcome to the SuperKart System"


# --------------------------------------------------
# 4. Single product prediction
# --------------------------------------------------

@superkart_api.post("/v1/predict")
def predict_sales():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A valid JSON payload is required"
        }), 400

    required_fields = [
        "Product_Weight",
        "Product_Sugar_Content",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type",
        "Product_Id_char",
        "Store_Age_Years",
        "Product_Type_Category"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:

        # Extract the 10 frontend features
        sample = {
            field: data[field]
            for field in required_fields
        }

        # Convert to DataFrame
        input_data = pd.DataFrame([sample])

        # Derive the 11th feature
        input_data = add_mrp_tier(input_data)

        # Generate prediction
        prediction = float(model.predict(input_data)[0])

        return jsonify({
            "Sales": round(prediction, 2)
        })

    except (TypeError, ValueError) as e:

        return jsonify({
            "error": str(e)
        }), 400

    except Exception:

        superkart_api.logger.exception(
            "Single prediction failed"
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500





# --------------------------------------------------
# 5. Batch product prediction
# --------------------------------------------------

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():

    try:

        # ------------------------------------------
        # 1. Validate uploaded CSV file
        # ------------------------------------------

        if "file" not in request.files:
            return jsonify({
                "error": "Please upload a CSV file using the file field"
            }), 400

        file = request.files["file"]

        if not file.filename:
            return jsonify({
                "error": "No file selected"
            }), 400

        # Read uploaded CSV
        original_data = pd.read_csv(file)

        # Clean column headers
        original_data.columns = (
            original_data.columns.str.strip()
        )

        if original_data.empty:
            return jsonify({
                "error": "The uploaded CSV contains no records"
            }), 400

        # ------------------------------------------
        # 2. Identify CSV format
        # ------------------------------------------

        engineered_fields = [
            "Product_Weight",
            "Product_Sugar_Content",
            "Product_Allocated_Area",
            "Product_MRP",
            "Store_Size",
            "Store_Location_City_Type",
            "Store_Type",
            "Product_Id_char",
            "Store_Age_Years",
            "Product_Type_Category"
        ]

        original_fields = [
            "Product_Id",
            "Product_Weight",
            "Product_Sugar_Content",
            "Product_Allocated_Area",
            "Product_Type",
            "Product_MRP",
            "Store_Id",
            "Store_Establishment_Year",
            "Store_Size",
            "Store_Location_City_Type",
            "Store_Type"
        ]

        uploaded_columns = set(original_data.columns)

        if set(engineered_fields).issubset(uploaded_columns):
            input_format = "engineered"

        elif set(original_fields).issubset(uploaded_columns):
            input_format = "original"

        else:
            return jsonify({
                "error": "Unrecognized CSV format",
                "expected_formats": [
                    "Assignment test CSV with 10 engineered features",
                    "Original SuperKart dataset with product and store details"
                ],
                "received_columns": original_data.columns.tolist()
            }), 400

        # ------------------------------------------
        # 3. Check availability of actual sales
        # ------------------------------------------

        has_actual_sales = (
            "Product_Store_Sales_Total"
            in original_data.columns
        )

        # ------------------------------------------
        # 4. Preserve identifiers for reporting
        # ------------------------------------------

        if input_format == "original":

            result_data = original_data[
                [
                    "Product_Id",
                    "Store_Id",
                    "Product_Type"
                ]
            ].copy()

        else:

            result_data = pd.DataFrame({
                "Record_Number": range(
                    1, len(original_data) + 1
                ),
                "Product_Id_char": original_data[
                    "Product_Id_char"
                ].values,
                "Product_Type_Category": original_data[
                    "Product_Type_Category"
                ].values
            })

        # ------------------------------------------
        # 5. Preserve actual sales if available
        # ------------------------------------------

        if has_actual_sales:

            actual_sales = pd.to_numeric(
                original_data["Product_Store_Sales_Total"],
                errors="raise"
            )

            if actual_sales.isna().any():
                raise ValueError(
                    "Product_Store_Sales_Total contains missing values"
                )

            result_data["Actual_Sales"] = (
                actual_sales.round(2)
            )

        # ------------------------------------------
        # 6. Prepare model input
        # ------------------------------------------

        # Actual sales must never be included
        # among the model prediction features.

        input_data = original_data.drop(
            columns=["Product_Store_Sales_Total"],
            errors="ignore"
        ).copy()

        # ------------------------------------------
        # 7. Feature engineering for original CSV
        # ------------------------------------------

        if input_format == "original":

            # Derive Product_Id_char
            input_data["Product_Id_char"] = (
                input_data["Product_Id"]
                .astype("string")
                .str[:2]
            )

            # Derive Store_Age_Years
            input_data["Store_Establishment_Year"] = (
                pd.to_numeric(
                    input_data["Store_Establishment_Year"],
                    errors="raise"
                )
            )

            input_data["Store_Age_Years"] = (
                2009 -
                input_data["Store_Establishment_Year"]
            )

            if (input_data["Store_Age_Years"] < 0).any():
                raise ValueError(
                    "Store_Establishment_Year cannot be later than 2009"
                )

            # Derive Product_Type_Category
            perishables = [
                "Dairy",
                "Meat",
                "Fruits and Vegetables",
                "Breakfast",
                "Breads",
                "Seafood"
            ]

            input_data["Product_Type_Category"] = (
                input_data["Product_Type"].apply(
                    lambda x: (
                        "Perishables"
                        if x in perishables
                        else "Non Perishables"
                    )
                )
            )

        # ------------------------------------------
        # 8. Derive MRP_Tier for both formats
        # ------------------------------------------

        input_data["Product_MRP"] = pd.to_numeric(
            input_data["Product_MRP"],
            errors="raise"
        )

        input_data = add_mrp_tier(input_data)

        # ------------------------------------------
        # 9. Select exact model features
        # ------------------------------------------

        feature_columns = [
            "Product_Weight",
            "Product_Sugar_Content",
            "Product_Allocated_Area",
            "Product_MRP",
            "Store_Size",
            "Store_Location_City_Type",
            "Store_Type",
            "Product_Id_char",
            "Store_Age_Years",
            "Product_Type_Category",
            "MRP_Tier"
        ]

        model_input = input_data[
            feature_columns
        ].copy()

        # ------------------------------------------
        # 10. Generate predictions
        # ------------------------------------------

        predictions = model.predict(model_input)

        result_data["Predicted_Sales"] = [
            round(float(pred), 2)
            for pred in predictions
        ]

        # ------------------------------------------
        # 11. Calculate prediction errors
        # ------------------------------------------

        if has_actual_sales:

            result_data["Prediction_Error"] = (
                result_data["Predicted_Sales"]
                - result_data["Actual_Sales"]
            ).round(2)

            result_data["Absolute_Error"] = (
                result_data["Prediction_Error"]
                .abs()
                .round(2)
            )

        # ------------------------------------------
        # 12. Calculate summary statistics
        # ------------------------------------------

        summary = {

            "total_records": int(len(result_data)),

            "average_predicted_sales": round(
                float(result_data["Predicted_Sales"].mean()), 2
            ),

            "highest_predicted_sales": round(
                float(result_data["Predicted_Sales"].max()), 2
            ),

            "lowest_predicted_sales": round(
                float(result_data["Predicted_Sales"].min()), 2
            )
        }

        if has_actual_sales:

            summary["average_actual_sales"] = round(
                float(result_data["Actual_Sales"].mean()), 2
            )

            summary["mean_absolute_error"] = round(
                float(result_data["Absolute_Error"].mean()), 2
            )

        # ------------------------------------------
        # 13. Return structured JSON response
        # ------------------------------------------

        return jsonify({

            "input_format": input_format,

            "actual_sales_available": has_actual_sales,

            "summary": summary,

            "predictions": result_data.to_dict(
                orient="records"
            )

        }), 200

    except (TypeError, ValueError) as e:

        return jsonify({
            "error": str(e)
        }), 400

    except Exception:

        superkart_api.logger.exception(
            "Batch prediction failed"
        )

        return jsonify({
            "error": "Batch prediction failed"
        }), 500



# --------------------------------------------------
# 6. Run application
# --------------------------------------------------

if __name__ == "__main__":

    superkart_api.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
