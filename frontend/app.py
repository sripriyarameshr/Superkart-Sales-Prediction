
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Page title
st.title("SuperKart System")
st.write(
    "Enter the product and store details below to predict the total sales."
)

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Single Prediction
if st.button("Predict", type='primary'):

    response = requests.post(
        f"{BACKEND_URL}/v1/predict",
        json=product_data
    )

    if response.status_code == 200:
        result = response.json()
        predicted_sales = result["Sales"]
        st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:.2f}")
    else:
        st.error("Unable to connect to the prediction API.")


# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

st.subheader("Batch Sales Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing product and store details",
    type=["csv"],
    help="The CSV should contain the 10 engineered model input features."
)

if uploaded_file is not None:

    st.write(f"**Uploaded file:** {uploaded_file.name}")

    if st.button("Predict for Batch", type="primary"):

        try:
            # Read the original uploaded data
            input_df = pd.read_csv(uploaded_file)

            # Submit the CSV file to the Flask backend
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                },
                timeout=120
            )

            if response.status_code == 200:

                results = response.json()

                # Extract prediction records
                predictions = results["predictions"]

                prediction_df = pd.DataFrame(predictions)

                # Check that every input record has a prediction
                if len(input_df) != len(prediction_df):
                    raise ValueError(
                        "The number of predictions does not match "
                        "the number of uploaded records."
                    )

                # Append predictions to the original input data
                output_df = input_df.copy()

                output_df["Predicted_Sales"] = (
                    prediction_df["Predicted_Sales"].astype(float)
                )

                # --------------------------------------
                # Prediction Summary
                # --------------------------------------

                st.success(
                    "Batch predictions completed successfully!"
                )

                st.caption(
                    f"Input format: {results.get('input_format', 'N/A')} "
                    "| Actual sales available: "
                    f"{results.get('actual_sales_available', False)}"
                )

                st.subheader("Prediction Summary")

                total_records = len(output_df)

                average_sales = output_df["Predicted_Sales"].mean()

                highest_sales = output_df["Predicted_Sales"].max()

                lowest_sales = output_df["Predicted_Sales"].min()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Records",
                    f"{total_records:,}"
                )

                col2.metric(
                    "Average Predicted Sales",
                    f"₹{average_sales:,.2f}"
                )

                col3.metric(
                    "Highest Predicted Sales",
                    f"₹{highest_sales:,.2f}"
                )

                col4.metric(
                    "Lowest Predicted Sales",
                    f"₹{lowest_sales:,.2f}"
                )

                # --------------------------------------
                # Individual Prediction Results
                # --------------------------------------

                st.subheader("Individual Sales Predictions")

                st.dataframe(
                    output_df.style.format({
                        "Product_Weight": "{:.2f}",
                        "Product_Allocated_Area": "{:.3f}",
                        "Product_MRP": "₹{:,.2f}",
                        "Product_Store_Sales_Total": "₹{:,.2f}",
                        "Predicted_Sales": "₹{:,.2f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # --------------------------------------
                # Product Category Analysis
                # --------------------------------------

                if "Product_Type_Category" in output_df.columns:

                    st.subheader("Product Category Sales Summary")

                    category_summary = (
                        output_df.groupby("Product_Type_Category")
                        ["Predicted_Sales"]
                        .agg(
                            Total_Predicted_Sales="sum",
                            Average_Predicted_Sales="mean",
                            Product_Count="count"
                        )
                        .reset_index()
                    )

                    st.dataframe(
                        category_summary.style.format({
                            "Total_Predicted_Sales": "₹{:,.2f}",
                            "Average_Predicted_Sales": "₹{:,.2f}",
                            "Product_Count": "{:,.0f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                # --------------------------------------
                # Download Prediction Report
                # --------------------------------------

                st.subheader("Download Prediction Report")

                csv_data = output_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Batch Prediction Report",
                    data=csv_data,
                    file_name="SuperKart_Batch_Predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

                st.caption(
                    "The downloadable report includes all original "
                    "input columns and the additional Predicted_Sales "
                    "column. Actual sales values were not supplied."
                )

            else:

                try:
                    error_details = response.json()
                except ValueError:
                    error_details = response.text

                st.error(
                    f"Batch prediction failed: {error_details}"
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to connect to the prediction API: {e}"
            )

        except (KeyError, ValueError, TypeError) as e:

            st.error(
                f"Unable to process batch prediction results: {e}"
            )

        except Exception as e:

            st.error(
                f"An unexpected error occurred: {e}"
            )
