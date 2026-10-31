
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Landmines Classification",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Landmines Detection - Decision Tree Classification")
st.write("Machine Learning based Landmines Classification Project")

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload landmines.csv or a ZIP containing a CSV file",
    type=["csv", "zip"]
)

df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file) as z:
                csv_files = [
                    f for f in z.namelist()
                    if f.lower().endswith(".csv")
                    and not f.startswith("__MACOSX/")
                ]

                if csv_files:
                    df = pd.read_csv(z.open(csv_files[0]))
                else:
                    st.error("No CSV file found inside ZIP.")

        else:
            df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Error reading dataset: {e}")

if df is not None:

    st.success("Dataset uploaded successfully!")

    st.header("2. Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("3. Dataset Information")
    st.write(df.describe(include="all"))

    st.header("4. Select Target Column")

    target = st.selectbox(
        "Choose the column to predict",
        df.columns
    )

    if st.button("Train Decision Tree Model"):

        data = df.dropna(subset=[target]).copy()

        X = data.drop(columns=[target])
        y = data[target]

        # Remove columns containing only missing values
        X = X.dropna(axis=1, how="all")

        if X.shape[1] == 0:
            st.error("No usable feature columns found.")

        elif y.nunique() < 2:
            st.error("Target column must contain at least two classes.")

        else:
            numeric_cols = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_cols = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            transformers = []

            if numeric_cols:
                transformers.append(
                    ("numeric", numeric_pipeline, numeric_cols)
                )

            if categorical_cols:
                transformers.append(
                    ("categorical", categorical_pipeline, categorical_cols)
                )

            preprocessor = ColumnTransformer(
                transformers=transformers
            )

            model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=5,
                    random_state=42
                ))
            ])

            try:
                stratify_y = None

                if y.value_counts().min() >= 2:
                    stratify_y = y

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=stratify_y
                )

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                st.header("5. Model Performance")

                st.metric(
                    "Decision Tree Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

                st.write("Training samples:", len(X_train))
                st.write("Testing samples:", len(X_test))

                st.header("6. Confusion Matrix")

                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=model.classes_
                ).plot(ax=ax, cmap="Blues", xticks_rotation=45)

                st.pyplot(fig)
                plt.close(fig)

                st.header("7. Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(
                    pd.DataFrame(report).transpose()
                )

                st.header("8. Feature Importance")

                feature_names = (
                    model.named_steps["preprocessing"]
                    .get_feature_names_out()
                )

                importance = (
                    model.named_steps["classifier"]
                    .feature_importances_
                )

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                }).sort_values(
                    by="Importance",
                    ascending=False
                )

                st.bar_chart(
                    importance_df.set_index("Feature").head(15)
                )

                st.header("9. Sample Predictions")

                results = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": y_pred
                })

                st.dataframe(results.head(20))

                st.success("Decision Tree model trained successfully!")

            except Exception as e:
                st.error(f"Model training error: {e}")

else:
    st.info("Please upload your landmines CSV or ZIP dataset to begin.")
