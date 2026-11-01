
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Landmine Classification",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Landmine Classification System")
st.write("Decision Tree Classification using V, H and S features")

# 1. Load dataset
st.header("1. Upload Dataset")

if Path("landmines.csv").exists():
    df = pd.read_csv("landmines.csv")
    st.success("Dataset loaded successfully!")
else:
    uploaded_file = st.file_uploader(
        "Upload landmines.csv",
        type=["csv"]
    )
    if uploaded_file is None:
        st.info("Please upload your landmines.csv file.")
        st.stop()
    df = pd.read_csv(uploaded_file)

# 2. Display dataset
st.header("2. Dataset Overview")

st.write("Dataset shape:", df.shape)
st.dataframe(df.head())

# 3. Prepare features and target
required_columns = ["V", "H", "S", "M"]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset must contain V, H, S and M columns.")
    st.stop()

X = df[["V", "H", "S"]].apply(
    pd.to_numeric, errors="coerce"
)

y = df["M"]

valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(str)

if y.nunique() < 2:
    st.error("Target M must contain at least two classes.")
    st.stop()

# 4. Model settings
st.header("3. Decision Tree Settings")

criterion = st.selectbox(
    "Select criterion",
    ["gini", "entropy"]
)

max_depth = st.slider(
    "Select Maximum Depth",
    min_value=1,
    max_value=20,
    value=5
)

# 5. Train model
if st.button("Train Decision Tree Model"):

    stratify_value = (
        y if y.value_counts().min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.session_state["model"] = model
    st.session_state["accuracy"] = accuracy
    st.session_state["labels"] = sorted(y.unique())
    st.session_state["matrix"] = confusion_matrix(
        y_test,
        predictions,
        labels=st.session_state["labels"]
    )
    st.session_state["report"] = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

# 6. Model performance
if "model" in st.session_state:

    st.header("4. Model Performance")

    st.metric(
        "Decision Tree Accuracy",
        f"{st.session_state['accuracy'] * 100:.2f}%"
    )

    st.write("Training samples:", int(len(X) * 0.8))
    st.write("Testing samples:", len(X) - int(len(X) * 0.8))

    st.header("5. Confusion Matrix")

    labels = st.session_state["labels"]

    matrix_df = pd.DataFrame(
        st.session_state["matrix"],
        index=labels,
        columns=labels
    )

    st.dataframe(matrix_df)

    st.header("6. Classification Report")
    st.text(st.session_state["report"])

    # 7. Feature importance
    st.header("7. Feature Importance")

    importance = pd.DataFrame({
        "Feature": ["V", "H", "S"],
        "Importance": st.session_state["model"].feature_importances_
    })

    st.bar_chart(
        importance.set_index("Feature")
    )

    # 8. User prediction
    st.header("8. Predict Landmine Class")

    st.write("Enter the input values below:")

    with st.form("prediction_form"):

        v = st.number_input(
            "Enter V value",
            min_value=float(X["V"].min()),
            max_value=float(X["V"].max()),
            value=float(X["V"].median())
        )

        h = st.number_input(
            "Enter H value",
            min_value=float(X["H"].min()),
            max_value=float(X["H"].max()),
            value=float(X["H"].median())
        )

        s = st.number_input(
            "Enter S value",
            min_value=float(X["S"].min()),
            max_value=float(X["S"].max()),
            value=float(X["S"].median())
        )

        submitted = st.form_submit_button(
            "Predict Landmine Class"
        )

    if submitted:

        input_data = pd.DataFrame(
            [[v, h, s]],
            columns=["V", "H", "S"]
        )

        result = st.session_state["model"].predict(input_data)

        st.success(
            f"Predicted Landmine Class: {result[0]}"
        )

        st.write("Input values:")
        st.dataframe(input_data)

st.caption("Landmine Classification Project | Decision Tree ML")
