import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from io import BytesIO

st.set_page_config(page_title="Credit Risk — Streamlit", layout="wide")

st.title("Credit Risk Analysis App")
st.markdown("Upload your `credit_risk_dataset.csv` or use the sample upload to train and evaluate a model.")

@st.cache_data
def load_sample():
    # If you want to include a small sample shipped with the app, place it here.
    return None

def preprocess_engineer(df: pd.DataFrame):
    df = df.copy()
    # Basic cleaning: ensure expected columns exist
    expected = ['person_age','person_income','person_home_ownership','person_emp_length',
                'loan_intent','loan_grade','loan_amnt','loan_int_rate','loan_status',
                'loan_percent_income','cb_person_default_on_file','cb_person_cred_hist_length']
    # create DebtIncomeRatio
    if 'loan_amnt' in df.columns and 'person_income' in df.columns:
        df["DebtIncomeRatio"] = df["loan_amnt"] / df["person_income"].replace({0: np.nan})
    else:
        df["DebtIncomeRatio"] = np.nan

    # EmpLengthCategory
    if 'person_emp_length' in df.columns:
        bins_emp_length = [-1, 2, 5, 10, df['person_emp_length'].max() + 1]
        labels_emp_length = ['0-2 Years', '3-5 Years', '6-10 Years', '10+ Years']
        df['EmpLengthCategory'] = pd.cut(df['person_emp_length'].fillna(-1), bins=bins_emp_length,
                                         labels=labels_emp_length, right=False)
    else:
        df['EmpLengthCategory'] = 'Unknown'

    # InterestRateLevel
    if 'loan_int_rate' in df.columns:
        bins_int_rate = [df['loan_int_rate'].min() - 1, 7, 12, df['loan_int_rate'].max() + 1]
        labels_int_rate = ['Low', 'Medium', 'High']
        df['InterestRateLevel'] = pd.cut(df['loan_int_rate'].fillna(0), bins=bins_int_rate,
                                         labels=labels_int_rate, right=False)
    else:
        df['InterestRateLevel'] = 'Unknown'

    # Normalize some common boolean/text flags
    if 'cb_person_default_on_file' in df.columns:
        df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y':1,'N':0}).fillna(0)
    else:
        df['cb_person_default_on_file'] = 0

    return df

def build_pipeline(df: pd.DataFrame):
    numeric_features = []
    for col in ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate',
                'loan_percent_income','cb_person_cred_hist_length','DebtIncomeRatio']:
        if col in df.columns:
            numeric_features.append(col)

    categorical_features = []
    for col in ['person_home_ownership','loan_intent','loan_grade','EmpLengthCategory','InterestRateLevel']:
        if col in df.columns:
            categorical_features.append(col)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    return clf, numeric_features, categorical_features

def train_and_eval(df: pd.DataFrame):
    df = preprocess_engineer(df)
    if 'loan_status' not in df.columns:
        st.error("Dataset must include 'loan_status' column (target).")
        return None, None

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    clf, num_cols, cat_cols = build_pipeline(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)

    with st.spinner("Training model..."):
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    return clf, dict(acc=acc, report=report, cm=cm, X_test=X_test, y_test=y_test)

# UI layout
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload credit_risk_dataset.csv", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("No file uploaded. If your repo contains credit_risk_dataset.csv, you can push it to the repo and deploy, or upload here.")
        sample = load_sample()
        data = sample if sample is not None else None

    if data is not None:
        st.write("Preview of data:")
        st.dataframe(data.head())

with col2:
    if data is not None:
        st.header("Feature summary")
        st.write(data.describe(include='all').T)

if st.button("Train model") and (data is not None):
    result = train_and_eval(data)
    if result is None:
        st.stop()
    clf, metrics = result
    st.success(f"Training complete — accuracy: {metrics['acc']:.4f}")

    st.subheader("Classification report")
    st.text(metrics['report'])

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature importances — try to extract if the classifier supports it
    try:
        # we need feature names after preprocessing
        pre = clf.named_steps['preprocessor']
        num_cols = pre.transformers_[0][2]
        cat_pipeline = pre.transformers_[1][1]
        cat_encoder = cat_pipeline.named_steps['onehot']
        cat_cols = pre.transformers_[1][2]
        encoded_cat_cols = list(cat_encoder.get_feature_names_out(cat_cols))
        feat_names = list(num_cols) + encoded_cat_cols
        importances = clf.named_steps['classifier'].feature_importances_
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(20)
        st.subheader("Top feature importances")
        st.bar_chart(fi)
    except Exception as e:
        st.warning("Could not extract feature importances: " + str(e))

    # Offer model download
    buffer = BytesIO()
    joblib.dump(clf, buffer)
    buffer.seek(0)
    st.download_button("Download trained model (.pkl)", buffer, file_name="credit_risk_model.pkl")

# Single prediction interactive form
st.header("Predict single applicant")
with st.form("predict_form"):
    st.write("Enter feature values (leave blank for defaults if your dataset has different columns).")
    v_age = st.number_input("person_age", value=30)
    v_income = st.number_input("person_income", value=50000)
    v_home = st.selectbox("person_home_ownership", options=['RENT','OWN','MORTGAGE','OTHER'])
    v_emp_length = st.number_input("person_emp_length", value=2.0)
    v_loan_intent = st.selectbox("loan_intent", options=['PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION','OTHER'])
    v_loan_grade = st.selectbox("loan_grade", options=['A','B','C','D','E','F','G'])
    v_loan_amnt = st.number_input("loan_amnt", value=5000)
    v_loan_int_rate = st.number_input("loan_int_rate", value=12.0)
    v_loan_percent_income = st.number_input("loan_percent_income", value=0.2)
    v_cb_default = st.selectbox("cb_person_default_on_file", options=['N','Y'])
    v_cb_cred_len = st.number_input("cb_person_cred_hist_length", value=3)

    submitted = st.form_submit_button("Run prediction")
    if submitted:
        # We need a trained model in the session — ask user to upload or train one
        st.info("To predict you need a trained model. Either: 1) Train a model using the 'Train model' button above, or 2) Upload a pre-trained .pkl file below.")
        model_file = st.file_uploader("Or upload trained model (.pkl)", type=['pkl','joblib'])
        model = None
        if model_file:
            try:
                model = joblib.load(model_file)
            except Exception as e:
                st.error("Failed to load model: " + str(e))
        else:
            st.warning("No model provided — prediction cannot run until a model is trained or uploaded.")
        if model is not None:
            sample = pd.DataFrame([{
                'person_age': v_age,
                'person_income': v_income,
                'person_home_ownership': v_home,
                'person_emp_length': v_emp_length,
                'loan_intent': v_loan_intent,
                'loan_grade': v_loan_grade,
                'loan_amnt': v_loan_amnt,
                'loan_int_rate': v_loan_int_rate,
                'loan_percent_income': v_loan_percent_income,
                'cb_person_default_on_file': v_cb_default,
                'cb_person_cred_hist_length': v_cb_cred_len
            }])
            sample = preprocess_engineer(sample)
            pred = model.predict(sample)[0]
            st.write("Predicted loan_status:", int(pred))
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample)[0]
                st.write("Probabilities:", proba)
