import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Set Streamlit page title
st.set_page_config(page_title="Employee Productivity Dashboard", layout="wide")

# Function to load dataset
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Sidebar: File uploader
st.sidebar.header("Upload Employee Productivity Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Load dataset
    df = load_data(uploaded_file)

    # Preprocessing
    df.dropna(inplace=True)
    df['work_mode'] = df['work_mode'].map({'Remote': 1, 'In-Office': 0, 'Hybrid': 2})  # Updated mapping
    productivity_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df['productivity_level'] = df['productivity_level'].map(productivity_mapping)

    # Display dataset
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Show summary statistics
    st.write("### Data Summary")
    st.write(df.describe())

    # Correlation Heatmap
    st.write("### Feature Correlation")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Boxplot: Work Mode vs. Productivity
    st.write("### Work Mode vs. Productivity Level")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x="work_mode", y="productivity_level", data=df)
    ax.set_xticklabels(["In-Office", "Remote", "Hybrid"])
    st.pyplot(fig)

    # Productivity Prediction Model
    st.write("### Predict Productivity Level")
    X = df[["hours_worked", "tasks_completed", "experience", "work_mode"]]
    y = df["productivity_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Performance
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write(f"**R2 Score:** {r2:.2f}")

    # Feature Importance
    st.write("### Feature Importance")
    feature_importance = pd.Series(model.coef_, index=X.columns)
    fig, ax = plt.subplots(figsize=(6,4))
    feature_importance.plot(kind='barh', ax=ax)
    st.pyplot(fig)

else:
    st.write("### Upload a dataset to get started!")

