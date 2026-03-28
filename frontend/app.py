import streamlit as st
import requests
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="DataSense AI",
    layout="wide",
    page_icon="📊"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Configuration")
st.sidebar.write("Configure your ML pipeline")

problem_type = st.sidebar.selectbox(
    "Select Problem Type",
    ["classification", "regression", "clustering"]
)

target_column = st.sidebar.text_input(
    "Target Column (leave empty for clustering)"
)

# ------------------ MAIN TITLE ------------------
st.title("📊 DataSense AI")
st.markdown("### Automated Data Cleaning + Model Recommendation System")

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

df_preview = None

if file is not None:
    df_preview = pd.read_csv(file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df_preview.head(), use_container_width=True)

    # Basic dataset info
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df_preview.shape[0])
    col2.metric("Columns", df_preview.shape[1])
    col3.metric("Missing Values", df_preview.isnull().sum().sum())

# ------------------ RUN BUTTON ------------------
if st.button("🚀 Run Analysis"):

    if file is None:
        st.error("Please upload a CSV file first")

    else:
        try:
            with st.spinner("⏳ Running analysis..."):
                response = requests.post(
                    "https://ml-automation-dashboard.onrender.com/train",
                    files={"file": (file.name, file.getvalue(), "text/csv")},
                    data={
                        "problem_type": problem_type,
                        "target_column": target_column
                    }
                )

            if response.status_code != 200:
                st.error("❌ Backend error")
                st.write(response.text)
            else:
                result = response.json()

                if "error" in result:
                    st.error(result["error"])

                else:
                    st.success("✅ Analysis Completed")


                

                    # ------------------ METRICS ------------------
                    col1, col2 = st.columns(2)
                    best = result["best_model"]

                    if isinstance(best, list):
                        model_name = best[0]
                        model_score = best[1]
                    else:
                        model_name = best
                        model_score = None

                    col1.metric("🏆 Best Model", model_name)

                    if model_score is not None:
                        col2.metric("📊 Score", round(model_score, 3))
                        best = result["best_model"]

                        if isinstance(best, list):
                            model_name = best[0]
                            model_score = best[1]
                        else:
                            model_name = best
                            model_score = None

                        col1.metric("🏆 Best Model", model_name)

                        if model_score is not None:
                            col2.metric("📈 Score", round(model_score, 4))

                    # ------------------ MODEL RESULTS ------------------
                    st.subheader("📊 Model Comparison")

                    df_results = pd.DataFrame(
                        result["results"],
                        columns=["Model", "Score"]
                    )

                    df_results = df_results.sort_values(by="Score", ascending=False)

                    st.dataframe(df_results, use_container_width=True)
                    st.bar_chart(df_results.set_index("Model"))

                    # ------------------ CLEANING STEPS ------------------
                    if "cleaning_steps" in result:
                        st.subheader("🧹 Data Cleaning Steps")
                        for step in result["cleaning_steps"]:
                            st.write("✔️", step)

                    # ------------------ EXPLANATION ------------------
                    st.subheader("🧠 Model Explanation")
                    st.info(result["explanation"])

                    # ------------------ FEATURE IMPORTANCE ------------------
                    st.subheader("📈 Feature Importance")

                    importance = result.get("feature_importance", None)

                    if isinstance(importance, dict):
                        imp_df = pd.DataFrame(
                            list(importance.items()),
                            columns=["Feature", "Importance"]
                        )

                        imp_df = imp_df.sort_values(by="Importance", ascending=False)

                        st.dataframe(imp_df, use_container_width=True)
                        st.bar_chart(imp_df.set_index("Feature"))
                    else:
                        st.write("No feature importance available")

                    # ------------------ DOWNLOAD REPORT ------------------
                    st.subheader("⬇️ Download Report")

                    report = f"""
                    DataSense AI Report

                    Best Model: {model_name}
                    Score: {model_score}

                    Explanation:
                    {result['explanation']}
                    """

                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name="report.txt",
                        mime="text/plain"
                    )

                    # ------------------ DOWNLOAD DATASET ------------------
                    if df_preview is not None:
                        st.download_button(
                            label="📥 Download Dataset",
                            data=df_preview.to_csv(index=False),
                            file_name="dataset.csv",
                            mime="text/csv"
                        )

        except requests.exceptions.ConnectionError:
            st.error("❌ Backend not reachable. Start FastAPI server.")


        # ------------------ HISTORY SECTION ------------------
st.markdown("---")
st.subheader("📜 Previous Runs")

show_history = st.checkbox("Show History")

if show_history:
    try:
        response = requests.get("https://ml-automation-dashboard.onrender.com/history")

        history = response.json()

        if isinstance(history, dict) and "error" in history:
            st.error(history["error"])

        elif len(history) == 0:
            st.warning("No history available yet")

        else:
            df_history = pd.DataFrame(history)

            st.dataframe(df_history, use_container_width=True)

            st.subheader("📊 Score Trend")

            if "model" in df_history.columns and "score" in df_history.columns:
                st.bar_chart(df_history.set_index("model")["score"])
            else:
                st.write("Chart data not available")

    except requests.exceptions.ConnectionError:
        st.error("❌ Backend not reachable")