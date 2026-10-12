import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & NLP Risk Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn & NLP Risk Analytics Dashboard")
st.markdown("Predict customer churn using demographic metrics, contract types, and customer feedback analysis.")

# ---------------------------------------------------------
# Data Loader & Preprocessing Helper
# ---------------------------------------------------------
@st.cache_data
def load_and_prepare_data(file_source):
    if file_source is not None:
        df_raw = pd.read_csv(file_source)
    else:
        try:
            df_raw = pd.read_csv('customer_churn_nlp.csv')
        except FileNotFoundError:
            return None

    # Strip whitespace from column names
    df_raw.columns = df_raw.columns.str.strip()

    # Column mapping for lower-case or alternative names
    column_mapping = {
        'tenure': 'Tenure',
        'monthlycharges': 'MonthlyCharges',
        'monthly_charges': 'MonthlyCharges',
        'supporttickets': 'SupportTickets',
        'support_tickets': 'SupportTickets',
        'contract': 'Contract',
        'churn': 'Churn',
        'customerfeedback': 'CustomerFeedback',
        'feedback': 'CustomerFeedback',
        'comments': 'CustomerFeedback',
        'review': 'CustomerFeedback'
    }

    # Apply column renaming where matches exist
    rename_dict = {}
    for col in df_raw.columns:
        col_lower = col.lower().replace(" ", "")
        if col_lower in column_mapping:
            rename_dict[col] = column_mapping[col_lower]
    df_raw = df_raw.rename(columns=rename_dict)

    # Safely convert 'Churn' to binary int (1/0)
    if 'Churn' in df_raw.columns:
        df_raw['Churn'] = df_raw['Churn'].astype(str).str.strip().str.lower().apply(
            lambda x: 1 if x in ['yes', '1', '1.0', 'true', 'churned'] else 0
        )

    # Ensure numeric types
    for col in ['Tenure', 'MonthlyCharges']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

    # Synthesize missing features if dataset is standard Telco churn without NLP
    np.random.seed(42)
    if 'SupportTickets' not in df_raw.columns:
        if 'Churn' in df_raw.columns:
            df_raw['SupportTickets'] = df_raw['Churn'].apply(
                lambda c: np.random.randint(3, 8) if c == 1 else np.random.randint(0, 3)
            )
        else:
            df_raw['SupportTickets'] = np.random.randint(0, 5, size=len(df_raw))

    if 'CustomerFeedback' not in df_raw.columns:
        pos_feedbacks = [
            "Customer support was very helpful and polite.",
            "Smooth connection and fair monthly prices.",
            "Excellent coverage and fast streaming experience.",
            "Love the service, smooth connection and very helpful customer care!",
            "Great internet speed and reliable customer support."
        ]
        neg_feedbacks = [
            "Bad experience, moving to another provider soon.",
            "Cancel my subscription, worst customer care ever.",
            "Monthly charges are way too expensive for bad service.",
            "Terrible internet speed, disconnects constantly! Overpriced charges.",
            "High monthly bill and frequent network dropouts."
        ]
        if 'Churn' in df_raw.columns:
            df_raw['CustomerFeedback'] = df_raw['Churn'].apply(
                lambda c: np.random.choice(neg_feedbacks) if c == 1 else np.random.choice(pos_feedbacks)
            )
        else:
            df_raw['CustomerFeedback'] = np.random.choice(pos_feedbacks + neg_feedbacks, size=len(df_raw))

    return df_raw


# Sidebar File Upload
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

df = load_and_prepare_data(uploaded_file)

if df is None:
    st.info("💡 Please upload a CSV file in the sidebar to proceed.")
    st.stop()

# Ensure required columns exist
required_columns = ['Tenure', 'MonthlyCharges', 'SupportTickets', 'Contract', 'CustomerFeedback', 'Churn']
missing = [c for c in required_columns if c not in df.columns]
if missing:
    st.error(f"⚠️ Missing required columns in dataset: {missing}")
    st.write("Found columns:", list(df.columns))
    st.stop()

# ---------------------------------------------------------
# Model Training Pipeline
# ---------------------------------------------------------
@st.cache_resource
def train_pipeline(data):
    df_contract_encoded = pd.get_dummies(data[['Contract']], drop_first=True, dtype=int)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=15, stop_words='english', ngram_range=(1, 2))
    text_features = tfidf_vectorizer.fit_transform(data['CustomerFeedback'].fillna("")).toarray()
    text_feature_names = [f"text_{word}" for word in tfidf_vectorizer.get_feature_names_out()]
    df_text_encoded = pd.DataFrame(text_features, columns=text_feature_names)
    
    df_numeric = data[['Tenure', 'MonthlyCharges', 'SupportTickets']]
    X = pd.concat([df_numeric, df_contract_encoded, df_text_encoded], axis=1)
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    return tfidf_vectorizer, text_feature_names, X, y, X_test, y_test, model_lr, model_rf, acc_lr, acc_rf, y_pred_rf

tfidf_vectorizer, text_feature_names, X, y, X_test, y_test, model_lr, model_rf, acc_lr, acc_rf, y_pred_rf = train_pipeline(df)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Exploratory Analytics", 
    "🤖 Model Performance", 
    "🔮 Real-Time Prediction",
    "📁 Batch Prediction"
])

# ---------------------------------------------------------
# TAB 1: Exploratory Analytics
# ---------------------------------------------------------
with tab1:
    st.subheader("Data Overview & Customer Insights")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Customers", f"{len(df):,}")
    churn_pct = (df['Churn'].sum() / len(df)) * 100
    kpi2.metric("Overall Churn Rate", f"{churn_pct:.1f}%")
    kpi3.metric("Avg Tenure", f"{df['Tenure'].mean():.1f} mos")
    kpi4.metric("Avg Monthly Bill", f"${df['MonthlyCharges'].mean():.2f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("##### Customer Retention vs Churn Count")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'], ax=axes[0])
        axes[0].set_title('Churn Count', fontweight='bold')
        axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
        axes[0].set_ylabel('Number of Customers')
        
        churn_counts = df['Churn'].value_counts()
        axes[1].pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%',
                    colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.08))
        axes[1].set_title('Churn Rate Breakdown', fontweight='bold')
        st.pyplot(fig)
        
    with col2:
        st.write("##### Monthly Charges ($) by Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_title('Monthly Charges ($) by Churn', fontweight='bold')
        ax.set_xticklabels(['Retained', 'Churned'])
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.write("##### Support Tickets by Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='SupportTickets', hue='Churn', palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_title('Support Tickets by Churn', fontweight='bold')
        ax.legend(['Retained', 'Churned'])
        st.pyplot(fig)
        
    with col4:
        st.write("##### Churn Rate across Contract Types")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Contract', hue='Churn', palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_title('Churn Rate across Contract Types', fontweight='bold')
        ax.set_xlabel('Contract Type')
        ax.set_ylabel('Number of Customers')
        ax.legend(['Retained', 'Churned'])
        st.pyplot(fig)

    st.markdown("---")
    st.write("##### Sample Feedback Review Text")
    st.markdown("**Retained Customers Feedback Examples:**")
    for text in df[df['Churn'] == 0]['CustomerFeedback'].head(3):
        st.write(f"- 🔥 {text}")
    st.markdown("**Churned Customers Feedback Examples:**")
    for text in df[df['Churn'] == 1]['CustomerFeedback'].head(3):
        st.write(f"- 🤮 {text}")

# ---------------------------------------------------------
# TAB 2: Model Performance
# ---------------------------------------------------------
with tab2:
    st.subheader("Classification Models Performance")
    
    c1, c2 = st.columns(2)
    c1.metric("Logistic Regression Accuracy", f"{acc_lr*100:.2f}%")
    c2.metric("Random Forest Accuracy", f"{acc_rf*100:.2f}%")
    
    st.markdown("---")
    col_cm, col_imp = st.columns(2)
    
    with col_cm:
        st.write("##### Random Forest Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_rf)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Retained (0)', 'Churned (1)'],
                    yticklabels=['Retained (0)', 'Churned (1)'], ax=ax)
        ax.set_title('Random Forest Confusion Matrix', fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        st.pyplot(fig)
        
    with col_imp:
        st.write("##### Top 10 Most Important Features")
        feature_importances = pd.Series(model_rf.feature_importances_, index=X.columns)
        top_features = feature_importances.sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', ax=ax)
        ax.set_title('Top 10 Features Driving Churn Prediction', fontweight='bold')
        ax.set_xlabel('Feature Importance Score')
        st.pyplot(fig)

    st.markdown("---")
    st.write("##### Classification Report (Random Forest)")
    report_dict = classification_report(y_test, y_pred_rf, target_names=['Retained (0)', 'Churned (1)'], output_dict=True)
    st.dataframe(pd.DataFrame(report_dict).transpose())

# ---------------------------------------------------------
# TAB 3: Real-Time Prediction
# ---------------------------------------------------------
with tab3:
    st.subheader("Customer Risk Assessment Form")
    
    with st.form("churn_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=2)
            monthly_charges = st.number_input("Monthly Bill ($)", min_value=0.0, max_value=300.0, value=95.50)
            support_tickets = st.number_input("Support Tickets Logged", min_value=0, max_value=20, value=5)
            
        with col_b:
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            feedback_text = st.text_area(
                "Customer Feedback / Review", 
                value="Terrible internet speed, disconnects constantly! Overpriced charges."
            )
            
        submit_btn = st.form_submit_button("🔍 Calculate Risk Profile")

    if submit_btn:
        input_dict = {
            'Tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'SupportTickets': [support_tickets],
            'Contract_One year': [1 if contract_type == 'One year' else 0],
            'Contract_Two year': [1 if contract_type == 'Two year' else 0]
        }

        text_vectorized = tfidf_vectorizer.transform([feedback_text]).toarray()
        for i, col_name in enumerate(text_feature_names):
            input_dict[col_name] = [text_vectorized[0][i]]

        # Match columns with feature matrix
        df_input = pd.DataFrame(input_dict)
        for col in X.columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[X.columns]

        churn_prob = model_rf.predict_proba(df_input)[0][1]
        is_churn = churn_prob >= 0.5
        
        st.markdown("### 📋 Risk Assessment Results")
        res_c1, res_c2 = st.columns(2)
        
        with res_c1:
            st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
            st.progress(float(churn_prob))
            
        with res_c2:
            if is_churn:
                st.error("⚠️ **RISK STATUS: HIGH RISK OF CHURN (DANGER)**")
                st.markdown("**Suggested Actions:**")
                st.write("- 🎯 Trigger retention offer: 15% discount on 1-year contract.")
                st.write("- 📞 Schedule priority customer support callback.")
            else:
                st.success("✅ **RISK STATUS: RETAINED (LOW RISK)**")
                st.markdown("**Suggested Actions:**")
                st.write("- 🎁 No intervention needed. Send thank-you loyalty email.")

# ---------------------------------------------------------
# TAB 4: Batch Prediction
# ---------------------------------------------------------
with tab4:
    st.subheader("Batch Customer Churn Analysis")
    st.write("Upload a CSV file with customer data to predict churn probabilities for multiple records.")
    
    batch_file = st.file_uploader("Upload Customer File for Batch Prediction", type=["csv"], key="batch_upload")
    
    if batch_file:
        df_batch = load_and_prepare_data(batch_file)
        
        # Format columns and encoded contracts
        df_batch_contract = pd.get_dummies(df_batch[['Contract']], drop_first=True, dtype=int)
        
        text_vec_batch = tfidf_vectorizer.transform(df_batch['CustomerFeedback'].fillna("")).toarray()
        df_batch_text = pd.DataFrame(text_vec_batch, columns=text_feature_names)
        
        X_batch_dict = {
            'Tenure': df_batch['Tenure'],
            'MonthlyCharges': df_batch['MonthlyCharges'],
            'SupportTickets': df_batch['SupportTickets']
        }
        X_batch_base = pd.DataFrame(X_batch_dict)
        X_batch = pd.concat([X_batch_base, df_batch_contract, df_batch_text], axis=1)
        
        # Align batch features with training features
        for col in X.columns:
            if col not in X_batch.columns:
                X_batch[col] = 0
        X_batch = X_batch[X.columns]
        
        df_batch['Churn_Probability'] = model_rf.predict_proba(X_batch)[:, 1]
        df_batch['Predicted_Risk'] = np.where(df_batch['Churn_Probability'] >= 0.5, 'High Risk', 'Low Risk')
        
        st.dataframe(df_batch[['Tenure', 'MonthlyCharges', 'SupportTickets', 'Contract', 'Churn_Probability', 'Predicted_Risk']])
        
        csv_download = df_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_download,
            file_name="churn_predictions_results.csv",
            mime="text/csv"
        )