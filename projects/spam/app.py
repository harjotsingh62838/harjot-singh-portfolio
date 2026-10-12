import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import warnings

# NLTK Imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Scikit-Learn Imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# NLTK Resource Downloader
# ---------------------------------------------------------
@st.cache_resource
def download_nltk_resources():
    for res in ['wordnet', 'omw-1.4', 'stopwords', 'punkt', 'punkt_tab']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Classifier & NLP Explorer",
    page_icon="📩",
    layout="wide"
)

st.title("📩 SMS Spam Detection & NLP Playground")
st.markdown("Explore text preprocessing techniques, BoW/TF-IDF vectorization, and train Naive Bayes spam classification models.")

# ---------------------------------------------------------
# Preprocessing Helper Functions
# ---------------------------------------------------------
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_stem(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

def preprocess_text_lem(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ---------------------------------------------------------
# Sidebar Data Upload & Dataset Prep
# ---------------------------------------------------------
st.sidebar.header("📂 Dataset Config")
uploaded_file = st.sidebar.file_uploader("Upload 'spam.csv'", type=["csv"])

@st.cache_data
def load_and_prep_data(file_source):
    df_raw = None
    if file_source is not None:
        try:
            df_raw = pd.read_csv(file_source, encoding='latin-1')
        except Exception:
            df_raw = pd.read_csv(file_source, encoding='utf-8')
    else:
        try:
            df_raw = pd.read_csv('spam.csv', encoding='latin-1')
        except FileNotFoundError:
            return None

    df_raw.columns = df_raw.columns.str.strip()

    # Align column names for Kaggle Spam CSVs or custom CSVs
    if 'v1' in df_raw.columns and 'v2' in df_raw.columns:
        df_raw = df_raw[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    elif 'Category' in df_raw.columns and 'Message' in df_raw.columns:
        df_raw = df_raw[['Category', 'Message']].rename(columns={'Category': 'label', 'Message': 'text'})

    # Target Mapping
    if 'label' in df_raw.columns:
        df_raw['is_spam'] = df_raw['label'].astype(str).str.strip().str.lower().apply(
            lambda x: 1 if x in ['spam', '1', '1.0', 'true'] else 0
        )

    return df_raw

df = load_and_prep_data(uploaded_file)

if df is None or 'text' not in df.columns or 'is_spam' not in df.columns:
    st.info("💡 Please upload `spam.csv` (containing SMS text and labels) in the sidebar to proceed.")
    st.stop()

# ---------------------------------------------------------
# Cached Model Pipeline
# ---------------------------------------------------------
@st.cache_resource
def train_models(data):
    df_proc = data.copy()
    df_proc['clean_stem'] = df_proc['text'].apply(preprocess_text_stem)
    df_proc['clean_lem'] = df_proc['text'].apply(preprocess_text_lem)

    # Train / Test Split for Stemmed Pipeline
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        df_proc['clean_stem'], df_proc['is_spam'], test_size=0.20, random_state=42, stratify=df_proc['is_spam']
    )

    tfidf_stem = TfidfVectorizer(max_features=2500)
    X_train_stem_vec = tfidf_stem.fit_transform(X_train_s)
    X_test_stem_vec = tfidf_stem.transform(X_test_s)

    nb_stem = MultinomialNB()
    nb_stem.fit(X_train_stem_vec, y_train_s)
    y_pred_stem = nb_stem.predict(X_test_stem_vec)
    acc_stem = accuracy_score(y_test_s, y_pred_stem)

    # Train / Test Split for Lemmatized Pipeline
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        df_proc['clean_lem'], df_proc['is_spam'], test_size=0.20, random_state=42, stratify=df_proc['is_spam']
    )

    tfidf_lem = TfidfVectorizer(max_features=2500)
    X_train_lem_vec = tfidf_lem.fit_transform(X_train_l)
    X_test_lem_vec = tfidf_lem.transform(X_test_l)

    nb_lem = MultinomialNB()
    nb_lem.fit(X_train_lem_vec, y_train_l)
    y_pred_lem = nb_lem.predict(X_test_lem_vec)
    acc_lem = accuracy_score(y_test_l, y_pred_lem)

    return (
        df_proc, tfidf_stem, nb_stem, acc_stem, y_test_s, y_pred_stem,
        tfidf_lem, nb_lem, acc_lem, y_test_l, y_pred_lem
    )

(df, tfidf_stem, nb_stem, acc_stem, y_test_s, y_pred_stem,
 tfidf_lem, nb_lem, acc_lem, y_test_l, y_pred_lem) = train_models(df)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛠️ Step-by-Step NLP Demo", 
    "📊 Vectorization (BoW vs TF-IDF)", 
    "🤖 Spam Classifier Performance",
    "🔮 Real-Time SMS Predictor",
    "📁 Batch Spam Scanner"
])

# ---------------------------------------------------------
# TAB 1: NLP Preprocessing Step-by-Step
# ---------------------------------------------------------
with tab1:
    st.subheader("Interactive Text Preprocessing Pipeline")
    sample_input = st.text_input(
        "Enter text to test NLTK operations:",
        value="Hello World! We are learning text preprocessing today!"
    )

    if sample_input:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 1. Lowercasing")
            t_lower = sample_input.lower()
            st.code(t_lower)

            st.markdown("##### 2. Tokenization")
            t_tokens = word_tokenize(t_lower)
            st.write(t_tokens)

            st.markdown("##### 3. Punctuation Removal")
            t_no_punct = [w for w in t_tokens if w not in string.punctuation]
            st.write(t_no_punct)

        with col2:
            st.markdown("##### 4. Stopwords Removal")
            t_no_stop = [w for w in t_no_punct if w not in stop_words]
            st.write(t_no_stop)

            st.markdown("##### 5. Stemming (PorterStemmer)")
            t_stem = [stemmer.stem(w) for w in t_no_stop]
            st.write(t_stem)

            st.markdown("##### 6. Lemmatization (WordNet)")
            t_lem = [lemmatizer.lemmatize(w) for w in t_no_stop]
            st.write(t_lem)

# ---------------------------------------------------------
# TAB 2: BoW vs TF-IDF Demo
# ---------------------------------------------------------
with tab2:
    st.subheader("Bag-of-Words (CountVectorizer) vs. TF-IDF Representation")
    
    toy_corpus = [
        "The cat chased the mouse.",
        "The dog chased the cat.",
        "The bird sang in the tree."
    ]
    st.markdown("**Sample Toy Corpus:**")
    for doc in toy_corpus:
        st.write(f"- {doc}")

    col_bow, col_tfidf = st.columns(2)
    with col_bow:
        st.write("##### 1. Bag-of-Words (Counts)")
        cv = CountVectorizer()
        bow_mat = cv.fit_transform(toy_corpus)
        df_bow = pd.DataFrame(bow_mat.toarray(), columns=cv.get_feature_names_out())
        st.dataframe(df_bow)

    with col_tfidf:
        st.write("##### 2. TF-IDF (Scaled Weights)")
        tv = TfidfVectorizer()
        tfidf_mat = tv.fit_transform(toy_corpus)
        df_tfidf = pd.DataFrame(tfidf_mat.toarray(), columns=tv.get_feature_names_out()).round(3)
        st.dataframe(df_tfidf)

# ---------------------------------------------------------
# TAB 3: Model Performance & Comparison
# ---------------------------------------------------------
with tab3:
    st.subheader("Spam Detection Models Comparison")

    m1, m2 = st.columns(2)
    m1.metric("Stemming + Naive Bayes Accuracy", f"{acc_stem*100:.2f}%")
    m2.metric("Lemmatization + Naive Bayes Accuracy", f"{acc_lem*100:.2f}%")

    st.markdown("---")
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.write("##### Confusion Matrix (Stemming Pipeline)")
        cm_s = confusion_matrix(y_test_s, y_pred_stem)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham (0)', 'Spam (1)'],
                    yticklabels=['Ham (0)', 'Spam (1)'], ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        st.pyplot(fig)

    with col_cm2:
        st.write("##### Confusion Matrix (Lemmatization Pipeline)")
        cm_l = confusion_matrix(y_test_l, y_pred_lem)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_l, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham (0)', 'Spam (1)'],
                    yticklabels=['Ham (0)', 'Spam (1)'], ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        st.pyplot(fig)

    st.markdown("---")
    st.write("##### Classification Report (Lemmatization Pipeline)")
    report_dict = classification_report(y_test_l, y_pred_lem, target_names=['Ham', 'Spam'], output_dict=True)
    st.dataframe(pd.DataFrame(report_dict).transpose())

# ---------------------------------------------------------
# TAB 4: Real-Time SMS Predictor
# ---------------------------------------------------------
with tab4:
    st.subheader("Custom Message Spam Classifier")

    custom_text = st.text_area(
        "Enter SMS message text to evaluate:",
        value="Congratulations! You have won a free $1000 cash prize. Click here now to claim your reward."
    )

    if st.button("🚨 Analyze Message Risk"):
        cleaned_msg = preprocess_text_lem(custom_text)
        tfidf_msg = tfidf_lem.transform([cleaned_msg])
        pred_label = nb_lem.predict(tfidf_msg)[0]
        prob_spam = nb_lem.predict_proba(tfidf_msg)[0][1]

        st.markdown("### Prediction Result")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("Spam Probability", f"{prob_spam*100:.1f}%")
            st.progress(float(prob_spam))

        with res_col2:
            if pred_label == 1:
                st.error("🚨 **FLAGGED AS SPAM**")
                st.write("Warning: This message contains common spam/phishing patterns.")
            else:
                st.success("✅ **SAFE / HAM MESSAGE**")
                st.write("This message appears legitimate.")

# ---------------------------------------------------------
# TAB 5: Batch Prediction
# ---------------------------------------------------------
with tab5:
    st.subheader("Batch Message Spam Scanner")
    st.write("Upload a CSV file with a text column to evaluate multiple messages at once.")

    batch_csv = st.file_uploader("Upload CSV for Batch Processing", type=["csv"], key="batch_spam_upload")

    if batch_csv:
        try:
            try:
                df_b = pd.read_csv(batch_csv, encoding='utf-8')
            except UnicodeDecodeError:
                batch_csv.seek(0)
                df_b = pd.read_csv(batch_csv, encoding='latin-1')

            text_col = next((c for c in df_b.columns if 'text' in c.lower() or 'msg' in c.lower() or 'message' in c.lower() or 'v2' in c.lower()), df_b.columns[0])
            st.info(f"Using column **'{text_col}'** for text analysis.")

            df_b['clean_text'] = df_b[text_col].astype(str).apply(preprocess_text_lem)
            vec_b = tfidf_lem.transform(df_b['clean_text'])
            df_b['Is_Spam_Prediction'] = nb_lem.predict(vec_b)
            df_b['Spam_Probability'] = nb_lem.predict_proba(vec_b)[:, 1]
            df_b['Status'] = np.where(df_b['Is_Spam_Prediction'] == 1, 'Spam 🚨', 'Ham ✅')

            st.dataframe(df_b[[text_col, 'Status', 'Spam_Probability']])

            csv_download = df_b.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Results CSV",
                data=csv_download,
                file_name="spam_predictions_output.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing batch file: {e}")