import streamlit as st
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

st.set_page_config(page_title="AI Chat Log Summarizer", layout="wide")

# Center the title using custom CSS
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5em;
    }
    .modern-separator {
        width: 100%;
        border: none;
        border-top: 2px solid #bbb;
        margin: 2em 0 2em 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="centered-title">AI Chat Log Summarizer</div>', unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align:center; font-size:1.1rem; margin-bottom: 0.5em;">Built by <b>Mamunur Rahman Moon</b></div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <a href="https://github.com/MamunurRahmanMoon/AI_Chat_Log_Summarizer_by_Moon" target="_blank" style="text-decoration: none;">
        <img src="https://img.shields.io/badge/GitHub-Repo-black?logo=github&style=for-the-badge" alt="GitHub Repo"/>
    </a>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
Upload AI chat log file in the format:
```
User: Hello!
AI: Hi! How can I assist you today?
User: Can you explain what machine learning is?
AI: Certainly! Machine learning is a field of AI that allows systems to learn from data.
```
"""
)
# Use three columns: left, separator, right
col1, col_sep, col2 = st.columns([5, 0.2, 5])

with col_sep:
    st.markdown(
        '<div style="height: 100vh; border-left: 2px solid #bbb; margin: 0 0.5em;"></div>',
        unsafe_allow_html=True,
    )

# --- Session state for results ---
if "approach1_results" not in st.session_state:
    st.session_state["approach1_results"] = None
if "approach2_results" not in st.session_state:
    st.session_state["approach2_results"] = None

# ------------------ Approach-1 (Single File, Button Triggered) ------------------
with col1:
    st.header("Approach-1: Frequency-based (Basic Python) (Single Log)")

    uploaded_file_1 = st.file_uploader(
        "Choose a chat log (.txt) file for Approach-1", type="txt", key="approach1"
    )
    run_approach1 = st.button("Run Approach-1")

    if uploaded_file_1 and run_approach1:
        lines = uploaded_file_1.read().decode("utf-8").splitlines()

        # --- Chat Log Parsing (from notebook) ---
        messages = []
        current_speaker = None
        current_message = ""

        for line in lines:
            line = line.strip()
            if line.startswith("User:"):
                if current_speaker is not None:
                    messages.append((current_speaker, current_message.strip()))
                current_speaker = "User"
                current_message = line[len("User: ") :].strip()
            elif line.startswith("AI:"):
                if current_speaker is not None:
                    messages.append((current_speaker, current_message.strip()))
                current_speaker = "AI"
                current_message = line[len("AI: ") :].strip()
            else:
                current_message += " " + line
        if current_speaker is not None:
            messages.append((current_speaker, current_message.strip()))

        # --- Message Statistics (from notebook) ---
        user_messages = [msg for msg in messages if msg[0] == "User"]
        ai_messages = [msg for msg in messages if msg[0] == "AI"]
        exchanges_count = min(len(user_messages), len(ai_messages))

        # --- Keyword Extraction (from notebook) ---
        stop_words = set(
            [
                "the",
                "is",
                "a",
                "an",
                "and",
                "or",
                "to",
                "for",
                "of",
                "in",
                "on",
                "with",
                "can",
                "i",
                "you",
                "it",
                "its",
                "me",
                "about",
                "what",
                "we",
                "do",
                "hi",
                "why",
            ]
        )
        word_counter = Counter()
        for speaker, message in messages:
            words = message.split()
            for word in words:
                word = word.strip(".,!?").lower()
                if word not in stop_words:
                    word_counter[word] += 1
        most_frequent_words = [kw for kw, count in word_counter.most_common(5)]
        main_topic = most_frequent_words[0] if most_frequent_words else None

        # Store results in session state
        st.session_state["approach1_results"] = {
            "messages": messages,
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "exchanges_count": exchanges_count,
            "main_topic": main_topic,
            "most_frequent_words": most_frequent_words,
            "word_counter": word_counter,
        }

    # Display results if available
    if st.session_state["approach1_results"]:
        res = st.session_state["approach1_results"]
        st.subheader("Parsed Messages")
        for speaker, message in res["messages"]:
            st.write(f"**{speaker}:** {message}")
        st.markdown(f"- **Total messages:** {len(res['messages'])}")
        st.markdown(f"- **User messages:** {len(res['user_messages'])}")
        st.markdown(f"- **AI messages:** {len(res['ai_messages'])}")
        st.subheader("Summary")
        st.write(f"- The conversation had **{res['exchanges_count']} exchanges**.")
        st.write(f"- The user asked mainly about **{res['main_topic']}** and its uses.")
        st.write(f"- Most common keywords: {', '.join(res['most_frequent_words'])}")
        st.markdown("#### Top 10 Word Frequencies")
        freq_table = [
            {"Word": w, "Count": c} for w, c in res["word_counter"].most_common(10)
        ]
        st.table(freq_table)

# ------------------ Approach-2 (Multiple Files, Each Summarized Separately, Button Triggered) ------------------
with col2:
    st.header("Approach-2: TF-IDF, NLTK (Multiple Logs, Each Summarized Separately)")
    st.markdown(
        """
    Upload one or more chat log files in the same format as above.  
    Each file will be processed and summarized separately.
    """
    )
    uploaded_files_2 = st.file_uploader(
        "Choose one or more chat log (.txt) files for Approach-2",
        type="txt",
        accept_multiple_files=True,
        key="approach2",
    )
    run_approach2 = st.button("Run Approach-2")

    if uploaded_files_2 and run_approach2:
        approach2_results = []
        for uploaded_file in uploaded_files_2:
            lines = uploaded_file.read().decode("utf-8").splitlines()
            # --- Chat Log Parsing (from Approach-2 notebook) ---
            messages = []
            current_speaker = None
            current_message = ""
            for line in lines:
                line = line.strip()
                if line.startswith("User:"):
                    if current_speaker is not None:
                        messages.append((current_speaker, current_message.strip()))
                    current_speaker = "User"
                    current_message = line[len("User: ") :].strip()
                elif line.startswith("AI:"):
                    if current_speaker is not None:
                        messages.append((current_speaker, current_message.strip()))
                    current_speaker = "AI"
                    current_message = line[len("AI: ") :].strip()
                else:
                    current_message += " " + line
            if current_speaker is not None:
                messages.append((current_speaker, current_message.strip()))
            # --- Lemmatization (from Approach-2 notebook) ---
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words("english"))
            custom_stopwords = {
                "hi",
                "hello",
                "hey",
                "thanks",
                "thank",
                "please",
                "ok",
                "okay",
                "sure",
                "yes",
                "no",
                "maybe",
                "let's",
                "let us",
            }
            stop_words.update(custom_stopwords)
            lemmatized_docs = []
            for msg in messages:
                tokens = word_tokenize(msg[1])
                lemmatized = " ".join(
                    lemmatizer.lemmatize(w.lower())
                    for w in tokens
                    if w.isalpha() and w.lower() not in stop_words
                )
                lemmatized_docs.append(lemmatized)
            # --- TF-IDF (from Approach-2 notebook) ---
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(2, 3))
            tfidf_matrix = vectorizer.fit_transform(lemmatized_docs)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-5:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            # --- Message Statistics (from Approach-2 notebook) ---
            user_messages = [msg for msg in messages if msg[0] == "User"]
            ai_messages = [msg for msg in messages if msg[0] == "AI"]
            exchanges_count = min(len(user_messages), len(ai_messages))
            main_topic = top_keywords[0] if top_keywords else "No main topic found"
            tfidf_table = [
                {"Keyword": feature_names[i], "Score": round(scores[i], 3)}
                for i in scores.argsort()[-10:][::-1]
            ]
            approach2_results.append(
                {
                    "filename": uploaded_file.name,
                    "messages": messages,
                    "user_messages": user_messages,
                    "ai_messages": ai_messages,
                    "exchanges_count": exchanges_count,
                    "main_topic": main_topic,
                    "top_keywords": top_keywords,
                    "tfidf_table": tfidf_table,
                }
            )
        st.session_state["approach2_results"] = approach2_results

    # Display results if available
    if st.session_state["approach2_results"]:
        for res in st.session_state["approach2_results"]:
            st.markdown(f"---\n#### File: `{res['filename']}`")
            st.markdown("**Parsed Messages**")
            for speaker, message in res["messages"]:
                st.write(f"**{speaker}:** {message}")
            st.markdown("**Summary**")
            st.write(f"- The conversation had **{res['exchanges_count']} exchanges**.")
            st.write(
                f"- The user asked mainly about **{res['main_topic']}** and {res['top_keywords'][1] if len(res['top_keywords']) > 1 else ''}."
            )
            st.write(f"- Most common keywords: {', '.join(res['top_keywords'])}")
            st.markdown("**Top 10 TF-IDF Keywords**")
            st.table(res["tfidf_table"])
