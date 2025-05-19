# AI Chat Log Summarizer

An app for summarizing AI chat logs using two approaches:
- **Approach-1:** Frequency-based keyword extraction (basic Python, single file)
- **Approach-2: (Bonus)** TF-IDF with NLTK lemmatization and n-grams (multiple files, each summarized separately)

---

# My Thought Process

- 1. First of all, worked with the input file and spent the most of the time getting the input in a efficient representation

- 2. Then, **Approach-1** `ai_chat_log_summarizer(Approach-1).ipynb` is just using simple python

- 3. As per the requirement for the **Bonus Part** built the **Approach-2** `ai_chat_log_summarizer(Approach-2).ipynb` using the power of **TF-IDF** and **NLTK Library**
       * For better keyword extraction a **custom stopword list**, **lemmatized docs** were created
         
- 4. Multiple log file handling added `ai_chat_log_summarizer(Approach-2)_multiple_log.ipynb`: As there were no specific requirement, So I built like Multiple chat log file will be processed at a time in bulk and give summarization for each log

    
> **NLTK Library** for:
* removing stopwords
* tokenizing 
* lemmatization to get the root word resulting better word extraction

> **TF-IDF** for:
* Finding words that are not just frequent, but also important/unique to the document.
* It reduces the weight of common words that appear in many documents.
---

## 🚀 Features

- **Upload chat logs** in plain text format.
- **Approach-1:** Analyze a single chat log for most frequent keywords and summary.
- **Approach-2:** Analyze one or more chat logs, each summarized separately using TF-IDF and lemmatization(using NLTK).
- **Side-by-side comparison** of both approaches.

---

## 📦 Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- (see `requirements.txt` for details)

---

## 🛠️ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/MamunurRahmanMoon/AI_Chat_Log_Summarizer_by_Moon.git
    cd AI_Chat_Log_Summarizer_by_Moon/app
    ```

2. **Install dependencies:**
    ```bash
    pip install -r ../requirements.txt
    ```

---

## 🏃‍♂️ Running the App

```bash
streamlit run app.py
```

The app will open in your browser.

---

## 📄 Chat Log Format

Each chat log should be a `.txt` file with lines like:

```
User: Hello!
AI: Hi! How can I assist you today?
User: Can you explain what machine learning is?
AI: Certainly! Machine learning is a field of AI that allows systems to learn from data.
```

---

## 🖥️ Usage

- **Approach-1 (left):**
    - Upload a single `.txt` chat log.
    - Click **Run Approach-1**.
    - See parsed messages, stats, top keywords, and summary.

- **Approach-2 (right):**
    - Upload one or more `.txt` chat logs.
    - Click **Run Approach-2**.
    - Each file is parsed, lemmatized, analyzed with TF-IDF, and summarized separately.

- **Both approaches can be run independently and results remain visible.**

---

## 🌐 Deployment

### Streamlit Community Cloud

- Link: 

### CI/CD

- Streamlit Cloud automatically redeploys app on every push to the selected branch.
- For extra CI (tests, linting),will add a GitHub Actions workflow in future if needed.


---

## Purpose
This project is a task given by **QTec Solutions Ltd.** for the role "Junior Python Developer (AI Focus)"

## 🤝 Credits

**Built by [Mamunur Rahman Moon](https://github.com/MamunurRahmanMoon)**

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black?logo=github&style=for-the-badge)](https://github.com/MamunurRahmanMoon/AI_Chat_Log_Summarizer_by_Moon)

---

## 📃 License

This project is licensed under the MIT License.

---
