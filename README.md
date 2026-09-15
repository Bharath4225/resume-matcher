# 📄 Resume Matcher - Intelligent NLP Skill Gap & Similarity Engine

A deterministic, highly interpretable **Resume–Job Description Matcher** web application. Built with **Python, Flask, Scikit-learn, spaCy**, and a modern dark glassmorphism frontend, it scores document similarity using **TF-IDF + Cosine Similarity**, extracts missing technical skills, provides ATS tailoring recommendations, and includes an interactive **Interview Prep Guide**.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## ✨ Features

- 🎯 **TF-IDF + Cosine Similarity Scoring**: Converts resumes and job descriptions into high-dimensional numerical term vectors and calculates geometric similarity normalized for document length.
- 🔍 **Categorized Skill Taxonomy Extraction**: Scans text against technical taxonomy dictionaries covering *Languages, Frameworks, Cloud & DevOps, Databases, AI/ML, Tools, and Soft Skills*.
- 💡 **Skill Gap Identification**: Highlights exact matched keywords and isolates missing critical skills present in the JD but missing from the resume.
- 📄 **Multi-Format Document Parsing**: Supports raw text pasting and drag-and-drop file uploads for `.pdf` (PyPDF2), `.docx` (python-docx), and `.txt` files.
- ⚡ **1-Click Demo Presets**: Includes pre-loaded benchmarks (*Full-Stack Developer*, *Data Scientist / ML Engineer*) for instant 1-click testing.
- 🎓 **Interactive Interview Prep Drawer**: Built-in modal guide detailing step-by-step mathematical formulas, vector matrix mechanics, and trade-off interview Q&A (*Why TF-IDF over LLMs/BERT*).
- 🎨 **Modern Dark Glassmorphism UI**: Glowing radial match score gauge, animated sub-metric progress bars, and copyable missing skill chips.

---

## 🧮 Mathematical Foundations & Formulas

### 1. Term Frequency - Inverse Document Frequency (TF-IDF)
$$TF(t, d) = \frac{\text{count}(t, d)}{\text{total words in } d}$$

$$IDF(t) = \log\left(\frac{N}{\text{df}(t)}\right) + 1$$

$$\text{TF-IDF}(t, d) = (1 + \log(\text{TF}(t, d))) \times \text{IDF}(t)$$

### 2. Cosine Similarity Formula
$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11, Flask, Flask-CORS
- **NLP & Machine Learning**: `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`), `spaCy` (`en_core_web_sm`), Pure Python Sublinear TF-IDF fallback engine
- **Document Parsing**: `PyPDF2`, `python-docx`
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism theme, CSS Grid/Flexbox), JavaScript (ES6+), MathJax (LaTeX rendering)

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.9+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Bharath4225/resume-matcher.git
cd resume-matcher
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Development Server
```bash
python app.py
```

Open your browser and navigate to **`http://localhost:5000`**.

---

## 📊 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page web app |
| `GET` | `/api/health` | Health check endpoint |
| `GET` | `/api/samples` | Returns 1-click sample resume/JD pairs |
| `POST` | `/api/match` | Primary analysis endpoint (Accepts JSON or Multipart file uploads) |

---

## 🎯 Why TF-IDF over Transformers for ATS Matchers?

1. **100% Deterministic & Interpretable**: Every matching score can be traced back to exact term weight contribution products.
2. **Zero Hallucination**: Prevents false semantic associations (e.g. treating 'React' and 'Angular' as identical because both are frontend frameworks).
3. **Sub-millisecond Performance**: Runs instantly on standard CPUs without requiring GPU infrastructure.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
