"""
NLP Engine for Resume-Job Description Matcher
==============================================
This module provides an interpretable NLP analysis pipeline using:
1. Text Cleaning & Normalization
2. TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization
   - Primary: scikit-learn TfidfVectorizer
   - Fallback: Pure Python Sublinear TF-IDF Engine
3. Cosine Similarity Vector Scoring
4. Categorized Tech Taxonomy & SpaCy/Regex Skill Extraction
5. Skill Gap Identification & Actionable ATS Recommendations
6. Structured Interview Prep Explanations (Math formulas & trade-off rationale)
"""

import re
import math
from collections import Counter

# Try importing scikit-learn, graceful fallback to pure python vectorizer if not installed
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try loading spaCy, provide graceful fallback if model is not installed
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except ImportError:
    nlp = None

# Comprehensive Taxonomy of Tech & Professional Skills
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang", "rust",
        "ruby", "php", "swift", "kotlin", "r", "scala", "matlab", "perl", "bash", "shell", "sql", "html", "css"
    ],
    "Frameworks & Libraries": [
        "react", "react.js", "next.js", "vue", "vue.js", "angular", "node.js", "express", "express.js",
        "django", "flask", "fastapi", "spring", "spring boot", "asp.net", "rails", "laravel",
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "spacy", "nltk", "opencv",
        "pandas", "numpy", "matplotlib", "seaborn", "tailwind", "tailwindcss", "bootstrap"
    ],
    "Databases & Data Stores": [
        "postgresql", "postgres", "mysql", "mongodb", "sqlite", "redis", "elasticsearch",
        "dynamodb", "cassandra", "oracle", "sql server", "neo4j", "snowflake", "bigquery"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "github actions", "gitlab ci", "ci/cd", "nginx",
        "linux", "unix", "bash", "cloudformation", "prometheus", "grafana"
    ],
    "AI, ML & Data Engineering": [
        "machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
        "data mining", "data pipelines", "spark", "pyspark", "hadoop", "kafka", "airflow",
        "vector databases", "embeddings", "tf-idf", "llm", "rag", "transformers", "feature engineering"
    ],
    "Tools, Testing & Architecture": [
        "git", "github", "gitlab", "jira", "postman", "rest api", "graphql", "microservices",
        "system design", "unit testing", "pytest", "jest", "cypress", "agile", "scrum", "kanban",
        "oop", "object-oriented programming", "design patterns"
    ],
    "Soft Skills & Management": [
        "problem solving", "leadership", "communication", "team collaboration", "critical thinking",
        "time management", "project management", "stakeholder management", "analytical skills"
    ]
}

# Standard English stop words fallback
STOP_WORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
])

def clean_text(text: str) -> str:
    """
    Step 1: Text Preprocessing & Cleaning
    -------------------------------------
    Normalizes text by:
    - Lowercasing to ensure case-insensitivity
    - Preserving technical punctuation (e.g., C++, Node.js, Next.js)
    - Removing non-alphanumeric noise
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'c\+\+', 'cpp_token', text)
    text = re.sub(r'c\#', 'csharp_token', text)
    text = re.sub(r'node\.js', 'nodejs_token', text)
    text = re.sub(r'react\.js', 'reactjs_token', text)
    text = re.sub(r'next\.js', 'nextjs_token', text)
    text = re.sub(r'vue\.js', 'vuejs_token', text)
    text = re.sub(r'express\.js', 'expressjs_token', text)
    text = re.sub(r'\.net', 'dotnet_token', text)

    text = re.sub(r'[^a-z0-9_\s]', ' ', text)
    
    text = text.replace('cpp_token', 'c++')
    text = text.replace('csharp_token', 'c#')
    text = text.replace('nodejs_token', 'node.js')
    text = text.replace('reactjs_token', 'react.js')
    text = text.replace('nextjs_token', 'next.js')
    text = text.replace('vuejs_token', 'vue.js')
    text = text.replace('expressjs_token', 'express.js')
    text = text.replace('dotnet_token', '.net')

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text: str) -> dict:
    """
    Step 2: Keyword & Skill Extraction
    -----------------------------------
    Scans the text using taxonomy pattern matching and spaCy noun chunk analysis.
    """
    clean = clean_text(text)
    lower_raw = text.lower()
    
    found_skills = set()
    categorized = {cat: [] for cat in SKILL_TAXONOMY}
    skill_counts = Counter()

    for cat, skills_list in SKILL_TAXONOMY.items():
        for skill in skills_list:
            pattern = r'(?<![a-zA-Z0-9#+])' + re.escape(skill) + r'(?![a-zA-Z0-9#+])'
            matches = len(re.findall(pattern, lower_raw))
            if matches > 0:
                found_skills.add(skill)
                categorized[cat].append(skill)
                skill_counts[skill] = matches

    if nlp is not None and len(clean) > 0:
        try:
            doc = nlp(clean[:50000])
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip().lower()
                if len(phrase) > 3 and phrase not in STOP_WORDS:
                    for cat, skills_list in SKILL_TAXONOMY.items():
                        if phrase in skills_list and phrase not in found_skills:
                            found_skills.add(phrase)
                            categorized[cat].append(phrase)
                            skill_counts[phrase] = 1
        except Exception:
            pass

    return {
        "all_skills": sorted(list(found_skills)),
        "categorized_skills": {k: v for k, v in categorized.items() if v},
        "skill_counts": dict(skill_counts)
    }

def pure_python_tfidf_similarity(doc1: str, doc2: str):
    """
    Pure Python Sublinear TF-IDF + Cosine Similarity Engine
    Used as an ultra-reliable, zero-dependency fallback.
    """
    def tokenize(t):
        words = [w for w in t.split() if w not in STOP_WORDS and len(w) > 1]
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        return words + bigrams

    tokens1 = tokenize(doc1)
    tokens2 = tokenize(doc2)

    tf1 = Counter(tokens1)
    tf2 = Counter(tokens2)
    vocab = set(tf1.keys()).union(set(tf2.keys()))

    if not vocab:
        return 0.0, [], [], []

    # Calculate IDF: log((N + 1) / (df + 1)) + 1
    N = 2
    idf = {}
    for term in vocab:
        df = (1 if term in tf1 else 0) + (1 if term in tf2 else 0)
        idf[term] = math.log((N + 1.0) / (df + 1.0)) + 1.0

    # Sublinear TF scaling: 1 + log(tf) if tf > 0 else 0
    vec1 = {term: (1 + math.log(tf1[term])) * idf[term] for term in tf1}
    vec2 = {term: (1 + math.log(tf2[term])) * idf[term] for term in tf2}

    # Dot product
    dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in vocab)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        sim = 0.0
    else:
        sim = (dot_product / (norm1 * norm2)) * 100.0

    # Overlapping contributions
    overlapping = []
    for t in vocab:
        w1 = vec1.get(t, 0)
        w2 = vec2.get(t, 0)
        if w1 > 0 and w2 > 0:
            overlapping.append({
                "term": t,
                "resume_weight": round(w1, 4),
                "jd_weight": round(w2, 4),
                "contribution": round(w1 * w2, 4)
            })

    overlapping.sort(key=lambda x: x["contribution"], reverse=True)
    
    top_r = [{"term": t, "weight": round(w, 4)} for t, w in sorted(vec1.items(), key=lambda x: x[1], reverse=True)[:15]]
    top_j = [{"term": t, "weight": round(w, 4)} for t, w in sorted(vec2.items(), key=lambda x: x[1], reverse=True)[:15]]

    return round(sim, 2), top_r, top_j, overlapping[:15]

def calculate_tfidf_similarity(resume_clean: str, jd_clean: str):
    """
    Step 3: Vectorization & Similarity Math
    ----------------------------------------
    Calculates TF-IDF matrices and Cosine Similarity score.
    Attempts scikit-learn first; falls back to pure Python implementation if sklearn is unavailable.
    """
    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0, [], [], []

    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                sublinear_tf=True
            )
            corpus = [resume_clean, jd_clean]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            sim_score = float(similarity_matrix[0][0]) * 100.0

            feature_names = vectorizer.get_feature_names_out()
            resume_vector = tfidf_matrix[0].toarray()[0]
            jd_vector = tfidf_matrix[1].toarray()[0]

            top_resume_terms = []
            top_jd_terms = []
            overlapping_terms = []

            for idx, name in enumerate(feature_names):
                r_score = float(resume_vector[idx])
                j_score = float(jd_vector[idx])
                if r_score > 0:
                    top_resume_terms.append({"term": name, "weight": round(r_score, 4)})
                if j_score > 0:
                    top_jd_terms.append({"term": name, "weight": round(j_score, 4)})
                if r_score > 0 and j_score > 0:
                    overlapping_terms.append({
                        "term": name,
                        "resume_weight": round(r_score, 4),
                        "jd_weight": round(j_score, 4),
                        "contribution": round(r_score * j_score, 4)
                    })

            top_resume_terms.sort(key=lambda x: x["weight"], reverse=True)
            top_jd_terms.sort(key=lambda x: x["weight"], reverse=True)
            overlapping_terms.sort(key=lambda x: x["contribution"], reverse=True)

            return round(sim_score, 2), top_resume_terms[:15], top_jd_terms[:15], overlapping_terms[:15]
        except Exception:
            pass

    # Pure Python Fallback
    return pure_python_tfidf_similarity(resume_clean, jd_clean)

def generate_ats_recommendations(missing_skills: list, match_score: float) -> list:
    """Generates practical actionable ATS tailoring tips based on gap analysis."""
    recs = []
    if match_score >= 85:
        recs.append("🎉 Excellent Match! Your resume covers almost all core requirements. Ensure your achievements include quantifiable metrics (e.g. 'Improved speed by 35%').")
    elif match_score >= 70:
        recs.append("👍 Strong Match! You have a solid foundation for this role. Adding a few targeted missing keywords will push your ATS score into the top tier.")
    elif match_score >= 50:
        recs.append("⚠️ Moderate Match. Several key skills requested in the JD are currently missing from your resume.")
    else:
        recs.append("❗ Low Similarity Match. Consider tailoring your resume significantly or adding relevant project experience matching this job profile.")

    if missing_skills:
        top_missing = missing_skills[:5]
        recs.append(f"💡 Key Missing Terms: We recommend highlighting experience with {', '.join(top_missing)} in your Skills or Project sections.")
        recs.append("📌 Action Tip: Use action verbs paired with missing technical terms (e.g., 'Utilized PyTorch to develop NLP pipelines...').")

    return recs

def analyze_resume_vs_jd(resume_text: str, jd_text: str) -> dict:
    """
    Main Orchestrator Function
    """
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    resume_skills_info = extract_skills(resume_text)
    jd_skills_info = extract_skills(jd_text)

    resume_skill_set = set(resume_skills_info["all_skills"])
    jd_skill_set = set(jd_skills_info["all_skills"])

    matched_skills = sorted(list(resume_skill_set.intersection(jd_skill_set)))
    missing_skills = sorted(list(jd_skill_set.difference(resume_skill_set)))
    extra_skills = sorted(list(resume_skill_set.difference(jd_skill_set)))

    skill_match_percentage = round((len(matched_skills) / len(jd_skill_set) * 100), 2) if jd_skill_set else 0.0

    tfidf_score, top_resume_terms, top_jd_terms, term_contributions = calculate_tfidf_similarity(resume_clean, jd_clean)

    if jd_skill_set:
        final_score = round(0.6 * tfidf_score + 0.4 * skill_match_percentage, 1)
    else:
        final_score = tfidf_score

    if final_score >= 85:
        match_tier = "Excellent Match"
        tier_color = "#10b981"
    elif final_score >= 70:
        match_tier = "Strong Match"
        tier_color = "#06b6d4"
    elif final_score >= 50:
        match_tier = "Moderate Match"
        tier_color = "#f59e0b"
    else:
        match_tier = "Low Match"
        tier_color = "#ef4444"

    categorized_matched = {}
    categorized_missing = {}
    
    for cat, skills in SKILL_TAXONOMY.items():
        m_in_cat = [s for s in matched_skills if s in skills]
        miss_in_cat = [s for s in missing_skills if s in skills]
        if m_in_cat:
            categorized_matched[cat] = m_in_cat
        if miss_in_cat:
            categorized_missing[cat] = miss_in_cat

    recommendations = generate_ats_recommendations(missing_skills, final_score)

    interview_explanation = {
        "summary": "This NLP pipeline computes similarity using TF-IDF vectorization and Cosine Similarity, complemented by a rule-based & spaCy entity extraction engine.",
        "steps": [
            {
                "step": "1. Text Preprocessing & Tokenization",
                "details": "Converts raw text into lowercased tokens, preserves technical symbols (e.g. C++, Node.js), strips stop-words, and eliminates formatting noise."
            },
            {
                "step": "2. TF-IDF Vectorization",
                "formula": "TF(t, d) = count(t, d) / total_words(d)  |  IDF(t) = log(N / df(t)) + 1",
                "details": "Converts resume and job description into high-dimensional numerical feature vectors. Sublinear scaling (1 + log(tf)) prevents high frequency terms from distorting scores."
            },
            {
                "step": "3. Cosine Similarity Computation",
                "formula": "Cosine Similarity(A, B) = (A · B) / (||A|| * ||B||)",
                "details": "Measures the cosine of the angle between the multi-dimensional TF-IDF vectors. Independent of document length differences."
            },
            {
                "step": "4. Skill Gap Extraction",
                "details": "Scans texts against a standardized technical taxonomy. Identifies exact matched keywords and missing mandatory skills present in JD but absent from Resume."
            }
        ],
        "tradeoff_qa": [
            {
                "question": "Why use TF-IDF + Cosine Similarity instead of BERT or Large Language Models (LLMs)?",
                "answer": "TF-IDF provides 100% deterministic interpretability: we can show the exact mathematical weight of every single term. LLMs can hallucinate or introduce non-deterministic embeddings, take seconds to infer, and require GPU resources. TF-IDF is ultra-fast, lightweight, and perfect for keyword-driven Applicant Tracking Systems (ATS)."
            },
            {
                "question": "How does Cosine Similarity handle length disparity between a 1-page Resume and a 3-page JD?",
                "answer": "Cosine similarity normalizes vectors by their Euclidean norm (L2 norm: ||A||). Because it evaluates the geometric direction/angle of the vectors rather than magnitude, length differences do not penalize the score."
            }
        ]
    }

    return {
        "scores": {
            "overall_match_percentage": final_score,
            "tfidf_cosine_similarity": tfidf_score,
            "skill_match_coverage": skill_match_percentage,
            "match_tier": match_tier,
            "tier_color": tier_color
        },
        "skills_analysis": {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "total_jd_skills_count": len(jd_skill_set),
            "total_matched_skills_count": len(matched_skills),
            "categorized_matched": categorized_matched,
            "categorized_missing": categorized_missing
        },
        "tfidf_details": {
            "top_resume_terms": top_resume_terms,
            "top_jd_terms": top_jd_terms,
            "overlapping_contributors": term_contributions
        },
        "ats_recommendations": recommendations,
        "interview_guide": interview_explanation
    }
