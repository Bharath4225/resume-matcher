"""
NLP Engine for Resume-Job Description Matcher (High-Precision Edition)
=======================================================================
This module provides a highly accurate, deterministic NLP analysis engine:
1. Advanced Text Preprocessing & Tech Token Protection
2. Enhanced Tech Taxonomy & Multi-Gram Phrase Skill Extractor
3. Corporate Stop-Word Filtering & TF-IDF Vectorization
4. Length-Normalized Cosine Similarity Computation
5. Categorized Skill Gap Breakdown & ATS Recommendation Generator
6. Math & Architectural Explanations for Interview Prep
"""

import re
import math
from collections import Counter

# Try importing scikit-learn
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try loading spaCy
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
        "ruby", "php", "swift", "kotlin", "r", "scala", "matlab", "perl", "bash", "shell",
        "sql", "html", "html5", "css", "css3", "dart", "elixir", "haskell", "assembly"
    ],
    "Frameworks & Libraries": [
        "react", "react.js", "reactjs", "next.js", "nextjs", "vue", "vue.js", "vuejs", "angular", "angularjs",
        "node.js", "nodejs", "express", "express.js", "expressjs", "django", "flask", "fastapi", "spring", "spring boot",
        "asp.net", "rails", "ruby on rails", "laravel", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "spacy", "nltk", "opencv", "pandas", "numpy", "matplotlib", "seaborn", "tailwind",
        "tailwindcss", "bootstrap", "redux", "graphql", "apollo", "flutter", "react native", "electron"
    ],
    "Databases & Data Stores": [
        "postgresql", "postgres", "mysql", "mongodb", "sqlite", "redis", "elasticsearch",
        "dynamodb", "cassandra", "oracle", "sql server", "mssql", "neo4j", "snowflake", "bigquery",
        "mariadb", "firebase", "firestore", "cockroachdb", "clickhouse", "pinecone", "chromadb"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "github actions", "gitlab ci", "ci/cd", "nginx", "apache",
        "linux", "unix", "bash", "cloudformation", "prometheus", "grafana", "helm", "istio", "serverless",
        "ec2", "s3", "lambda", "ecs", "eks", "cloudwatch", "route53"
    ],
    "AI, ML & Data Engineering": [
        "machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
        "data mining", "data pipelines", "spark", "pyspark", "hadoop", "kafka", "airflow", "dbt",
        "vector databases", "embeddings", "tf-idf", "llm", "rag", "transformers", "feature engineering",
        "bert", "gpt", "langchain", "llamaIndex", "model deployment", "hyperparameter tuning"
    ],
    "Tools, Testing & Architecture": [
        "git", "github", "gitlab", "jira", "postman", "rest api", "restful api", "microservices",
        "system design", "unit testing", "pytest", "jest", "cypress", "selenium", "mocha", "chai",
        "agile", "scrum", "kanban", "oop", "object-oriented programming", "design patterns",
        "solid principles", "tdd", "test-driven development", "web sockets", "grpc"
    ],
    "Soft Skills & Methodologies": [
        "problem solving", "leadership", "communication", "team collaboration", "critical thinking",
        "time management", "project management", "stakeholder management", "analytical skills",
        "cross-functional collaboration", "code review", "mentorship", "decision making"
    ]
}

# Corporate & JD Filler Words to Exclude from TF-IDF Feature Matrices
FILLER_WORDS = set([
    "candidate", "candidates", "requirement", "requirements", "responsibility", "responsibilities",
    "qualification", "qualifications", "role", "roles", "position", "positions", "job", "jobs",
    "description", "looking", "seeking", "ideal", "opportunity", "environment", "team", "teams",
    "work", "working", "experience", "experiences", "years", "year", "ability", "proven", "track",
    "record", "strong", "understanding", "knowledge", "expertise", "proficient", "proficiency",
    "skill", "skills", "duty", "duties", "company", "organization", "client", "business",
    "successful", "must", "have", "plus", "preferred", "minimum", "maximum", "etc"
])

STANDARD_STOP_WORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent",
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

ALL_STOP_WORDS = set([re.sub(r'[^a-z0-9]', '', w) for w in STANDARD_STOP_WORDS.union(FILLER_WORDS) if w.strip()])

def clean_text(text: str) -> str:
    """
    Step 1: Text Preprocessing & Token Protection
    ---------------------------------------------
    Normalizes text while preserving complex technical symbols and multi-word tech terms.
    """
    if not text:
        return ""
    
    text = text.lower()

    # Protect specific technical tokens before stripping punctuation
    token_map = {
        r'c\+\+': 'cpp_token',
        r'c\#': 'csharp_token',
        r'node\.js': 'nodejs_token',
        r'react\.js': 'reactjs_token',
        r'next\.js': 'nextjs_token',
        r'vue\.js': 'vuejs_token',
        r'express\.js': 'expressjs_token',
        r'\.net': 'dotnet_token',
        r'ci/cd': 'cicd_token',
        r'restful api': 'restfulapi_token',
        r'rest api': 'restapi_token'
    }

    for pattern, replacement in token_map.items():
        text = re.sub(pattern, replacement, text)

    # Strip special punctuation except alphanumeric and underscore
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)

    # Restore protected tokens
    reverse_map = {
        'cpp_token': 'c++',
        'csharp_token': 'c#',
        'nodejs_token': 'node.js',
        'reactjs_token': 'react.js',
        'nextjs_token': 'next.js',
        'vuejs_token': 'vue.js',
        'expressjs_token': 'express.js',
        'dotnet_token': '.net',
        'cicd_token': 'ci/cd',
        'restfulapi_token': 'restful api',
        'restapi_token': 'rest api'
    }

    for token, word in reverse_map.items():
        text = text.replace(token, word)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text: str) -> dict:
    """
    Step 2: Skill Extraction Engine
    -------------------------------
    Scans text against taxonomy with regex boundary matching, alias resolution,
    and spaCy noun chunk identification.
    """
    clean = clean_text(text)
    lower_raw = text.lower()
    
    found_skills = set()
    categorized = {cat: [] for cat in SKILL_TAXONOMY}
    skill_counts = Counter()

    # Alias mapping to canonical names (e.g., nodejs -> node.js, reactjs -> react.js)
    alias_map = {
        "nodejs": "node.js",
        "reactjs": "react.js",
        "vuejs": "vue.js",
        "nextjs": "next.js",
        "expressjs": "express.js",
        "k8s": "kubernetes",
        "postgres": "postgresql",
        "sklearn": "scikit-learn",
        "tailwindcss": "tailwind"
    }

    for cat, skills_list in SKILL_TAXONOMY.items():
        for skill in skills_list:
            pattern = r'(?<![a-zA-Z0-9#+])' + re.escape(skill) + r'(?![a-zA-Z0-9#+])'
            matches = len(re.findall(pattern, lower_raw))
            if matches > 0:
                canonical_skill = alias_map.get(skill, skill)
                found_skills.add(canonical_skill)
                if canonical_skill not in categorized[cat]:
                    categorized[cat].append(canonical_skill)
                skill_counts[canonical_skill] += matches

    # SpaCy Noun Chunk Analysis
    if nlp is not None and len(clean) > 0:
        try:
            doc = nlp(clean[:50000])
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip().lower()
                if len(phrase) > 2 and phrase not in ALL_STOP_WORDS:
                    for cat, skills_list in SKILL_TAXONOMY.items():
                        if phrase in skills_list:
                            canonical_skill = alias_map.get(phrase, phrase)
                            found_skills.add(canonical_skill)
                            if canonical_skill not in categorized[cat]:
                                categorized[cat].append(canonical_skill)
                            skill_counts[canonical_skill] += 1
        except Exception:
            pass

    return {
        "all_skills": sorted(list(found_skills)),
        "categorized_skills": {k: v for k, v in categorized.items() if v},
        "skill_counts": dict(skill_counts)
    }

def pure_python_tfidf_similarity(doc1: str, doc2: str):
    """
    Pure Python Sublinear TF-IDF Engine Fallback
    """
    def tokenize(t):
        words = [w for w in t.split() if w not in ALL_STOP_WORDS and len(w) > 1]
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        return words + bigrams

    tokens1 = tokenize(doc1)
    tokens2 = tokenize(doc2)

    tf1 = Counter(tokens1)
    tf2 = Counter(tokens2)
    vocab = set(tf1.keys()).union(set(tf2.keys()))

    if not vocab:
        return 0.0, [], [], []

    N = 2
    idf = {}
    for term in vocab:
        df = (1 if term in tf1 else 0) + (1 if term in tf2 else 0)
        idf[term] = math.log((N + 1.0) / (df + 1.0)) + 1.0

    vec1 = {term: (1 + math.log(tf1[term])) * idf[term] for term in tf1}
    vec2 = {term: (1 + math.log(tf2[term])) * idf[term] for term in tf2}

    dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in vocab)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        sim = 0.0
    else:
        sim = (dot_product / (norm1 * norm2)) * 100.0

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
    Step 3: High-Precision TF-IDF Vectorization & Cosine Distance
    --------------------------------------------------------------
    """
    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0, [], [], []

    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=list(ALL_STOP_WORDS),
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

    return pure_python_tfidf_similarity(resume_clean, jd_clean)

def generate_ats_recommendations(missing_skills: list, match_score: float) -> list:
    """Generates precise, actionable ATS tailoring advice based on gap analysis."""
    recs = []
    if match_score >= 85:
        recs.append("🎉 Excellent Match! Your resume covers almost all core requirements. Ensure your achievements include quantifiable metrics (e.g. 'Improved API response speed by 35%').")
    elif match_score >= 70:
        recs.append("👍 Strong Match! You have a solid technical foundation for this role. Incorporating the missing keywords below into your Experience bullet points will push your ATS score into the top tier.")
    elif match_score >= 50:
        recs.append("⚠️ Moderate Match. Several mandatory technical skills requested in the Job Description are currently missing from your resume.")
    else:
        recs.append("❗ Low Similarity Match. Consider tailoring your resume significantly or emphasizing relevant project experience matching this job profile.")

    if missing_skills:
        top_missing = missing_skills[:5]
        recs.append(f"💡 Key Missing Keywords to Add: We recommend including experience with {', '.join(top_missing)} in your Skills or Professional Experience sections.")
        recs.append("📌 Actionable Example: Use strong action verbs combined with missing terms (e.g., 'Engineered microservices using Docker and PostgreSQL to handle 50k daily active users').")

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
    
    # Sort missing skills by frequency in Job Description for maximum relevance
    jd_counts = jd_skills_info["skill_counts"]
    missing_skills_raw = list(jd_skill_set.difference(resume_skill_set))
    missing_skills = sorted(missing_skills_raw, key=lambda s: jd_counts.get(s, 1), reverse=True)
    
    extra_skills = sorted(list(resume_skill_set.difference(jd_skill_set)))

    skill_match_percentage = round((len(matched_skills) / len(jd_skill_set) * 100), 2) if jd_skill_set else 0.0

    tfidf_score, top_resume_terms, top_jd_terms, term_contributions = calculate_tfidf_similarity(resume_clean, jd_clean)

    # Balanced Scoring Algorithm: 55% TF-IDF Semantic Overlap + 45% Skill Taxonomy Coverage
    if jd_skill_set:
        final_score = round(0.55 * tfidf_score + 0.45 * skill_match_percentage, 1)
    else:
        final_score = tfidf_score

    if final_score >= 85:
        match_tier = "Excellent Match"
        tier_color = "#10b981" # Emerald
    elif final_score >= 70:
        match_tier = "Strong Match"
        tier_color = "#06b6d4" # Cyan
    elif final_score >= 50:
        match_tier = "Moderate Match"
        tier_color = "#f59e0b" # Amber
    else:
        match_tier = "Low Match"
        tier_color = "#ef4444" # Red

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
                "step": "1. Text Preprocessing & Token Protection",
                "details": "Converts raw text into lowercased tokens, preserves technical symbols (e.g. C++, Node.js, REST API), strips corporate stop-words, and eliminates formatting noise."
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
