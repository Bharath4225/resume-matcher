import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from file_parser import extract_text_from_file
from nlp_engine import analyze_resume_vs_jd

app = Flask(__name__, static_folder="static")
CORS(app)

# Max upload limit: 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Preset Sample Data for 1-Click Testing
PRESET_SAMPLES = {
    "fullstack": {
        "title": "Full-Stack Developer",
        "resume": """
Senior Full-Stack Software Engineer with 4+ years of experience building scalable web applications.
Proficient in Python, JavaScript, TypeScript, React.js, Node.js, Express, and PostgreSQL.
Hands-on experience with Docker, AWS (S3, EC2, Lambda), Git, REST APIs, and GraphQL.
Implemented CI/CD pipelines using GitHub Actions. Strong knowledge of Data Structures, Algorithms,
Object-Oriented Programming (OOP), and Agile/Scrum methodologies.
Successfully led a team to redesign microservices architecture, reducing latency by 40%.
""",
        "jd": """
We are seeking a Full-Stack Engineer to design, develop, and maintain high-performance web systems.
Key Requirements:
- 3+ years of experience with React, Node.js, TypeScript, and Python.
- Deep understanding of SQL databases (PostgreSQL/MySQL) and Redis caching.
- Familiarity with Cloud Platforms (AWS or GCP), Docker, and Kubernetes (k8s).
- Experience creating RESTful APIs and microservices.
- Excellent problem-solving, communication skills, and unit testing experience with Jest/Pytest.
- Exposure to MongoDB, Kafka, or GraphQL is a strong plus.
"""
    },
    "datascientist": {
        "title": "Data Scientist / ML Engineer",
        "resume": """
Data Scientist with a background in machine learning, natural language processing (NLP), and statistical modeling.
Strong proficiency in Python, SQL, Pandas, NumPy, Scikit-learn, PyTorch, and TensorFlow.
Built end-to-end NLP pipelines using TF-IDF, spaCy, and transformer embeddings for sentiment analysis.
Experienced with MySQL, MongoDB, Docker, Git, and Jupyter Notebooks.
Demonstrated ability to deploy ML models via Flask REST APIs on AWS EC2.
""",
        "jd": """
Looking for a Data Scientist / Machine Learning Engineer to join our AI team.
Responsibilities:
- Build predictive machine learning and NLP models using Python, Scikit-Learn, PyTorch, or TensorFlow.
- Perform feature engineering, data extraction, and TF-IDF / Cosine Similarity analysis.
- Collaborate with software engineers to deploy ML pipelines as microservices (Flask/FastAPI, Docker).
- Required Skills: Python, SQL, Pandas, Scikit-Learn, PyTorch, NLP, Docker, AWS, Spark, Vector Databases.
"""
    }
}

@app.route("/")
def serve_index():
    """Serves the main single-page web app."""
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    """Serves static assets (CSS, JS, images)."""
    return send_from_directory("static", path)

@app.route("/api/health", methods=["GET"])
def health_check():
    """API Health Check."""
    return jsonify({"status": "healthy", "service": "Resume Matcher NLP Engine"})

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns sample datasets for instant 1-click UI demos."""
    return jsonify(PRESET_SAMPLES)

@app.route("/api/match", methods=["POST"])
def match_resume():
    """
    Main Endpoint: Accepts text inputs or file uploads for Resume and Job Description.
    Processes NLP extraction, TF-IDF vectorization, Cosine Similarity scoring, and skill gap identification.
    """
    try:
        resume_text = ""
        jd_text = ""

        # Case 1: Form-data with file uploads or text fields
        if request.files or request.form:
            if 'resume_file' in request.files and request.files['resume_file'].filename:
                resume_text = extract_text_from_file(request.files['resume_file'])
            elif 'resume_text' in request.form:
                resume_text = request.form['resume_text']

            if 'jd_file' in request.files and request.files['jd_file'].filename:
                jd_text = extract_text_from_file(request.files['jd_file'])
            elif 'jd_text' in request.form:
                jd_text = request.form['jd_text']

        # Case 2: JSON Payload
        elif request.is_json:
            data = request.get_json() or {}
            resume_text = data.get("resume_text", "")
            jd_text = data.get("jd_text", "")

        # Validation
        if not resume_text or not resume_text.strip():
            return jsonify({"error": "Resume content is required. Please paste text or upload a file (.txt, .pdf, .docx)."}), 400

        if not jd_text or not jd_text.strip():
            return jsonify({"error": "Job Description content is required. Please paste text or upload a file (.txt, .pdf, .docx)."}), 400

        # Perform NLP Analysis
        results = analyze_resume_vs_jd(resume_text, jd_text)
        return jsonify(results)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during processing: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Resume Matcher Flask Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
