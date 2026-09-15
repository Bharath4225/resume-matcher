from nlp_engine import analyze_resume_vs_jd

resume = """
Senior Full-Stack Developer with expertise in Python, React, TypeScript, Node.js, and PostgreSQL.
Experience with AWS, Docker, Git, CI/CD pipelines, and REST API development.
Strong background in data structures, algorithms, and microservices architecture.
"""

jd = """
We are looking for a Full-Stack Engineer skilled in Python, React, TypeScript, and Docker.
Required skills: Python, React, TypeScript, Docker, Kubernetes, GraphQL, Redis, PostgreSQL.
Experience with microservices and AWS is a plus.
"""

result = analyze_resume_vs_jd(resume, jd)
print("=== NLP PIPELINE TEST RESULTS ===")
print("Overall Match Score:", result["scores"]["overall_match_percentage"], "%")
print("TF-IDF Cosine Sim:", result["scores"]["tfidf_cosine_similarity"], "%")
print("Skill Match Coverage:", result["scores"]["skill_match_coverage"], "%")
print("Match Tier:", result["scores"]["match_tier"])
print("Matched Skills:", result["skills_analysis"]["matched_skills"])
print("Missing Skills:", result["skills_analysis"]["missing_skills"])
print("Top Overlapping TF-IDF Contributor Terms:")
for term in result["tfidf_details"]["overlapping_contributors"]:
    print(f"  - {term['term']}: contribution +{term['contribution']}")
print("\n[OK] NLP Pipeline Verification SUCCESS!")
