document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const resumeText = document.getElementById("resume-text");
    const jdText = document.getElementById("jd-text");
    const resumeFileInput = document.getElementById("resume-file-input");
    const jdFileInput = document.getElementById("jd-file-input");
    const resumeFileName = document.getElementById("resume-file-name");
    const jdFileName = document.getElementById("jd-file-name");
    
    const analyzeBtn = document.getElementById("analyze-btn");
    const analyzeSpinner = document.getElementById("analyze-spinner");
    const clearAllBtn = document.getElementById("clear-all-btn");

    const resultsSection = document.getElementById("results-section");
    const overallScoreVal = document.getElementById("overall-score-val");
    const gaugeCircle = document.getElementById("gauge-circle");
    const matchTierBadge = document.getElementById("match-tier-badge");
    const scoreDescText = document.getElementById("score-desc-text");

    const valTfidf = document.getElementById("val-tfidf");
    const barTfidf = document.getElementById("bar-tfidf");
    const valSkill = document.getElementById("val-skill");
    const barSkill = document.getElementById("bar-skill");

    const missingCount = document.getElementById("missing-count");
    const matchedCount = document.getElementById("matched-count");
    const missingSkillsChips = document.getElementById("missing-skills-chips");
    const categorizedMatchedContainer = document.getElementById("categorized-matched-container");
    const atsRecommendationsList = document.getElementById("ats-recommendations-list");
    const tfidfTableBody = document.getElementById("tfidf-table-body");
    const copyMissingBtn = document.getElementById("copy-missing-btn");

    const openInterviewGuideBtn = document.getElementById("open-interview-guide");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const interviewModal = document.getElementById("interview-modal");

    let presetData = {};

    // 1. Fetch Preset Samples
    fetchPresets();

    async function fetchPresets() {
        try {
            const res = await fetch("/api/samples");
            if (res.ok) {
                presetData = await res.json();
            }
        } catch (err) {
            console.error("Failed to load sample presets:", err);
        }
    }

    // Preset Button Click Handlers
    document.querySelectorAll(".btn-preset").forEach(btn => {
        btn.addEventListener("click", () => {
            const key = btn.getAttribute("data-preset");
            if (presetData[key]) {
                // Switch both cards to text paste tab
                activateTab("resume-card", "resume-paste");
                activateTab("jd-card", "jd-paste");
                
                resumeText.value = presetData[key].resume.trim();
                jdText.value = presetData[key].jd.trim();
                
                // Clear file inputs
                resumeFileInput.value = "";
                jdFileInput.value = "";
                resumeFileName.textContent = "";
                jdFileName.textContent = "";

                // Auto-scroll to analyze button smoothly
                analyzeBtn.scrollIntoView({ behavior: "smooth", block: "center" });
            }
        });
    });

    clearAllBtn.addEventListener("click", () => {
        resumeText.value = "";
        jdText.value = "";
        resumeFileInput.value = "";
        jdFileInput.value = "";
        resumeFileName.textContent = "";
        jdFileName.textContent = "";
        resultsSection.hidden = true;
    });

    // 2. Tab Switching Logic
    document.querySelectorAll(".tab-btn").forEach(tab => {
        tab.addEventListener("click", (e) => {
            const card = e.target.closest(".input-card");
            const targetId = e.target.getAttribute("data-target");
            activateTab(card.id, targetId);
        });
    });

    function activateTab(cardId, targetId) {
        const card = document.getElementById(cardId);
        card.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        card.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        
        card.querySelector(`[data-target="${targetId}"]`).classList.add("active");
        document.getElementById(targetId).classList.add("active");
    }

    // 3. File Input Drag & Drop Handlers
    setupDropzone("resume-dropzone", resumeFileInput, resumeFileName);
    setupDropzone("jd-dropzone", jdFileInput, jdFileName);

    function setupDropzone(dropzoneId, fileInput, nameDisplay) {
        const zone = document.getElementById(dropzoneId);

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                nameDisplay.textContent = `📄 Selected: ${e.target.files[0].name}`;
            }
        });

        zone.addEventListener("dragover", (e) => {
            e.preventDefault();
            zone.classList.add("dragover");
        });

        zone.addEventListener("dragleave", () => {
            zone.classList.remove("dragover");
        });

        zone.addEventListener("drop", (e) => {
            e.preventDefault();
            zone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                nameDisplay.textContent = `📄 Selected: ${e.dataTransfer.files[0].name}`;
            }
        });
    }

    // 4. Main Analysis Trigger
    analyzeBtn.addEventListener("click", async () => {
        const formData = new FormData();

        // Check Resume source
        const resumeActiveTab = document.querySelector("#resume-card .tab-btn.active").getAttribute("data-target");
        if (resumeActiveTab === "resume-paste") {
            if (!resumeText.value.trim()) {
                alert("Please paste resume text or upload a resume file.");
                return;
            }
            formData.append("resume_text", resumeText.value.trim());
        } else {
            if (!resumeFileInput.files[0]) {
                alert("Please select a resume file to upload.");
                return;
            }
            formData.append("resume_file", resumeFileInput.files[0]);
        }

        // Check Job Description source
        const jdActiveTab = document.querySelector("#jd-card .tab-btn.active").getAttribute("data-target");
        if (jdActiveTab === "jd-paste") {
            if (!jdText.value.trim()) {
                alert("Please paste job description text or upload a file.");
                return;
            }
            formData.append("jd_text", jdText.value.trim());
        } else {
            if (!jdFileInput.files[0]) {
                alert("Please select a job description file to upload.");
                return;
            }
            formData.append("jd_file", jdFileInput.files[0]);
        }

        // Start Loading State
        analyzeBtn.disabled = true;
        analyzeSpinner.hidden = false;

        try {
            const response = await fetch("/api/match", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                alert(`Analysis Error: ${data.error || "Unknown server error"}`);
                return;
            }

            // Render Analysis Results
            renderResults(data);
            resultsSection.hidden = false;
            resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

        } catch (error) {
            console.error("Match API Error:", error);
            alert("Failed to connect to Resume Matcher API backend server.");
        } finally {
            analyzeBtn.disabled = false;
            analyzeSpinner.hidden = true;
        }
    });

    // 5. Render Results Function
    function renderResults(data) {
        const scores = data.scores;
        const skills = data.skills_analysis;
        const tfidf = data.tfidf_details;

        // Overall score gauge
        const finalScore = scores.overall_match_percentage;
        overallScoreVal.textContent = `${finalScore}%`;
        
        // Gauge circumference is 314 (radius = 50, 2*PI*50 ≈ 314)
        const offset = 314 - (finalScore / 100) * 314;
        gaugeCircle.style.strokeDashoffset = offset;
        gaugeCircle.style.stroke = scores.tier_color;

        // Badge & Meta
        matchTierBadge.textContent = scores.match_tier;
        matchTierBadge.style.backgroundColor = `${scores.tier_color}25`;
        matchTierBadge.style.color = scores.tier_color;

        valTfidf.textContent = `${scores.tfidf_cosine_similarity}%`;
        barTfidf.style.width = `${scores.tfidf_cosine_similarity}%`;

        valSkill.textContent = `${scores.skill_match_coverage}%`;
        barSkill.style.width = `${scores.skill_match_coverage}%`;

        // Missing Skills Chips
        missingCount.textContent = skills.missing_skills.length;
        missingSkillsChips.innerHTML = "";
        
        if (skills.missing_skills.length === 0) {
            missingSkillsChips.innerHTML = `<span class="skill-chip chip-matched">🎉 No Missing Skills Detected! Excellent coverage.</span>`;
        } else {
            skills.missing_skills.forEach(skill => {
                const chip = document.createElement("span");
                chip.className = "skill-chip chip-missing";
                chip.innerHTML = `⚠️ ${skill}`;
                missingSkillsChips.appendChild(chip);
            });
        }

        // Matched Skills Categorized
        matchedCount.textContent = skills.matched_skills.length;
        categorizedMatchedContainer.innerHTML = "";

        if (Object.keys(skills.categorized_matched).length === 0) {
            categorizedMatchedContainer.innerHTML = `<span class="skill-chip chip-missing">No direct technical skills matched.</span>`;
        } else {
            for (const [category, catSkills] of Object.entries(skills.categorized_matched)) {
                const block = document.createElement("div");
                block.className = "category-block";
                
                const title = document.createElement("div");
                title.className = "category-title";
                title.textContent = category;
                block.appendChild(title);

                const flex = document.createElement("div");
                flex.className = "skills-flex";
                
                catSkills.forEach(skill => {
                    const chip = document.createElement("span");
                    chip.className = "skill-chip chip-matched";
                    chip.innerHTML = `✓ ${skill}`;
                    flex.appendChild(chip);
                });

                block.appendChild(flex);
                categorizedMatchedContainer.appendChild(block);
            }
        }

        // ATS Recommendations List
        atsRecommendationsList.innerHTML = "";
        data.ats_recommendations.forEach(rec => {
            const li = document.createElement("li");
            li.textContent = rec;
            atsRecommendationsList.appendChild(li);
        });

        // TF-IDF Table
        tfidfTableBody.innerHTML = "";
        if (!tfidf.overlapping_contributors || tfidf.overlapping_contributors.length === 0) {
            tfidfTableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; color: var(--text-muted);">No overlapping TF-IDF terms found between documents.</td></tr>`;
        } else {
            tfidf.overlapping_contributors.forEach(row => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${row.term}</td>
                    <td>${row.resume_weight}</td>
                    <td>${row.jd_weight}</td>
                    <td style="color: var(--emerald); font-weight: 700;">+${row.contribution}</td>
                `;
                tfidfTableBody.appendChild(tr);
            });
        }
    }

    // 6. Copy Missing Skills Button
    copyMissingBtn.addEventListener("click", () => {
        const chips = document.querySelectorAll(".chip-missing");
        const terms = Array.from(chips).map(c => c.textContent.replace("⚠️ ", "").trim()).filter(Boolean);
        
        if (terms.length === 0) return;

        navigator.clipboard.writeText(terms.join(", ")).then(() => {
            const originalText = copyMissingBtn.textContent;
            copyMissingBtn.textContent = "✓ Copied!";
            setTimeout(() => copyMissingBtn.textContent = originalText, 2000);
        });
    });

    // 7. Modal Handlers for Interview Guide
    openInterviewGuideBtn.addEventListener("click", () => {
        interviewModal.hidden = false;
        if (window.MathJax) {
            window.MathJax.typesetPromise();
        }
    });

    closeModalBtn.addEventListener("click", () => {
        interviewModal.hidden = true;
    });

    interviewModal.addEventListener("click", (e) => {
        if (e.target === interviewModal) {
            interviewModal.hidden = true;
        }
    });
});
