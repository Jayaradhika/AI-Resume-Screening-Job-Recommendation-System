# AI Resume Screening & Job Recommendation System

**Intermediate AIML Project — AIME Internship**
Python | NLP | Machine Learning | Scikit-learn

## 📌 Overview
Recruitment involves screening large volumes of resumes, which is slow, inconsistent, and inefficient when done manually. This system uses **NLP and Machine Learning** to automatically analyze resumes, extract skills and qualifications, compare them against job descriptions, and recommend the most suitable job roles with a match score.

## 🎯 Objectives
- Automatically analyze resumes
- Extract text, skills, and qualifications
- Identify relevant keywords
- Compare resumes with job descriptions
- Recommend suitable job roles with a match score

## 🔄 System Workflow
```
Resume + Job Description (Input)
   ↓
1. Resume Upload
2. Text Extraction
3. Text Preprocessing
4. Skill & Keyword Extraction
5. Feature Extraction
6. Compare with Job Description
7. Matching Score (TF-IDF + Cosine Similarity)
8. Job Recommendation
   ↓
Matching Score + Recommended Job Roles (Output)
```

## 🛠️ Technologies & Algorithms
| Technologies | Algorithms / Techniques |
|---|---|
| Python | TF-IDF |
| Pandas | Cosine Similarity |
| NumPy | Logistic Regression |
| NLTK / spaCy | Naive Bayes |
| Scikit-learn | |
| Flask (UI) | |

**Core Technique:** NLP + Text Similarity

## 📂 Project Structure
```
resume-screening-system/
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── data/
│   ├── job_descriptions.csv   # Job roles & required skills dataset
│   ├── resumes_dataset.csv    # Sample labeled resumes for training
│   └── skills_list.csv        # Skill vocabulary for extraction
├── src/
│   ├── extract_text.py        # PDF/DOCX text extraction
│   ├── preprocess.py           # Text cleaning & tokenization
│   ├── skill_extractor.py     # Skill & keyword extraction
│   ├── matcher.py              # TF-IDF + cosine similarity matching
│   └── evaluate.py             # Model training & evaluation metrics
├── templates/
│   ├── index.html              # Upload page
│   └── results.html            # Results page
├── static/
│   └── style.css               # UI styling
├── notebooks/
│   └── model_evaluation.ipynb  # Exploratory analysis & evaluation
└── README.md
```

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/resume-screening-system.git
cd resume-screening-system
pip install -r requirements.txt
```

## ▶️ Usage
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser, upload a resume (PDF/DOCX), and view the match score and recommended job roles.

## 📊 Results & Evaluation
| Role | Match Score |
|---|---|
| Python Developer | 92% |
| Data Analyst | 85% |
| ML Engineer | 78% |
| Web Developer | 64% |

**Best Recommendation:** Python Developer — 92% Match

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score (see `src/evaluate.py` and `notebooks/model_evaluation.ipynb`)

## 🚀 Future Scope
- AI-powered interview question generation
- Automated candidate ranking
- LinkedIn/job portal integration
- Skill gap analysis
- Personalized job recommendations
- Resume improvement suggestions
- AI chatbot for career guidance

## 📄 License
MIT License
