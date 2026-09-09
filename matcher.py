"""
Matching Module
Converts resume and job description text into TF-IDF vectors,
computes cosine similarity, and ranks job roles by match score.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_document


def load_job_descriptions(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def compute_match_scores(resume_text: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cosine similarity between the resume and every job description.
    Returns the jobs_df with an added 'match_score' column, sorted descending.
    """
    cleaned_resume = preprocess_document(resume_text)

    job_texts = (
        jobs_df["required_skills"].astype(str)
        + " "
        + jobs_df["description"].astype(str)
    ).apply(preprocess_document)

    corpus = [cleaned_resume] + job_texts.tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    resume_vector = tfidf_matrix[0:1]
    job_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(resume_vector, job_vectors).flatten()

    result_df = jobs_df.copy()
    result_df["match_score"] = (similarities * 100).round(2)
    result_df = result_df.sort_values(by="match_score", ascending=False).reset_index(drop=True)

    return result_df


def get_best_match(resume_text: str, jobs_df: pd.DataFrame) -> dict:
    ranked = compute_match_scores(resume_text, jobs_df)
    best = ranked.iloc[0]
    return {
        "job_role": best["job_role"],
        "match_score": float(best["match_score"]),
        "all_matches": ranked[["job_role", "match_score"]].to_dict(orient="records"),
    }


if __name__ == "__main__":
    jobs = load_job_descriptions("../data/job_descriptions.csv")
    sample_resume = """
    Experienced software engineer skilled in Python, SQL, Machine Learning,
    Pandas, NumPy, and Scikit-learn. Built REST APIs using Flask.
    """
    result = get_best_match(sample_resume, jobs)
    print(result)
