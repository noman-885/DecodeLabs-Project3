"""
Project 3 Capstone: Tech Stack Recommender
DecodeLabs Industrial Training Kit

Architecture: IPO Model + Content-Based Filtering
  Phase 1: Ingestion   -> load job roles (items) + user skills (profile)
  Phase 2: Vectorize   -> TF-IDF: turn text tags into weighted numeric vectors
  Phase 3: Score        -> Cosine Similarity between user vector and every role
  Phase 4: Rank/Filter  -> sort descending, return Top-N

No sklearn used deliberately, so the TF-IDF and cosine similarity math is
fully visible and explainable.
"""

import csv
import math
from collections import Counter

CSV_PATH = "raw_skills.csv"
TOP_N = 3

# PHASE 1: INGESTION
def load_job_roles(csv_path):
    """Read role -> [skills] from CSV. Each role is treated as one 'document'."""
    roles = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills = [s.strip().lower() for s in row["skills"].split(",")]
            roles[row["role"]] = skills
    return roles

# PHASE 2: VECTORIZE (TF-IDF)
def compute_idf(documents):
    """
    IDF(term) = log(total_documents / documents_containing_term)
    Rare, specific skills get a high IDF. Common skills (appear in every
    role, e.g. 'git') get a low IDF -> they matter less for similarity.
    """
    n_docs = len(documents)
    doc_freq = Counter()
    for doc in documents:
        for term in set(doc):  # count each term once per document
            doc_freq[term] += 1

    idf = {}
    for term, freq in doc_freq.items():
        idf[term] = math.log(n_docs / freq)
    return idf


def compute_tf(doc):
    """TF(term) = count of term in doc / total terms in doc."""
    total = len(doc)
    counts = Counter(doc)
    return {term: count / total for term, count in counts.items()}


def vectorize(doc, idf, vocabulary):
    """Build a TF-IDF vector for one document, aligned to the full vocabulary."""
    tf = compute_tf(doc)
    return [tf.get(term, 0.0) * idf.get(term, 0.0) for term in vocabulary]

# PHASE 3: SCORE (Cosine Similarity)
def cosine_similarity(vec_a, vec_b):
    """
    cos(theta) = (A . B) / (||A|| * ||B||)
    Measures the ANGLE between vectors, ignoring magnitude -- so a user
    with 3 skills can still be compared fairly against a role with 6 skills.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # cold-start guard: no overlap possible
    return dot_product / (magnitude_a * magnitude_b)

# PHASE 4: RANK + FILTER (Top-N)
def recommend(user_skills, roles, top_n=TOP_N):
    user_doc = [s.strip().lower() for s in user_skills]

    all_docs = list(roles.values()) + [user_doc]
    vocabulary = sorted({term for doc in all_docs for term in doc})

    idf = compute_idf(all_docs)
    user_vector = vectorize(user_doc, idf, vocabulary)

    scores = []
    for role, skills in roles.items():
        role_vector = vectorize(skills, idf, vocabulary)
        score = cosine_similarity(user_vector, role_vector)
        scores.append((role, score))

    scores.sort(key=lambda pair: pair[1], reverse=True)  # Step 3: Sorting
    return scores[:top_n]  # Step 4: Filtering

def main():
    roles = load_job_roles(CSV_PATH)

    print("Tech Stack Recommender")
    print("Enter at least 3 skills, comma-separated (e.g. Python, Cloud, Automation)")
    raw = input("Your skills: ")
    user_skills = [s.strip() for s in raw.split(",") if s.strip()]

    if len(user_skills) < 3:
        print("Please enter at least 3 skills for an accurate match.")
        return

    results = recommend(user_skills, roles)

    print("\nTop recommended career paths:")
    for rank, (role, score) in enumerate(results, start=1):
        print(f"{rank}. {role}  (match: {score:.2%})")


if __name__ == "__main__":
    main()