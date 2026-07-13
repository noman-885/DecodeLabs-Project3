# DecodeLabs-Project3
Rule-based AI chatbot and TF-IDF/cosine-similarity tech stack recommender — DecodeLabs AI Industrial Training Kit, built from scratch in Python.

Tech Stack Recommender (tech_stack_recommender.py)

A content-based recommendation engine that maps a user's skills to the closest-matching job roles using TF-IDF weighting and cosine similarity — implemented manually (no scikit-learn) so every step of the math is transparent.

Pipeline: Ingestion → Vectorize (TF-IDF) → Score (Cosine Similarity) → Rank & Filter (Top-3)

bashpython3 tech_stack_recommender.py

Requires raw_skills.csv in the same directory (role → comma-separated skills).

Requirements


Python 3.8+
No external dependencies — both scripts use only the standard library (csv, math, random, collections)


Concepts Covered

ConceptWhereDeterministic vs. probabilistic systemschatbot.pyDictionary-based O(1) lookup vs. if-elif ladderschatbot.pyContent-based filteringtech_stack_recommender.pyTF-IDF (Term Frequency – Inverse Document Frequency)tech_stack_recommender.pyCosine similarity for vector comparisontech_stack_recommender.pyCold-start handlingtech_stack_recommender.py

Author

Noman Ali — CS student.

License

For educational use as part of the DecodeLabs Industrial Training Kit.
