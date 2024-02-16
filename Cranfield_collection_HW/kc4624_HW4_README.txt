1. Data Loading and Parsing
load_data(filename):
* Loads the content of a file into memory.
parse_cranfield_data(data, data_type="query"):
* Parses the Cranfield dataset, extracting either the queries or the abstracts, depending on the data_type parameter.
2. Preprocessing
preprocess_text(text):
* Tokenizes and lowercases the input text.
* Removes stop words (common words that don't contribute much to the meaning of a document) using a predefined stop list.
* Uses the Porter stemming algorithm to reduce words to their root/base form. This helps in treating different forms of a word as a single entity (e.g., "running" and "runner" both become "run").
3. Term Frequency and Inverse Document Frequency Computation
compute_tf(data, top_idf_words):
* Calculates the term frequency (TF) of words in the document. If a word exists in the document, its TF is set to 1, otherwise, it's 0.
* Only considers the terms that are present in the top IDF scoring words.
compute_idf(data):
* Calculates the inverse document frequency (IDF) of each word in the corpus. IDF diminishes the weight of terms that occur very frequently and increases the weight of terms that occur rarely.
compute_top_idf_words(data, N):
* Returns the top N words with the highest IDF scores.
4. TF-IDF Computation
compute_tfidf(tf_data, idf_data):
* Computes the TF-IDF score for each word in each document/query. This score represents the importance of a word in a document relative to the entire corpus.
5. Cosine Similarity
cosine_similarity(vec_a, vec_b):
* Calculates the cosine similarity between two vectors. This metric determines the cosine of the angle between two vectors, providing a measure of their similarity. A value of 1 indicates complete similarity, while a value of 0 indicates no similarity.
6. Ranking and Output Generation
The code compares each query vector with every abstract vector using cosine similarity. If the similarity score is greater than zero, it's stored in a list. The results are then sorted in descending order of similarity for each query. The top 100 matching abstracts for each query, based on their cosine similarity scores, are written to the output file.