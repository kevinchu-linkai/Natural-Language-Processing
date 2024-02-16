import re
import numpy as np
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def load_data(filename):
    with open(filename, 'r') as f:
        content = f.read()
    return content

stop_list_content = load_data('stop_list.py')
stop_list = eval(re.search(r"closed_class_stop_words = (.+)", stop_list_content, re.DOTALL).group(1))

cran_queries = load_data('cran.qry')
cran_abstracts = load_data('cran.all.1400')

def parse_cranfield_data(data, data_type="query"):
    parsed_data = {}
    items = re.split(r"\.I \d+", data)[1:]
    for item in items:
        match = re.search(r"\.W\n(.+)", item, re.DOTALL)
        if match:
            content = match.group(1).strip()
            parsed_data[len(parsed_data) + 1] = content
    return parsed_data

queries = parse_cranfield_data(cran_queries, data_type="query")
abstracts = parse_cranfield_data(cran_abstracts, data_type="abstract")

def preprocess_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word.lower()) for word in tokens if word.lower() not in stop_list and word.isalpha()]
    return tokens

preprocessed_queries = {qid: preprocess_text(query) for qid, query in queries.items()}
preprocessed_abstracts = {aid: preprocess_text(abstract) for aid, abstract in abstracts.items()}

def compute_tf(data, top_idf_words):
    tf_data = {}
    for doc_id, tokens in data.items():
        token_freq = defaultdict(int)
        for token in tokens:
            if token in top_idf_words:
                token_freq[token] += 1
        if token_freq: 
            tf_data[doc_id] = {token: 1 for token, freq in token_freq.items()} 
    return tf_data

def compute_idf(data):
    num_docs = len(data)
    term_doc_count = defaultdict(int)
    for _, tokens in data.items():
        for token in set(tokens):
            term_doc_count[token] += 1
    idf_data = {term: np.log(num_docs / count) for term, count in term_doc_count.items()}
    return idf_data

def compute_top_idf_words(data, N):
    idf_data = compute_idf(data)
    sorted_idf = sorted(idf_data.items(), key=lambda x: x[1], reverse=True)
    top_n_words = set([word for word, _ in sorted_idf[:N]])
    return top_n_words

N = 100  
top_idf_words_abstracts = compute_top_idf_words(preprocessed_abstracts, N)

for _, query in preprocessed_queries.items():
    top_idf_words_abstracts.update(query)

tf_queries = compute_tf(preprocessed_queries, top_idf_words_abstracts)
tf_abstracts = compute_tf(preprocessed_abstracts, top_idf_words_abstracts)

idf_queries = compute_idf(preprocessed_queries)
idf_abstracts = compute_idf(preprocessed_abstracts)

def compute_tfidf(tf_data, idf_data):
    tfidf_data = {}
    for doc_id, token_freq in tf_data.items():
        tfidf_scores = {}
        for token, freq in token_freq.items():
            tfidf_scores[token] = freq * idf_data.get(token, 0)
        tfidf_data[doc_id] = tfidf_scores
    return tfidf_data

tfidf_queries = compute_tfidf(tf_queries, idf_queries)
tfidf_abstracts = compute_tfidf(tf_abstracts, idf_abstracts)

def cosine_similarity(vec_a, vec_b):
    common_terms = set(vec_a.keys()).intersection(set(vec_b.keys()))
    dot_product = sum([vec_a[term] * vec_b[term] for term in common_terms])
    magnitude_a = np.sqrt(sum([val**2 for val in vec_a.values()]))
    magnitude_b = np.sqrt(sum([val**2 for val in vec_b.values()]))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)

similarity_scores = defaultdict(list)
for qid, q_vec in tfidf_queries.items():
    for aid, abs_vec in tfidf_abstracts.items():
        sim_score = cosine_similarity(q_vec, abs_vec)
        if sim_score > 0:
            similarity_scores[qid].append((aid, sim_score))
for qid in similarity_scores:
    similarity_scores[qid].sort(key=lambda x: x[1], reverse=True)

output_data = []
for qid, scores in similarity_scores.items():
    for aid, sim_score in scores[:100]:
        output_data.append(f"{qid} {aid} {sim_score:.3f}")

with open("output.txt", 'w') as f:
    f.write('\n'.join(output_data))
