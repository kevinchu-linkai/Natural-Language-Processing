def enhanced_simple_stemmer(word):
    """
    A more enhanced simple stemmer that trims common word endings.
    """
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def is_capitalized(word):
    """
    Check if the word is capitalized.
    """
    return word[0].isupper()

def extract_features(lines, include_bio_tag=True):
    """
    Extract enhanced features from lines of the WSJ file.
    include_bio_tag: Set to False for test data (no BIO tags in the .pos file)
    """
    features = []
    for i, line in enumerate(lines):
        if line.strip():
            parts = line.strip().split('\t')
            token, pos_tag = parts[0], parts[1]
            bio_tag = parts[2] if include_bio_tag else None

            # Basic features
            feature_line = [token, f"POS={pos_tag}", f"STEM={enhanced_simple_stemmer(token)}", f"CAPITALIZED={is_capitalized(token)}"]

            # Features of previous words
            if i > 1 and lines[i-2].strip():
                prev_token_2, prev_pos_tag_2 = lines[i-2].strip().split('\t')[:2]
                feature_line.extend([f"prev2_POS={prev_pos_tag_2}", f"prev2_word={prev_token_2}"])

            if i > 0 and lines[i-1].strip():
                prev_token_1, prev_pos_tag_1 = lines[i-1].strip().split('\t')[:2]
                feature_line.extend([f"prev1_POS={prev_pos_tag_1}", f"prev1_word={prev_token_1}"])

            # Features of next words
            if i < len(lines) - 2 and lines[i+2].strip():
                next_token_2, next_pos_tag_2 = lines[i+2].strip().split('\t')[:2]
                feature_line.extend([f"next2_POS={next_pos_tag_2}", f"next2_word={next_token_2}"])

            if i < len(lines) - 1 and lines[i+1].strip():
                next_token_1, next_pos_tag_1 = lines[i+1].strip().split('\t')[:2]
                feature_line.extend([f"next1_POS={next_pos_tag_1}", f"next1_word={next_token_1}"])

            # Add BIO tag for training data
            if bio_tag:
                feature_line.append(bio_tag)

            features.append('\t'.join(feature_line))
        else:
            # Preserve sentence boundaries
            features.append('')

    return features

# Paths for the input files
training_file_path = 'WSJ_02-21.pos-chunk'  # Update with actual file path
test_file_path = 'WSJ_23.pos'              # Update with actual file path

# Read and process the training data
with open(training_file_path, 'r') as file:
    training_lines = file.readlines()
training_features = extract_features(training_lines)

# Read and process the test data
with open(test_file_path, 'r') as file:
    test_lines = file.readlines()
test_features = extract_features(test_lines, include_bio_tag=False)

# Write the features to the output files
with open('training.feature', 'w') as file:  # Update with actual file path
    for line in training_features:
        file.write(f"{line}\n")

with open('test.feature', 'w') as file:     # Update with actual file path
    for line in test_features:
        file.write(f"{line}\n")