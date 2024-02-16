import re
import sys

def identify_telephone_numbers(text):
    telephone_pattern = r'\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    matches = re.findall(telephone_pattern, text)
    return matches

if len(sys.argv) != 2:
    print("Usage: python program2.py input_file.txt")
    sys.exit(1)

input_file_name = sys.argv[1]

try:
    with open(input_file_name, 'r') as corpus_file:
        corpus_text = corpus_file.read().replace('\n', ' ')
except FileNotFoundError:
    print(f"File '{input_file_name}' not found.")
    sys.exit(1)

telephone_matches = identify_telephone_numbers(corpus_text)

with open('telephone_output.txt', 'w') as output_file:
    for match in telephone_matches:
        output_file.write(match + '\n')

print("Done")
