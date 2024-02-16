import re
import sys

def identify_dollar_amounts(text):
    dollar_pattern = r'(?:\$\s*)?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|dollars|dollar|cents|cent))(?:\s+and\s+\d+\s+(?:cents|cent))?'
    matches = re.findall(dollar_pattern, text)
    return matches

if len(sys.argv) != 2:
    print("Usage: dollar_program.py input_file.txt")
    sys.exit(1)

input_file_name = sys.argv[1]

try:
    with open(input_file_name, 'r') as corpus_file:
        corpus_text = corpus_file.read().replace('\n', ' ')
except FileNotFoundError:
    print(f"File '{input_file_name}' not found.")
    sys.exit(1)

dollar_matches = identify_dollar_amounts(corpus_text)

with open('dollar_output.txt', 'w') as output_file:
    for match in dollar_matches:
        output_file.write(match + '\n')

print("Done")
