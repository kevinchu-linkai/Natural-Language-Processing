import sys
from collections import defaultdict, Counter
from string import punctuation

def load_data_basic(filename, is_test=False):
    data = []
    with open(filename, 'r') as f:
        content = f.read().strip().split('\n\n')
        for sentence in content:
            if is_test:
                data.append(sentence.split())
            else:
                sentence_data = []
                for word_pos in sentence.split('\n'):
                    word, pos = word_pos.split('\t')
                    sentence_data.append((word, pos))
                data.append(sentence_data)
    return data

def morphological_cues(word):   
    if word.endswith("ing"):
        return "VBG"
    elif word.endswith("ed"):
        return "VBD"
    elif word.endswith("ly"):
        return "RB"
    elif word[-1] in punctuation:
        return word[-1]
    else:
        return "NN"

def viterbi(sentence, likelihood, transitions, oov_likelihood):
    V = [{}]
    backpointer = [{}]
    all_states = list(transitions.keys())

    for state in all_states:
        em_prob = likelihood[state].get(sentence[0], oov_likelihood.get(morphological_cues(sentence[0]), oov_likelihood[state]))
        V[0][state] = transitions["Begin_Sent"].get(state, 0) * em_prob
        backpointer[0][state] = "Begin_Sent"

    for t in range(1, len(sentence)):
        V.append({})
        backpointer.append({})
        for state in all_states:
            max_tr_prob = max([(V[t-1][prev_state] * transitions[prev_state].get(state, 0), prev_state) for prev_state in all_states])
            em_prob = likelihood[state].get(sentence[t], oov_likelihood.get(morphological_cues(sentence[t]), oov_likelihood[state]))
            V[t][state] = max_tr_prob[0] * em_prob
            backpointer[t][state] = max_tr_prob[1]

    prev_st = max([(V[-1][state] * transitions[state].get("End_Sent", 0), state) for state in all_states])[1]
    best_path = [prev_st]

    for t in range(len(sentence)-1, 0, -1):
        current_tag = backpointer[t].get(prev_st, "NN")
        if current_tag in ["WP$", "WDT", "WP"]:
            current_tag = "NN"
        best_path.insert(0, current_tag)
        prev_st = backpointer[t].get(prev_st, "NN")

    new_best_path = []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dts = ["an", "a", "the"]
    ins = ["of", "at", "in", "by", "into"]

    for word, tag in zip(sentence, best_path):
        if word == '.':
            new_best_path.append('.')
        if word == 'which':
            new_best_path.append('WDT')
        elif word in weekdays:
            new_best_path.append('NNP')
        elif word.isdigit() or (word.replace('.','', 1).isdigit() if '.' in word else False) or (word.replace('\/','', 1).isdigit() if '/' in word else False) or (word.replace('/','', 1).isdigit() if '/' in word else False):
            new_best_path.append('CD')
        elif word in dts:
            new_best_path.append('DT')
        elif word in ins:
            new_best_path.append('IN')
        elif word == ',':
            new_best_path.append(',')          
        else:
            new_best_path.append(tag)
    best_path = new_best_path  

    if len(best_path) != len(sentence):
        best_path = ["NN"] * (len(sentence) - len(best_path)) + best_path

    return best_path

def smoothed_transitions(transitions, all_states, smoothing_constant=1e-5):
    smoothed = defaultdict(lambda: defaultdict(float))
    for prev_state in all_states:
        total = sum(transitions[prev_state].values()) + smoothing_constant * len(all_states)
        for state in all_states:
            smoothed[prev_state][state] = (transitions[prev_state].get(state, 0) + smoothing_constant) / total
    return smoothed

def pos_tagging(training_file, develop_file, test_file):
    training_data = load_data_basic(training_file) + load_data_basic(develop_file)
    test_data = load_data_basic(test_file, is_test=True)

    likelihood = defaultdict(lambda: defaultdict(float))
    transitions = defaultdict(lambda: defaultdict(float))

    for sentence in training_data:
        for i, (word, pos) in enumerate(sentence):
            likelihood[pos][word] += 1
            if i == 0:
                transitions["Begin_Sent"][pos] += 1
            else:
                prev_pos = sentence[i-1][1]
                transitions[prev_pos][pos] += 1

    all_states = list(transitions.keys())
    transitions = smoothed_transitions(transitions, all_states)

    oov = defaultdict(Counter)
    for pos, words in likelihood.items():
        for word, freq in words.items():
            if freq == 1:
                oov[pos][word] += 1

    oov_likelihood = defaultdict(float)
    total_oov_occurrences = sum([sum(counter.values()) for counter in oov.values()])
    for pos, word_counts in oov.items():
        oov_likelihood[pos] = sum(word_counts.values()) / total_oov_occurrences

    all_predicted_tags = []
    for sentence in test_data:
        tags = viterbi(sentence, likelihood, transitions, oov_likelihood)
        all_predicted_tags.append(tags)

    output_filename = "submission.pos"
    with open(output_filename, 'w') as f:
        for sentence, tags in zip(test_data, all_predicted_tags):
            for word, tag in zip(sentence, tags):
                f.write(f"{word}\t{tag}\n")
            f.write("\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: kc4624_HW3.py training_file develop_file test_file")
        sys.exit(1)

    training_file = sys.argv[1]
    develop_file = sys.argv[2]
    test_file = sys.argv[3]

    pos_tagging(training_file, develop_file, test_file)
    print(f"Output written to submission.pos")

if __name__ == '__main__':
    main()
