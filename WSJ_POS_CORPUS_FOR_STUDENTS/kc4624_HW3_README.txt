Kevin Chu
1. Data Loading:
   * The load_data_basic function reads the input data, either training/development or test data. For training and development sets, it splits the data into word-POS pairs, while for the test set, it just reads the words.
2. Morphological Clues:
   * To handle OOV, the program uses the morphological_cues function. It checks the ending of the word to make a guess about its POS. For instance, words ending in "ing" are likely VBG, while those ending in "ed" are likely VBD.
3. Viterbi Algorithm:
   * The core of the program is the viterbi function, which applies the Viterbi algorithm. This function constructs two tables, V and backpointer, to keep track of the most probable tags for each word in a sentence and their respective backpointers.
   * The program uses the likelihood of a word given a tag and the transition probability from one tag to another to calculate the most probable path of tags for a given sentence.
4. Handling OOV Words:
   * For OOV, the program uses the morphological cues and a distribution of tags for single-occurrence words in the training data (oov_likelihood) to make a guess about their tags.
5. Post-processing:
   * After determining the most probable tags, the program ensures that all periods are tagged as ".".
6. Pipeline:
   * The pos_tagging function integrates all the steps, including loading the data, computing likelihoods and transition probabilities, applying the Viterbi algorithm, and writing the results to an output file named out.pos.
How to Operate the Program:
1. Prepare a training file, a development file, and a test file. The training and development files should contain word-POS pairs separated by tabs, with each sentence separated by a blank line. The test file should contain words separated by spaces, with each sentence separated by a blank line.
2. Run the program using the command:
python kc4624_HW3.py training_file develop_file test_file