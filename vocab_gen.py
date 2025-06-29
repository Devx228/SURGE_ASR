from collections import Counter
import re

# File paths
hindi_file = "data/hindi.tsv"
sanskrit_file = "data/sanskrit.tsv"
output_vocab_file = "data/vocab.txt"
vocab_size = 100000

# Special tokens for ASR
special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']

# Function to extract words from one transcript file
def extract_words(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            sentence = parts[1]
            words += re.findall(r'\S+', sentence)  # Split on whitespace
    return words

# Extract and count words
hindi_words = extract_words(hindi_file)
sanskrit_words = extract_words(sanskrit_file)
all_words = hindi_words + sanskrit_words

word_counter = Counter(all_words)
most_common_words = [word for word, _ in word_counter.most_common(vocab_size - len(special_tokens))]
final_vocab = special_tokens + most_common_words

# Write vocab to file
with open(output_vocab_file, 'w', encoding='utf-8') as f:
    for word in final_vocab:
        f.write(word + '\n')

print(f"✅ Combined vocabulary saved to {output_vocab_file} with {len(final_vocab)} words.")
