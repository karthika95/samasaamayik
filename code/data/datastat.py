import re
import statistics
from collections import Counter

def get_data_statistics(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    total_sentences = len(sentences)
    print(f"Total sentences: {total_sentences}")

    # Tokenize by whitespace (adjust if you have your own tokenizer)
    tokenized_sentences = [re.findall(r'\S+', s) for s in sentences]
    tokens = [tok for sent in tokenized_sentences for tok in sent]

    total_tokens = len(tokens)
    vocab = set(tokens)

    # Sentence length stats (in tokens)
    sent_lengths = [len(s) for s in tokenized_sentences]
    avg_len = statistics.mean(sent_lengths)
    median_len = statistics.median(sent_lengths)
    max_len = max(sent_lengths)
    min_len = min(sent_lengths)

    # Character-based stats
    char_counts = [len(s) for s in sentences]
    avg_chars = statistics.mean(char_counts)
    max_chars = max(char_counts)

    # Word-level stats
    word_lengths = [len(w) for w in tokens]
    avg_word_len = statistics.mean(word_lengths)

    # Type–Token Ratio
    ttr = len(vocab) / total_tokens

    # Print summary
    print("\n--- Dataset Statistics ---")
    print(f"Total Sentences       : {total_sentences}")
    print(f"Total Tokens          : {total_tokens}")
    print(f"Vocabulary Size       : {len(vocab)}")
    print(f"Type–Token Ratio (TTR): {ttr:.4f}")
    print(f"Average Sent Length   : {avg_len:.2f} tokens")
    print(f"Median Sent Length    : {median_len:.2f} tokens")
    print(f"Max Sent Length       : {max_len} tokens")
    print(f"Average Sent Length   : {avg_chars:.2f} chars")
    print(f"Max Sentence Length   : {max_chars} chars")
    print(f"Average Word Length   : {avg_word_len:.2f} chars")

    # Optional: most frequent tokens
    print("\nTop 20 most frequent tokens:")
    for tok, freq in Counter(tokens).most_common(20):
        print(f"{tok}\t{freq}")

    return {
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
        "avg_sent_len_tokens": avg_len,
        "median_sent_len_tokens": median_len,
        "max_sent_len_tokens": max_len,
        "avg_sent_len_chars": avg_chars,
        "max_sent_len_chars": max_chars,
        "avg_word_len": avg_word_len,
        "ttr": ttr,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute dataset statistics for Hindi benchmark corpus")
    parser.add_argument("file", help="Path to the text file (one sentence per line)")
    args = parser.parse_args()

    get_data_statistics(args.file)
