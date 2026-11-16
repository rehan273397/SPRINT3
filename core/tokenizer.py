
import re
from collections import Counter


class TextTokenizer:

    def __init__(self, max_vocab_size=30000):
        self.max_vocab_size = max_vocab_size
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1
        }

    def tokenize(self, text):
        return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

    def fit(self, texts):
        frequencies = Counter()

        for text in texts:
            frequencies.update(self.tokenize(str(text)))

        most_common = frequencies.most_common(
            self.max_vocab_size - 2
        )

        for word, _ in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)

    def encode(self, text, max_length=256):
        tokens = self.tokenize(str(text))

        token_ids = [
            self.word_to_id.get(token, 1)
            for token in tokens[:max_length]
        ]

        token_ids += [0] * (max_length - len(token_ids))

        return token_ids

    @property
    def vocab_size(self):
        return len(self.word_to_id)

