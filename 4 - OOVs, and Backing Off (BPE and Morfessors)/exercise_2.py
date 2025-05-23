import math
from collections import Counter
from typing import List, Union

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from morfessor.baseline import BaselineModel
import morfessor


class TokenizerEntropy:

    def tokenize_bpe(self, tokenizer: Tokenizer, text: str) -> List[str]:
        return tokenizer.encode(text).tokens

    def tokenize_morfessor(self, tokenizer: BaselineModel, text: str) -> List[str]:
        words = text.strip().split()
        tokens = []
        for word in words:
            tokens.extend(tokenizer.viterbi_segment(word)[0])
        return tokens

    def get_probs(self, tokens: List[str]):
        token_freq = Counter(tokens)
        total = sum(token_freq.values())
        return {token: count / total for token, count in token_freq.items()}

    def compute_entropy(
        self, text: str, tokenizer: Union[Tokenizer, BaselineModel]
    ) -> float:
        if isinstance(tokenizer, Tokenizer):
            tokens = self.tokenize_bpe(tokenizer, text)
        elif isinstance(tokenizer, BaselineModel):
            tokens = self.tokenize_morfessor(tokenizer, text)
        else:
            raise ValueError("Tokenizer not supported.")

        probs = self.get_probs(tokens)
        return -sum(p * math.log2(p) for p in probs.values())

    def compute_entropy_and_oov(
        self, tokenized_texts: List[List[str]], vocab: set
    ) -> (float, float):
        all_tokens = [token for sentence in tokenized_texts for token in sentence]
        total = len(all_tokens)
        oov_count = sum(1 for token in all_tokens if token not in vocab)

        probs = self.get_probs(all_tokens)
        entropy = -sum(p * math.log2(p) for p in probs.values())
        oov_rate = oov_count / total if total else 0.0
        return entropy, oov_rate


def train_bpe(text: List[List[str]], vocab_size: int) -> Tokenizer:
    lines = [" ".join(tokens) for tokens in text]
    with open("bpe_train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tokenizer.train(["bpe_train.txt"], trainer)

    return tokenizer


def entropy_and_oov(test_text: List[List[str]], tokenizer: Tokenizer) -> (float, float):
    all_tokens = []
    oov_count = 0
    total_count = 0
    vocab = tokenizer.get_vocab()

    for sentence in test_text:
        encoded = tokenizer.encode(" ".join(sentence))
        tokens = encoded.tokens
        all_tokens.extend(tokens)
        total_count += len(tokens)
        oov_count += tokens.count("<unk>")

    token_freq = Counter(all_tokens)
    probs = [count / len(all_tokens) for count in token_freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    oov_rate = oov_count / total_count if total_count else 0.0

    return entropy, oov_rate


def prepare_morfessor_file(train_texts: List[List[str]], file_path: str):
    counts = Counter(word for sent in train_texts for word in sent)
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, count in counts.items():
            f.write(f"{count} {word}\n")


def train_morfessor_model(file_path: str, alpha: float = 0.53) -> BaselineModel:
    io = morfessor.MorfessorIO()
    model = BaselineModel(corpusweight=alpha)
    data = io.read_corpus_file(file_path)
    model.load_data(data)
    model.train_batch()
    return model


def morfessor_tokenize(model: BaselineModel, test_texts: List[List[str]]) -> List[List[str]]:
    tokenized = []
    for sentence in test_texts:
        tokens = []
        for word in sentence:
            segments = model.viterbi_segment(word)[0]
            tokens.extend(segments)
        tokenized.append(tokens)
    return tokenized
