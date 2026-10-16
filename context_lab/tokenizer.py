"""A fixed UTF-8 byte vocabulary: no fitting, unknown tokens, or test-set leakage."""


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    vocab_size = 258

    def encode(self, text: str, *, boundaries: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        return [self.bos_id, *tokens, self.eos_id] if boundaries else tokens

    def decode(self, tokens: list[int]) -> str:
        return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")
