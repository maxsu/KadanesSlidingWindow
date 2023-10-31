from dataclasses import dataclass


@dataclass
class ModelContext:
    """
    Represents a generation context for a model.
    """
    text: str|None = None
    tokens: list[int]|None = None    
    max_length: int|None = None


class TokenizedModelContext:
    """
    Represents a context that has been tokenized.
    """
    def __len__(self):
        return len(self.tokens)


@ dataclass
class ModelResponse:
    """
    Represents a response from a model.
    """
    tokens: list[int] = []
    text : str | None = None

    def __len__(self):
        return len(self.tokens)

