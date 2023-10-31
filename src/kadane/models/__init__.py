from dataclasses import dataclass


@dataclass
class ModelContext:
    """
    Represents a generation context for a model.
    """
    text: str|None = None
    tokens: list[int]|None = None    
    max_length: int|None = None

    def __len__(self):
        return len(self.tokens)