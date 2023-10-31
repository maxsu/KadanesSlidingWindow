from abc import ABC, abstractmethod
from dataclasses import dataclass

from value_objects import ModelContext, ModelResponse

class IModel(ABC):
    """
    Base interface for all models.
    """

    @abstractmethod
    def tokenize(self, context: ModelContext) -> ModelContext:
        """
        Tokenize a context without generating text.

        Args:
            context (ModelContext): The context to tokenize.

        Returns:
            ModelContext: The tokenized context.
        """
        ...
    
    @abstractmethod
    def generate(self, context:ModelContext) -> ModelContext:
        """
        Generate text using the model.

        Args:
            context (ModelContext): The context to generate text from.

        Returns:
            ModelResponse: The generated text.
        """
        ...