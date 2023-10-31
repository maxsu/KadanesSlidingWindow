from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import IModel, ModelContext, ModelResponse, TokenizedModelContext

from kadane.adaptors.huggingface_api_adaptor import HuggingFaceModelAdaptor

class CausalModel(IModel):
    """
    A causal language model.
    """

    name = "" 

    api_adaptor: HuggingFaceModelAdaptor

    def __init__(self):
        # Create the API adaptor
        api_adaptor = HuggingFaceModelAdaptor(self.name)

        # Map the API methods
        self._tokenizer = api_adaptor.tokenize
        self._decode = api_adaptor.decode
        self._generate = api_adaptor.generate
        
    def tokenize(self, context: ModelContext) -> TokenizedModelContext:
        """
        Allow users to tokenize a context without generating text.

        Args:
            context (ModelContext): The context to tokenize.

        Returns:
            TokenizedModelContext: The tokenized context.
        """
        # Tokenize the context
        self._tokenizer(context)

        return context

    def generate(self, context:ModelContext) -> ModelResponse:
        """
        Generate text using the model.

        Args:
            context (ModelContext): The context to generate text from.
        
        Returns:
            ModelResponse: The generated text.
        """
        # Tokenize the context if necessary
        if not context.tokens:
            self._tokenize(context)

        # Generate tokens
        response = self._generate(context)

        # Decode the tokens
        self._decode(response) 

        return response