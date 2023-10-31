from abc import ABC, abstractmethod
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer

from value_objects import ModelContext, ModelResponse


class IModelApiHelper(ABC):
    @abstractmethod
    def _generate(self, context: ModelContext)->ModelResponse:
        ...

    @abstractmethod
    def _tokenize(self, context: ModelContext)-> None:
        ...

    @abstractmethod
    def _decode(self, ModelResponse)->None:
        ...

@dataclass
class HuggingFaceContext:
    input_ids: list[int]
    max_length: int

    def __init__(self, ctx: ModelContext):
        self.input_ids=ctx.tokens
        self.max_length=ctx.max_length

class HuggingFaceGenerator:
    ...

def HuggingFaceCodec(cls):

    TOKENIZER = cls
    cls.tokenize = lambda ctx: TOKENIZER(ctx.text, return_tensors="pt" ).input_ids

    def wrapped_generator(ctx:ModelContext):
        # Tokenize request if necessary
        if not ctx.tokens:
            ctx.tokens = self._tokenize(ctx)

    ...

    
@dataclass
class HuggingFaceModel:
    """
    Represents a HuggingFace model and tokenizer.
    """
    def __init__(self, model_name: str):
        # Load the model and tokenizer
        PRETRAINED_MODEL = AutoModelForCausalLM.from_pretrained(model_name).generate
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)

        # Map the api end points
        self._generate = lambda ctx: PRETRAINED_MODEL.generate(*ctx)[0]
        self._tokenize = lambda ctx: TOKENIZER(ctx.text, return_tensors="pt" ).input_ids
        self._decode = lambda res: TOKENIZER.decode(res.tokens, skip_special_tokens=True)
    
    
    def tokenize(self, ctx: ModelContext) -> TokenizedModelContext:
        ctx.tokens = self._tokenize(ctx)


    def generate(self, ctx: ModelContext)-> ModelResponse:
        # Build huggingface context
        hug = HuggingFaceContext(ctx)

        # Generate Tokens
        response = ModelResponse(self._generate(*hug))
        
        # Decode response
        response.text = self._decode(response)
        return response
    
