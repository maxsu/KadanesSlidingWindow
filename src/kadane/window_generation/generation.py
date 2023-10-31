"""
This file contains an implementation of a Kadane Sliding Window for a HuggingFace causal language model.
"""

from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer

from KadanesSlidingWindow import KadaneSlidingWindow

from Models import IModel, RequestFactory



@dataclass
class Request:
    prompt: str
    follow_up_prompt: str
    conversation_tokens: int


@dataclass
class Response:
    text = ""
    tokens = []


class KadaneGenerator:
    """
    This Class generates text using a language model and a Kadane Window.

    Args:
        window (KadaneSlidingWindow): The sliding window to use for the model.
        model (AutoModelForCausalLM): A causal language model.
        tokenizer (AutoTokenizer): The tokenizer for the language model.
    """

    window: KadaneSlidingWindow
    model: IModel

    def __init__(
        self,
        window: KadaneSlidingWindow,
        model: IModel,
    ):
        self.window = window
        self.model = model
        self.request_factory = RequestFactory(self.model)

    def __call__(self, req: Request):
        # Add the prompt to the sliding window before starting the loop
        self.window.add(req.prompt.split())

        # Generate text until the response is long enough
        while len(response.tokens) < req.conversation_tokens:
            # Ask the model a follow-up prompt
            context = self.request_factory(
                text = req.follow_up_prompt + self.window.get_context()
            )           

            # Generate at most 75 tokens at a time
            max_length = min(len(context) + 75, req.conversation_tokens)

            # Generate text using the model with the encoded context as input
            output_text = self.model.generate(
                context,
                max_length=max_length,
            )

            response.text += output_text

            # Add the generated text to the sliding window
            # Are we sure we want to add the entire text to the sliding window?
            self.window.add(response.text.split())

        print(response.full_text)

        return response