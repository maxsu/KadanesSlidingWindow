"""
Test Module for the Kadane Generator Class
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from pytest import fixture

from fixtures import user_yes_no

from kadane.KadaneGenerator import KadaneSlidingWindow
from kadane.KadaneGenerator import KadaneGenerator
from kadane.KadaneGenerator import Request
from kadane.KadaneGenerator import Response


@fixture
class TestHarness:
    # Compose a generator for testing
    generator = KadaneGenerator(
        window=KadaneSlidingWindow(maxlen=1000),
        model=AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-1_5",
            torch_dtype="auto",
            trust_remote_code=True,
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            "microsoft/phi-1_5",
            torch_dtype="auto",
            trust_remote_code=True,
        ),
    )

    # Build a request for testing
    req = Request(
        prompt = "Once upon a time, there was a little girl named Alice who lived in a small village. One day, Alice was playing in the forest when she came across a strange rabbit hole. She followed the rabbit hole down into a wonderland...",
        follow_up_prompt= "After Alice fell down the rabbit hole, she...",
        conversation_length=100,
    )   


def TestKadaneGenerator(TestHarness):
    # Run a generation request
    response : Response = TestHarness.generator(TestHarness.req)

    # Assert response is valid
    assert(len(response.text.full) > 100)
    
    # Expected Effect: User Reads a Story and enjoys it!
    assert(user_yes_no('Did you enjoy reading the story?') == 'yes')


