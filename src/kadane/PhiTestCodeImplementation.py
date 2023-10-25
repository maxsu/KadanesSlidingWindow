"""
This file contains an implementation of a Kadane Sliding Window for the Phi model.

Configuration:
    PROMPT (str): The prompt to use for the Phi model.
    FOLLOW_UP_PROMPT (str): The follow-up prompt to use for the Phi model.
    MAX_LEN (int): The maximum length of the sliding window.
    MODEL (str): The name of the Phi model to use.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from KadanesSlidingWindow import KadaneSlidingWindow


# Configuration
PROMPT = "Once upon a time, there was a little girl named Alice who lived in a small village. One day, Alice was playing in the forest when she came across a strange rabbit hole. She followed the rabbit hole down into a wonderland..."
FOLLOW_UP_PROMPT = "After Alice fell down the rabbit hole, she..."
MAX_LEN = 1000
MODEL = "microsoft/phi-1_5"


def PhiTestCodeImplementation(
    window: KadaneSlidingWindow,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
):
    """
    Generate text using the Phi model.

    Args:
        window (KadaneSlidingWindow): The sliding window to use for the Phi model.
        model (AutoModelForCausalLM): The Phi model.
        tokenizer (AutoTokenizer): The tokenizer for the Phi model.
    """
    # Add the prompt to the sliding window before starting the loop
    window.add(PROMPT.split())

    generated_text = ""
    while len(generated_text) < 100:
        # Ask the model a follow-up prompt
        context = FOLLOW_UP_PROMPT + window.get_context()

        # Encode the context using the tokenizer
        input_ids = tokenizer(context, return_tensors="pt").input_ids

        # Generate text using the model with the encoded context as input
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=min(100, len(input_ids) + 75),
        )

        # Decode the generated text using the tokenizer
        generated_text += tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        # Add the generated text to the sliding window
        window.add(generated_text.split())

    print(generated_text)


if __name__ == "__main__":
    # Create a Kadane Sliding Window with a size of MAX_LEN tokens
    window: KadaneSlidingWindow = KadaneSlidingWindow(maxlen=MAX_LEN)

    # Initialize the Causal Language Model and Tokenizer
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    # Generate text using the Phi model and the Kadane Sliding Window
    PhiTestCodeImplementation(
        window=window,
        model=model,
        tokenizer=tokenizer,
    )
