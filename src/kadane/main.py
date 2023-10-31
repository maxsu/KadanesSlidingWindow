from sliding_window.controller import KadaneController


config, request = cli.get_config()







from click import command, option
ModelRepository = dict[str, IModel]
from base import Environment

MODEL_REPOSITORY: ModelRepository = None

@command()
@option('--model', default='phi-1_5', help='The model to use')
@option('--prompt', default='The quick brown fox jumps over the lazy dog', help='The prompt to use')
@option('--follow-up-prompt', default='The quick brown fox jumps over the lazy dog', help='The follow up prompt to use for extending the context')
def main(model, prompt, follow_up_prompt):
    print(f'Using model: {model}')
    print(f'Prompt: {prompt}')
    print(f'Follow up prompt: {follow_up_prompt}')

    # Create an environment
    env = Environment(model)


