"""
Controller for the KadaneGenerator use case.

Handles validation and generation of requests.
"""

from generator import KadaneGenerator, Request, Response
from errors import BadRequest


class Validate:
    def __call__(self, req: Request) -> bool:
        request_valid = all(
            req.follow_up_prompt,
            req.prompt,
            req.conversation_tokens > 0,
        )

        if request_valid:
            return True
        else:
            raise BadRequest("Bad generation request:", req)


class KadaneController:
    """
    This class is a controller for the KadaneGenerator use case.
    """

    generator: KadaneGenerator

    def __init__(self, config):
        self.generator = config.generator

    def __call__(self, req: Request) -> Response:
        # Validate the request
        Validate()(req)

        # Generate a response
        resp : Response = self.generator(req)
        return resp