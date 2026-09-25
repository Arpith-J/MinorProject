"""Direct-access client: asks the server by message id, receives the message.

The client has no privacy requirements; it simply sends the id over the (not
shown) channel and returns whatever the server answers.
"""

from __future__ import annotations

from src.common.message import Message
from src.direct.server import DirectServer


class DirectClient:
    """Client that retrieves a message by revealing its id."""

    def __init__(self, server: DirectServer) -> None:
        """Bind the client to a server.

        Args:
            server: The :class:`DirectServer` to query.
        """
        self.server = server

    def retrieve(self, msg_id: int) -> Message:
        """Retrieve the message with the given id.

        The id is sent verbatim to the server.

        Args:
            msg_id: The id of the message to retrieve.

        Returns:
            The requested :class:`Message`.
        """
        return self.server.query(msg_id)