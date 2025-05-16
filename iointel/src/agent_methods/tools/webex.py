import json
import os
from typing import Optional

import logging

from ...utilities.decorators import register_tool

logger = logging.getLogger(__name__)

try:
    from webexpythonsdk import WebexAPI
    from webexpythonsdk.exceptions import ApiError
except ImportError:
    logger.error("Webex tools require the `webexpythonsdk` package. Run `pip install webexpythonsdk` to install it.")


class WebexTools:
    def __init__(self, access_token: Optional[str] = None):

        if access_token is None:
            access_token = os.getenv("WEBEX_ACCESS_TOKEN")
        if access_token is None:
            raise ValueError("Webex access token is not set. Please set the WEBEX_ACCESS_TOKEN environment variable.")

        self.client = WebexAPI(access_token=access_token)

    @register_tool(name="webex_send_message")
    def send_message(self, room_id: str, text: str) -> str:
        """
        Send a message to a Webex Room.
        Args:
            room_id (str): The Room ID to send the message to.
            text (str): The text of the message to send.
        Returns:
            str: A JSON string containing the response from the Webex.
        """
        try:
            response = self.client.messages.create(roomId=room_id, text=text)
            return json.dumps(response.json_data)
        except ApiError as e:
            logger.error(f"Error sending message: {e} in room: {room_id}")
            return json.dumps({"error": str(e)})

    @register_tool(name="webex_list_rooms")
    def list_rooms(self) -> str:
        """
        List all rooms in the Webex.
        Returns:
            str: A JSON string containing the list of rooms.
        """
        try:
            response = self.client.rooms.list()
            rooms_list = [
                {
                    "id": room.id,
                    "title": room.title,
                    "type": room.type,
                    "isPublic": room.isPublic,
                    "isReadOnly": room.isReadOnly,
                }
                for room in response
            ]

            return json.dumps({"rooms": rooms_list}, indent=4)
        except ApiError as e:
            logger.error(f"Error listing rooms: {e}")
            return json.dumps({"error": str(e)})
