from typing import Optional
from agno.tools.trello import TrelloTools as AgnoTrelloTools
from .common import make_base, wrap_tool
from pydantic import Field


class Trello(make_base(AgnoTrelloTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    api_secret: Optional[str] = Field(default=None, frozen=True)
    token: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            api_secret=self.api_secret_,
            token=self.token_,
            create_card=True,
            get_board_lists=True,
            move_card=True,
            get_cards=True,
            create_board=True,
            create_list=True,
            list_boards=True,
        )

    @wrap_tool("agno__trello__create_card", AgnoTrelloTools.create_card)
    def create_card(
        self, board_id: str, list_name: str, card_title: str, description: str = ""
    ) -> str:
        return self._tool.create_card(board_id, list_name, card_title, description)

    @wrap_tool("agno__trello__get_board_lists", AgnoTrelloTools.get_board_lists)
    def get_board_lists(self, board_id: str) -> str:
        return self._tool.get_board_lists(board_id)

    @wrap_tool("agno__trello__move_card", AgnoTrelloTools.move_card)
    def move_card(self, card_id: str, list_id: str) -> str:
        return self._tool.move_card(card_id, list_id)

    @wrap_tool("agno__trello__get_cards", AgnoTrelloTools.get_cards)
    def get_cards(self, list_id: str) -> str:
        return self._tool.get_cards(list_id)

    @wrap_tool("agno__trello__create_board", AgnoTrelloTools.create_board)
    def create_board(self, name: str, default_lists: bool = False) -> str:
        return self._tool.create_board(name, default_lists)

    @wrap_tool("agno__trello__create_list", AgnoTrelloTools.create_list)
    def create_list(self, board_id: str, list_name: str, pos: str = "bottom") -> str:
        return self._tool.create_list(board_id, list_name, pos)

    @wrap_tool("agno__trello__list_boards", AgnoTrelloTools.list_boards)
    def list_boards(self, board_filter: str = "all") -> str:
        return self._tool.list_boards(board_filter)
