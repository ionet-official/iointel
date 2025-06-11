from agno.tools.trello import TrelloTools as AgnoTrelloTools
from .common import make_base, wrap_tool


class Trello(make_base(AgnoTrelloTools)):
    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            api_secret=self.api_secret_,
            token=self.token_,
            create_card=self.create_card_,
            get_board_lists=self.get_board_lists_,
            move_card=self.move_card_,
            get_cards=self.get_cards_,
            create_board=self.create_board_,
            create_list=self.create_list_,
            list_boards=self.list_boards,
        )

    @wrap_tool("agno__trello__create_card", AgnoTrelloTools.create_card)
    def create_card(
        self, board_id: str, list_name: str, card_title: str, description: str = ""
    ) -> str:
        return self.create_card(self, board_id, list_name, card_title, description)

    @wrap_tool("agno__trello__get_board_lists", AgnoTrelloTools.get_board_lists)
    def get_board_lists(self, board_id: str) -> str:
        return self.get_board_lists(self, board_id)

    @wrap_tool("agno__trello__move_card", AgnoTrelloTools.move_card)
    def move_card(self, card_id: str, list_id: str) -> str:
        return self.move_card(self, card_id, list_id)

    @wrap_tool("agno__trello__get_cards", AgnoTrelloTools.get_cards)
    def get_cards(self, list_id: str) -> str:
        return self.get_cards(self, list_id)

    @wrap_tool("agno__trello__create_board", AgnoTrelloTools.create_board)
    def create_board(self, name: str, default_lists: bool = False) -> str:
        return self.create_board(self, name, default_lists)

    @wrap_tool("agno__trello__create_list", AgnoTrelloTools.create_list)
    def create_list(self, board_id: str, list_name: str, pos: str = "bottom") -> str:
        return self.create_list(self, board_id, list_name, pos)

    @wrap_tool("agno__trello__list_boards", AgnoTrelloTools.list_boards)
    def list_boards(self, board_filter: str = "all") -> str:
        return self.list_boards(self, board_filter)
