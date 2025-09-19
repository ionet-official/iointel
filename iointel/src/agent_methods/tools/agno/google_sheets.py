from typing import Any, List, Optional
from agno.tools.googlesheets import GoogleSheetsTools as AgnoGoogleSheetsTools
from .common import make_base, wrap_tool
from pydantic import Field


class GoogleSheets(make_base(AgnoGoogleSheetsTools)):
    scopes: Optional[List[str]] = Field(default=None, frozen=True)
    spreadsheet_id: Optional[str] = Field(default=None, frozen=True)
    spreadsheet_range: Optional[str] = Field(default=None, frozen=True)
    creds: Optional[Any] = Field(default=None, frozen=True)
    creds_path: Optional[str] = Field(default=None, frozen=True)
    token_path: Optional[str] = Field(default=None, frozen=True)
    read: bool = Field(default=True, frozen=True)
    create: bool = Field(default=False, frozen=True)
    update: bool = Field(default=False, frozen=True)
    duplicate: bool = Field(default=False, frozen=True)

    def _get_tool(self):
        return self.Inner(
            scopes=self.scopes,
            spreadsheet_id=self.spreadsheet_id,
            spreadsheet_range=self.spreadsheet_range,
            creds=self.creds,
            creds_path=self.creds_path,
            token_path=self.token_path,
            read=self.read,
            create=self.create,
            update=self.update,
            duplicate=self.duplicate,
        )

    @wrap_tool("agno__google_sheets__read_sheet", AgnoGoogleSheetsTools.read_sheet)
    def read_sheet(
        self,
        spreadsheet_id: Optional[str] = None,
        spreadsheet_range: Optional[str] = None,
    ) -> str:
        return self._tool.read_sheet(spreadsheet_id, spreadsheet_range)

    @wrap_tool("agno__google_sheets__create_sheet", AgnoGoogleSheetsTools.create_sheet)
    def create_sheet(self, title: str) -> str:
        return self._tool.create_sheet(title)

    @wrap_tool("agno__google_sheets__update_sheet", AgnoGoogleSheetsTools.update_sheet)
    def update_sheet(
        self,
        data: List[List[Any]],
        spreadsheet_id: Optional[str] = None,
        range_name: Optional[str] = None,
    ) -> str:
        return self._tool.update_sheet(data, spreadsheet_id, range_name)

    @wrap_tool(
        "agno__google_sheets__create_duplicate_sheet",
        AgnoGoogleSheetsTools.create_duplicate_sheet,
    )
    def create_duplicate_sheet(
        self,
        source_id: str,
        new_title: Optional[str] = None,
        copy_permissions: bool = True,
    ) -> str:
        return self._tool.create_duplicate_sheet(source_id, new_title, copy_permissions)
