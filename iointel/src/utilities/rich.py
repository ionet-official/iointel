from contextlib import contextmanager
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({"electric_blue": "#7DF9FF"})

console = Console(theme=custom_theme)


class PrettyController:
    def __init__(self):
        self._enabled = True

    @property
    def is_enabled(self):
        return self._enabled

    def _onoff(self, enabled: bool):
        @contextmanager
        def inner(this=self):
            old_value, this._enabled = this._enabled, enabled
            try:
                yield this
            finally:
                this._enabled = old_value

        return inner()

    def enabled(self):
        return self._onoff(True)

    def disabled(self):
        return self._onoff(False)

    def global_onoff(self, enabled: bool):
        self._enabled = enabled


pretty_output = PrettyController()
