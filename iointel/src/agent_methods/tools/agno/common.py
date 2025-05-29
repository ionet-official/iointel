class DisableAgnoRegistryMixin:
    """
    Put this as first parent class when inheriting
    from Agno tool to disable Agno registry,
    because we only care about our own registry."""

    def _register_tools(self):
        """Disabled in favour of iointel registry."""

    def register(self, function, name=None):
        """Disabled in favour of iointel registry."""
