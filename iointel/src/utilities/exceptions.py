"""Custom exceptions for IOIntel."""


class IOIntelError(Exception):
    """Base error for IOIntel."""


class ValidationError(IOIntelError):
    """Error in validating parameters or return values."""


class ResourceError(IOIntelError):
    """Error in resource operations."""


class ToolError(IOIntelError):
    """Error in tool operations."""


class InvalidSignature(Exception):
    """Invalid signature for use with IOIntel."""