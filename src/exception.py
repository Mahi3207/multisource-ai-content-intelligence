"""Exceptions whose messages are safe to show to normal users."""


class ContentIntelligenceError(Exception):
    """Base class. str(error) is always a friendly, user-facing message."""


class ContentLoadError(ContentIntelligenceError):
    """The source could not be read or contained no usable text."""


class ContentTooLargeError(ContentIntelligenceError):
    """The text is too large to be analysed reliably."""


class AnalysisError(ContentIntelligenceError):
    """The LLM analysis failed (network, rate limit, service error...)."""
