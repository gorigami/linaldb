class LinalError(Exception):
    """Raised for a server-reported error (``status: "error"``) or a
    response whose shape doesn't match ``clients/CONTRACT.md``.

    Network-level failures (connection refused, DNS, etc.) are *not*
    wrapped in this — they propagate as whatever ``requests`` itself
    raises (``requests.exceptions.RequestException`` and subclasses),
    per the contract's "never swallowed" rule.
    """
