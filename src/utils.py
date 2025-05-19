from fake_useragent import FakeUserAgent

_fua = FakeUserAgent()


def fake_useragent() -> str:
    """
    Returns a random user agent string using the FakeUserAgent library.
    """
    return str(_fua.random)
