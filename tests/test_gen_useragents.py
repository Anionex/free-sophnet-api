from fake_useragent import FakeUserAgent

def test_gen():
    fua = FakeUserAgent()
    length = 10
    randoms = [fua.random for _ in range(length)]
    randoms_set = set(randoms)
    print(randoms)
    print(randoms_set)
    assert all(randoms)
    assert len(randoms_set) != 1
