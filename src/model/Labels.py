def from_char(char: str) -> int:
    if len(char) != 1:
        raise Exception("Label must be length 1, got", char)

    return ord(char) - ord('a')


def from_int(char: int) -> str:
    if char > from_char('z'):
        raise Exception("Label must be a-z, got", char)

    return str(chr(char + ord('a')))

