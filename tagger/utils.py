from typing import List


def split_str(s: str, separator=',') -> List[str]:
    return list(map(str.strip, s.split(separator)))
