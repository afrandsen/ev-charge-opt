from typing import List


class RunLogger:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, msg: str) -> None:
        print(msg)
        self.lines.append(str(msg))
