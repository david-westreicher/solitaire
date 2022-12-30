from dataclasses import dataclass
from random import shuffle
from collections import defaultdict
from copy import deepcopy
import heapq
from analyze import load_state


@dataclass(eq=True, frozen=True)
class Card:
    suite: str
    number: int

    def __str__(self):
        return f"{self.suite}{self.number}"


SUITE_STACK = {
    "R": 0,
    "G": 1,
    "B": 2,
    "S": 3,
}


class Board:
    def __init__(self, cards=None) -> None:
        self.stacks = [[] for _ in range(8)]
        self.temp = [[] for _ in range(3)]
        self.finished = [[] for _ in range(4)]

        if not cards:
            return
        stack = 0
        while len(cards):
            self.stacks[stack % len(self.stacks)].append(cards.pop(0))
            stack += 1
        self.normalize()

    def __lt__(self, other):
        return self.score > other.score

    def normalize(self):
        change = False
        for s in self.stacks + self.temp:
            if not s or s[-1].number == 0:
                continue
            last = s[-1]
            finished_stack = self.finished[SUITE_STACK[last.suite]]
            if last.number == 1:
                change = True
                finished_stack.append(s.pop())
            elif finished_stack and finished_stack[-1].number + 1 == last.number:
                change = True
                finished_stack.append(s.pop())

        counter = defaultdict(list)
        for s in self.stacks + self.temp:
            if s and s[-1].number == 0:
                counter[s[-1].suite].append(s)
        for suite, stacks in counter.items():
            if len(stacks) != 4:
                continue
            tmp_collect_index = -1
            try:
                tmp_collect_index = next(
                    i
                    for i in range(len(self.temp))
                    if self.temp[i] and self.temp[i][-1] == Card(suite, 0)
                )
            except StopIteration:
                pass
            try:
                tmp_collect_index = next(
                    i for i in range(len(self.temp)) if not self.temp[i]
                )
            except StopIteration:
                pass
            if tmp_collect_index == -1:
                continue
            for s in stacks:
                assert s[-1] == Card(suite, 0)
                s.pop()
            self.temp[tmp_collect_index] = [Card(suite, 0)] * 4
            change = True
        assert sum(len(s) for s in self.temp + self.stacks + self.finished) == 40
        assert all(len(s) <= 1 or all(c.number == 0 for c in s) for s in self.temp)

        if change:
            self.normalize()

    @property
    def is_finished(self):
        return all(len(s) == 0 for s in self.stacks)

    def compress(self):
        stacks = tuple(tuple(s) for s in self.stacks)
        temp = tuple(tuple(s) for s in self.stacks)
        finished = tuple(tuple(s) for s in self.stacks)
        return (stacks, temp, finished)

    def is_nice(self, stack):
        if len(stack) > 1 and any(c.number == 0 for c in stack):
            return False
        for i in range(len(stack) - 1):
            if stack[i].suite == stack[i + 1].suite:
                return False
            if stack[i].number - 1 != stack[i + 1].number:
                return False
        return True

    def children(self):
        stacks = [(s, False) for s in self.stacks] + [(s, True) for s in self.temp]
        for i in range(len(stacks)):
            for j in range(len(stacks)):
                if i == j:
                    continue
                s1, is_tmp1 = stacks[i]
                s2, is_tmp2 = stacks[j]
                if (is_tmp1 and is_tmp2) or (is_tmp2 and s2):
                    continue
                before = len(s2)
                start = 0 if not is_tmp2 else len(s1) - 1
                for k in range(start, len(s1)):
                    if not self.is_nice(s1[k:]):
                        continue
                    if (
                        len(s2) == 0
                        or s2[-1].suite != s1[k].suite
                        and s1[k].number != 0
                        and s1[k].number + 1 == s2[-1].number
                    ):
                        s2.extend(s1[k:])
                        del s1[k:]
                        yield self.clone(), (i, j)
                        s1.extend(s2[before:])
                        del s2[before:]

    def clone(self):
        copy = deepcopy(self)
        copy.normalize()
        return copy

    @property
    def score(self):
        return sum(len(s) for s in self.temp)

    def __str__(self) -> str:
        rows = []
        row = [(s[-1], len(s)) if s else (None, 0) for s in self.temp + self.finished]
        rows.append(
            "  |  ".join(
                (
                    " ".join(f"{e}{cnt}" if e else " - " for e, cnt in r)
                    for r in [row[:3], row[3:]]
                )
            )
        )
        rows.append("")
        max_row = max(len(s) for s in self.stacks)
        for i in range(max_row):
            row = []
            for s in self.stacks:
                row.append(None if i >= len(s) else s[i])
            rows.append("  ".join([str(e) if e else "--" for e in row]))
        rows.append("")
        return "\n".join(rows)


def generate_deck():
    deck = []
    for suite in ["R", "G", "B"]:
        for i in range(1, 10):
            deck.append(Card(suite, i))
        for _ in range(4):
            deck.append(Card(suite, 0))
    deck.append(Card("S", 1))
    assert len(deck) == 40, len(deck)
    return deck


def main():
    """
    cards = generate_deck()
    shuffle(cards)
    """
    cards = [Card(c[0], int(c[1])) for c in load_state("./solitaire.png")]
    print(cards)
    board = Board(cards)
    heap = [(board, [(board, None)])]
    seen = set()
    count = 0
    while heap:
        board, path = heapq.heappop(heap)
        if board.is_finished:
            print("FINISHED!!!" * 100)
            print(board)
            for b, m in path:
                print(m)
                print(b)
            print(len(path))
            break
        if count % 5000 == 0:
            print("wait")
            print(board)
        count += 1
        compressed = board.compress()
        if compressed in seen:
            continue
        seen.add(compressed)
        for child, move in board.children():
            heapq.heappush(heap, (child, path + [(child, move)]))


if __name__ == "__main__":
    main()
