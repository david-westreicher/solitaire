from collections import defaultdict
from concurrent.futures import TimeoutError
from copy import deepcopy
from dataclasses import dataclass
import heapq
import time
from typing import Optional

from analyze import (
    COLUMN_OFFSET,
    GAME_BOX,
    ROW_OFFSET,
    START,
    WIDTH,
    load_state_indirect,
)
from mss import mss
from pebble import concurrent
import pyautogui


MOUSE_TIME = 0.5
CLICK_TIME = 0.5


@dataclass()
class Move:
    from_stack: int
    from_card: int
    to_stack: int
    to_card: int
    collect_suite: Optional[str] = None


@dataclass(eq=True, frozen=True, order=True)
class Card:
    suite: str
    number: int

    def __str__(self):
        return f"{self.suite}{self.number}"


class Board:
    def __init__(self, cards=None, suite_stack=None) -> None:
        self.stacks = [[] for _ in range(8)]
        self.temp = [[] for _ in range(3)]
        self.finished = [[] for _ in range(4)]
        self.suite_stack = ["S"] + (suite_stack or [])

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
        for stack_num, s in enumerate(self.stacks + self.temp):
            if not s or s[-1].number == 0:
                continue
            last = s[-1]
            finished_stack = (
                self.finished[self.suite_stack.index(last.suite)]
                if last.suite in self.suite_stack
                else []
            )
            if last.number == 1:
                change = True
                if last.suite not in self.suite_stack:
                    self.suite_stack.append(last.suite)
                self.finished[self.suite_stack.index(last.suite)].append(s.pop())
                yield Move(
                    stack_num, len(s), 11 + self.suite_stack.index(last.suite), 0
                )
            elif finished_stack and finished_stack[-1].number + 1 == last.number:
                change = True
                finished_stack.append(s.pop())
                yield Move(
                    stack_num, len(s), 11 + self.suite_stack.index(last.suite), 0
                )

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
            yield Move(0, 0, 0, 0, collect_suite=suite)
        assert sum(len(s) for s in self.temp + self.stacks + self.finished) == 40
        assert all(len(s) <= 1 or all(c.number == 0 for c in s) for s in self.temp)

        if change:
            yield from self.normalize()

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
                        copy, normalize_moves = self.clone()
                        yield copy, [Move(i, k, j, before)] + normalize_moves
                        s1.extend(s2[before:])
                        del s2[before:]

    def clone(self):
        copy = deepcopy(self)
        normalize_moves = list(copy.normalize())
        return copy, normalize_moves

    @property
    def score(self):
        return sum(len(s) for s in self.finished)

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


def take_screenshot():
    # return "./monitor-1.png"
    with mss() as sct:
        sct.shot()
    time.sleep(2)
    return "./monitor-1.png"


@concurrent.process(timeout=20)
def compute_path(cards, order):
    board = Board(cards, suite_stack=order)
    heap = [(0, 0, board, [(board, [])])]
    seen = set()
    count = 0
    while heap:
        _, _, board, path = heapq.heappop(heap)
        if board.is_finished:
            return path
        if count % 5000 == 0:
            print("wait", len(path))
            print(board)
        count += 1
        compressed = board.compress()
        if compressed in seen:
            continue
        seen.add(compressed)
        for child, move in board.children():
            heapq.heappush(
                heap, (-child.score, len(path), child, path + [(child, move)])
            )


def execute_move(move: Move):
    CARD_WIDTH = 100
    MOVE_START = (GAME_BOX[0] + START[0], GAME_BOX[1] + START[1])

    def calc_pos(stack, card):
        if stack < 8:
            x = MOVE_START[0] + COLUMN_OFFSET * stack + CARD_WIDTH // 2
            y = MOVE_START[1] + ROW_OFFSET * card + WIDTH // 2
        elif stack < 11:
            stack -= 8
            x = MOVE_START[0] + COLUMN_OFFSET * stack + CARD_WIDTH // 2
            y = MOVE_START[1] - WIDTH * 13 + WIDTH // 2
        else:
            stack -= 11
            x = MOVE_START[0] + COLUMN_OFFSET * (stack + 4) + CARD_WIDTH // 2
            y = MOVE_START[1] - WIDTH * 13 + WIDTH // 2
        return x, y

    if move.collect_suite:
        suite_y = ["R", "G", "B"].index(move.collect_suite)
        x = MOVE_START[0] + COLUMN_OFFSET * 3 + WIDTH // 2
        y = MOVE_START[-1] + WIDTH * (-12 + suite_y * 4)
        pyautogui.moveTo(x, y)
        for _ in range(5):
            pyautogui.mouseDown()
            time.sleep(0.05)
            pyautogui.mouseUp()
            time.sleep(0.05)
        time.sleep(0.5)
    else:
        from_x, from_y = calc_pos(move.from_stack, move.from_card)
        to_x, to_y = calc_pos(move.to_stack, move.to_card)
        pyautogui.moveTo(from_x, from_y)
        pyautogui.mouseDown()
        time.sleep(0.05)
        pyautogui.moveTo(to_x, to_y)
        time.sleep(0.05)
        pyautogui.mouseUp()
        time.sleep(0.05)


def new_game():
    x = GAME_BOX[2] - 20
    y = GAME_BOX[3] - 20
    pyautogui.moveTo(x, y)
    pyautogui.dragTo(x, y, CLICK_TIME, button="left")


def play_round():
    screenshot = take_screenshot()
    cards, order = load_state_indirect(screenshot)
    cards = [Card(c[0], int(c[1])) for c in cards]
    if set(cards) != set(generate_deck()):
        print("PARSE ERROR")
        return False
    try:
        path = compute_path(cards, order).result()
    except TimeoutError:
        print("TIMEOUT")
        return False
    print("FINISHED!!!" * 100, len(path))
    for b, moves in path:
        print(moves)
        print(b)
        for move in moves:
            execute_move(move)
    print(len(path))
    return True


def main():
    time.sleep(1)
    while True:
        play_round()
        new_game()
        time.sleep(5)


if __name__ == "__main__":
    main()
