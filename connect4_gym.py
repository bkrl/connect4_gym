from typing import Callable, Literal, Optional, Union

import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register
from gym.utils.renderer import Renderer
from numpy.typing import NDArray

ObsType = NDArray[np.bool_]
ActType = int

DISPLAY_CHARACTER = {1: "🟡", -1: "🔴", 0: "⚪"}


class Connect4(gym.Env[ObsType, ActType]):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        get_opponent: Callable[
            [], Callable[[ObsType, NDArray[np.bool_]], int]
        ] = lambda: lambda _observation, _action_mask: int(input("Enter your move: ")),
        render_mode: Optional[str] = None,
        rows: int = 6,
        columns: int = 7,
        win_length: int = 4,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.get_opponent = get_opponent
        self.rows = rows
        self.columns = columns
        self.win_length = win_length

        self.observation_space = spaces.Box(0.0, 1.0, (2, rows, columns))
        self.action_space = spaces.Discrete(columns)
        self.reward_range = (-1.0, 1.0)

        self._renderer = Renderer(self.render_mode, self._render)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)

        self._opponent = self.get_opponent()
        self._board = np.zeros((self.rows, self.columns), dtype=np.int8)
        self._num_moves = 0

        if self.np_random.choice([True, False]):
            # Opponent moves first
            self._opponent_move()

        self._renderer.reset()
        self._renderer.render_step()

        return (self._get_observation(), {}) if return_info else self._get_observation()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if not self.action_space.contains(action) or self._board[-1, action] != 0:
            raise ValueError("invalid action")

        row = self._move(action, 1)
        self._renderer.render_step()
        if self._is_win(row, action):
            return self._get_observation(), 1.0, True, False, {}
        if self._num_moves == self.rows * self.columns:
            return self._get_observation(), 0.0, True, False, {}

        opponent_row, opponent_column = self._opponent_move()
        self._renderer.render_step()
        if self._is_win(opponent_row, opponent_column):
            return self._get_observation(), -1.0, True, False, {}
        if self._num_moves == self.rows * self.columns:
            return self._get_observation(), 0.0, True, False, {}

        return self._get_observation(), 0.0, False, False, {}

    def action_masks(self) -> NDArray[np.bool_]:
        return self._board[-1] == 0

    def _get_observation(self, player: Literal[1, -1] = 1) -> ObsType:
        player_pieces = self._board == player
        opponent_pieces = self._board == -player
        return np.stack([player_pieces, opponent_pieces]).astype(np.float32)

    def _move(self, column: int, piece: Literal[1, -1]) -> int:
        self._num_moves += 1
        for row in range(self.rows):
            if self._board[row, column] == 0:
                self._board[row, column] = piece
                return row
        raise ValueError("invalid move")

    def _opponent_move(self) -> tuple[int, int]:
        column = self._opponent(self._get_observation(-1), self.action_masks())
        if not self.action_space.contains(column) or self._board[-1, column] != 0:
            raise ValueError(f"invalid opponent move: {repr(column)}")

        return self._move(column, -1), column

    def _is_win(self, row: int, column: int) -> bool:
        return (
            _line_length(self._board[row::-1, column]) + 1 >= self.win_length
            or _line_length(self._board[row, column::-1])
            + _line_length(self._board[row, column:])
            + 1
            >= self.win_length
            or _line_length(self._board[row:, column:].diagonal())
            + _line_length(self._board[row::-1, column::-1].diagonal())
            + 1
            >= self.win_length
            or _line_length(self._board[row:, column::-1].diagonal())
            + _line_length(self._board[row::-1, column:].diagonal())
            + 1
            >= self.win_length
        )

    def render(self, mode: str = "human") -> None:
        if self.render_mode is not None:
            return self._renderer.get_renders()  # type: ignore
        return self._render(mode)

    def _render(self, mode: str) -> None:
        if mode == "human":
            for row in reversed(self._board):
                for piece in row:
                    print(DISPLAY_CHARACTER[piece], end="")
                print()
        else:
            raise ValueError("invalid render mode")


def _line_length(line: NDArray[np.int8]) -> int:
    length = 0
    for piece in line[1:]:
        if piece == line[0]:
            length += 1
        else:
            break
    return length


register(
    id="Connect4-v0",
    entry_point=Connect4,
    reward_threshold=1.0,
)
