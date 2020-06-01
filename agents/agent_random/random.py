import numpy as np
from agents.common import BoardPiece, PlayerAction, SavedState, NO_PLAYER
from typing import Optional, Tuple

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

    #get a an array of all open columns
    open_columns = np.argwhere(board[board.shape[0] - 1, :] == NO_PLAYER).flatten()
    # then pick a random column to play
    PlayerAction = np.random.choice(open_columns)

    return PlayerAction, saved_state

