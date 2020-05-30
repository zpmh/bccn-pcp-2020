import numpy as np
from agents.common import BoardPiece, PlayerAction, SavedState, NO_PLAYER
from typing import Optional, Tuple

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    open_columns = [0,1,2,3,4,5,6]

    #first check if the column is full by checking last row of board
    for i in board[5,:]:
        if i != NO_PLAYER:
            open_columns.remove(i)

    #then pick a random column to play
    action = np.random.choice(open_columns)

    return action, saved_state