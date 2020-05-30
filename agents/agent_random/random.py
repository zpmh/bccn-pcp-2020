import numpy as np
from agents.common import BoardPiece, PlayerAction, SavedState
from typing import Optional, Tuple

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action = 0
    return action, saved_state