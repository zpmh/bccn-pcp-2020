import numpy as np
from enum import Enum
from typing import Optional, Callable, Tuple
from numba import njit
from scipy.signal.sigtools import _convolve2d


BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played

CONNECT_N = 4

class SavedState:
	pass

GenMove = Callable[
	[np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
	Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

class GameState(Enum):
	IS_WIN = 1
	IS_DRAW = -1
	STILL_PLAYING = 0

def initialize_game_state() -> np.ndarray:
	"""
	Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
	"""
	return np.zeros((6, 7), dtype=BoardPiece)

def pretty_print_board(board: np.ndarray) -> str:
	"""
	Should return `board` converted to a human readable string representation,
	to be used when playing or printing diagnostics to the console (stdout). The piece in
	board[0, 0] should appear in the lower-left. Here's an example output:
	|==============|
	|              |
	|              |
	|    X X       |
	|    O X X     |
	|  O X O O     |
	|  O O X X     |
	|==============|
	|0 1 2 3 4 5 6 |
	"""
	pretty_board = [[None] * 16] * 9
	pretty_board[0] = "|==============|\n"
	pretty_board[7] = "|==============|\n"
	pretty_board[8] = "|0 1 2 3 4 5 6 |\n"
	i = 1
	for row in reversed(board):
		strrow = np.where(row[:] == NO_PLAYER, '  ', row)
		strrow = np.where(row[:] == PLAYER1, 'X ', strrow)
		strrow = np.where(row[:] == PLAYER2, 'O ', strrow)
		pretty_board[i] = "|" + ''.join(strrow) + "|\n"
		i+=1

	pretty_board_str = "".join(pretty_board)

	return pretty_board_str

def string_to_board(pp_board: str) -> np.ndarray:
	"""
	Takes the output of pretty_print_board and turns it back into an ndarray.
	This is quite useful for debugging, when the agent crashed and you have the last
	board state as a string.
	"""

	#convert pp_board to a list of rows
	row_list = list(pp_board.split("\n"))

	#remove everything that isn't part of the actual board
	row_list.remove("|==============|")
	row_list.remove("|==============|")
	row_list.remove("|0 1 2 3 4 5 6 |")
	row_list.remove('')

	#create smth to store board
	board = [[None] * 7] * 6     #board has 6 rows, and 7 columns

	#loop over rows
	for i, row in enumerate(row_list):
		#convert each row (string) to a list and remove "|" and spaces by only getting every second element
		row_list = list(row)[1:-1:2]
		#replace string representation by BoardPiece according to player
		board[i] = [PLAYER1 if x == 'X' else NO_PLAYER if x == ' ' else PLAYER2 if x == 'O' else x for x in row_list]

	#convert board (list) to np.ndarray
	board = np.array(board)
	#reverse order of rows
	board = np.flip(board,0)

	return board

def apply_player_action(
	board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
	"""
	Sets board[i, action] = player, where i is the lowest open row. The modified
	board is returned. If copy is True, makes a copy of the board before modifying it.
	"""
	if copy:
		board_copy = np.copy(board)

	for row in range(6):
		if board[row,action] == NO_PLAYER:
			board[row,action] = player
			break

	if copy:
		return board_copy, board
	else:
		return board

def connected_four(
	board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
	"""
	Returns True if there are four adjacent pieces equal to `player` arranged
	in either a horizontal, vertical, or diagonal line. Returns False otherwise.
	If desired, the last action taken (i.e. last column played) can be provided
	for potential speed optimisation.
	"""

	#loop over all rows and columns and check the column, row, and diagonal for adjacent 4 (only for half the board)
	for row in range(board.shape[0]):
		for column in range(board.shape[1]):
			#check row -- only half of each column, if there are 4 connected in a column they must reach up to first 4
			if column <= 3 and board[row, column] == player and board[row, column + 1] == player and board[row, column + 2] == player and board[row, column + 3] == player:
				return True
			#check column -- up to first 4 pieces
			if row <= 2 and board[row, column] == player and board[row + 1, column] == player and board[row + 2, column] == player and board[row + 3, column] == player:
				return True
			#check upper left - lower right diagonal -> this checks all possible diagonals bc looping over row and column
			if row <= 2 and column >= 3 and board[row, column] == player and board[row + 1, column - 1] == player and board[row + 2, column - 2] == player and board[row + 3, column - 3] == player:
				return True
			#check lower right upper left diagonal
			if row <= 2 and column <= 3 and board[row, column] == player and board[row + 1, column + 1] == player and board[row + 2, column + 2] == player and board[row + 3, column + 3] == player:
				return True

	#if no connected 4 are found
	return False

@njit()
def connected_four_iter(
	board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
	rows, cols = board.shape
	rows_edge = rows - CONNECT_N + 1
	cols_edge = cols - CONNECT_N + 1

	for i in range(rows):
		for j in range(cols_edge):
			if np.all(board[i, j:j+CONNECT_N] == player):
				return True

	for i in range(rows_edge):
		for j in range(cols):
			if np.all(board[i:i+CONNECT_N, j] == player):
				return True

	for i in range(rows_edge):
		for j in range(cols_edge):
			block = board[i:i+CONNECT_N, j:j+CONNECT_N]
			if np.all(np.diag(block) == player):
				return True
			if np.all(np.diag(block[::-1, :]) == player):
				return True

	return False

#glob variables required for connected_four_convolve
col_kernel = np.ones((CONNECT_N, 1), dtype=BoardPiece)
row_kernel = np.ones((1, CONNECT_N), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(CONNECT_N, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(CONNECT_N, dtype=BoardPiece))[::-1, :])

def connected_four_convolve(
	board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
	board = board.copy()

	other_player = BoardPiece(player % 2 + 1)
	board[board == other_player] = NO_PLAYER
	board[board == player] = BoardPiece(1)

	for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
		result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
		if np.any(result == CONNECT_N):
			return True
	return False

def check_board_full(board: np.ndarray) -> bool:
	"""
	Checks if there is any spot in the last row of board that's still empty (NO_PLAYER)
	:param board: current state of board
	:return: True if board is filled, False otherwise
	"""
	return np.all(board[board.shape[0] - 1, :] != NO_PLAYER)

def check_end_state(
	board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
	"""
	Returns the current game state for the current `player`, i.e. has their last
	action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
	or is play still on-going (GameState.STILL_PLAYING)?
	"""
	if (connected_four_convolve(board,player,last_action)):
		return GameState.IS_WIN
	elif check_board_full(board):
		return GameState.IS_DRAW
	else:
		return GameState.STILL_PLAYING

def check_open_columns(board: np.ndarray) -> list:
	'''
	Returns list of all open columns by checking which columns in last row are equal to NO_PLAYER
	:param board: current state of board
	:return: list of open columns
	'''
	return list(np.argwhere(board[board.shape[0] - 1, :] == NO_PLAYER).flatten())