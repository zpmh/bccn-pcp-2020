import numpy as np
from enum import Enum
from typing import Optional

from typing import Callable, Tuple

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played

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

	#remove everything that isn't part of the board
	row_list.remove("|==============|")
	row_list.remove("|==============|")
	row_list.remove("|0 1 2 3 4 5 6 |")
	row_list.remove('')

	#create smth to store board
	board = [[None] * 7] * 6     #board has 6 rows, and 7 columns

	#loop over rows
	for i, row in enumerate(reversed(row_list)):
		#convert each row (String) to a list and remove "|" and spaces by only getting every second element
		row_list = list(row)[1:-1:2]
		#replace string representation by boardPiece according to player
		row_bp = [PLAYER1 if x == 'X' else NO_PLAYER if x == ' ' else PLAYER2 if x == 'O' else x for x in row_list]
		board[i] = row_bp

	#convert board (list) to a np.ndarray
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
	#TODO: Check if I need a valid action function - currently for invalid action nothing happens
	if copy:
		board_copy = np.copy(board)

	for row in range(6):
		if board[row,action] == NO_PLAYER:
			board[row,action] = player
			break

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
	#check diagonal, row, column for last_action
	#TODO: OPTIMIZE THIS AND SEE IF YOU CAN USE THE LAST_ACTION IN SOME SORTA WAY
	if (last_action):
		if board[last_action-1] == player:
			x=1
	# or loop over all rows and columns and check the column, row, and diagonal for 4 (only for half the board?)
	for row in range(6):
		for column in range(7):
			#check row -- only half of each column, if there are 4 connected in a column they must reach up to first 4
			if column <= 3 and board[row, column] == player and board[row, column + 1] == player and board[row, column + 2] == player and board[row, column + 3] == player:
				return True
			#check column -- up to first 4 pieces
			if row <= 2 and board[row, column] == player and board[row + 1, column] == player and board[row + 2, column] == player and board[row + 3, column] == player:
				return True
			#check upper left - lower right diagonal
			if row <= 2 and column >= 3 and board[row, column] == player and board[row + 1, column - 1] == player and board[row + 2, column - 2] == player and board[row + 3, column - 3] == player:
				return True
			#check lower right upper left diagonal
			if row <= 2 and column <= 3 and board[row, column] == player and board[row + 1, column + 1] == player and board[row + 2, column + 2] == player and board[row + 3, column + 3] == player:
				return True

	#if no connected 4 are found:
	return False

def check_board_full(board):
	'''
	Returns True if board is full, Returns False if there are still empty spots (NO_PLAYER)
	simply loops over every row and checks if there is a spot with NO_PLAYER
	'''
	for row in range(6):
		if board[row,6]== NO_PLAYER:
			return False

	return True

def check_end_state(
	board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
	"""
	Returns the current game state for the current `player`, i.e. has their last
	action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
	or is play still on-going (GameState.STILL_PLAYING)?
	"""
	if (connected_four(board,player,last_action)):
		return GameState.IS_WIN
	elif check_board_full(board):
		return GameState.IS_DRAW
	else:
		return GameState.STILL_PLAYING