import numpy as np
import math
from typing import Optional, Tuple
from agents.common import BoardPiece, PlayerAction, SavedState, PLAYER1, PLAYER2, NO_PLAYER, GameState
from agents.common import connected_four, check_end_state, apply_player_action

#num_rows = board.shape[0]
#num_columns = board.shape[1]

def generate_move_minimax(
	board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

	alpha = -math.inf
	beta = math.inf
	depth = 4

	# Choose a valid, non-full column that maximizes score and return it as `action`
	PlayerAction = minimax(board, depth, alpha, beta, player, True)[0]

	return PlayerAction, saved_state

def check_open_columns(board: np.ndarray) -> list:
	'''
	Returns list of all open columns by checking which columns in last row are equal to NO_PLAYER
	:param board: current state of board
	:return: list of open columns
	'''
	return np.argwhere(board[board.shape[0] - 1, :] == NO_PLAYER).flatten()

def center_column_score(board: np.ndarray, player: BoardPiece) -> int:
	'''
	Prefer playing pieces in center column of the board
	:param board: current state of board
	:param player: agent
	:return: increased score for column in the center of board
	'''
	center_column = int(board.shape[1] / 2)
	center_column = list(board[:,center_column])
	pieces_count = center_column.count(player)

	return pieces_count * 3

def even_odd_row_scores(board: np.ndarray, player: BoardPiece) -> int:
	'''
	Prefer playing pieces in the odd or even rows depending on player
	:param board: current state of the board
	:param player: agent
	:return: increased score for even or odd rows of the board depending on player
	'''

	score = 0

	if (player == PLAYER1):
		# prefer odd rows
		start = 0
	else: #player must be PLAYER2
		#prefer even rows
		start = 1

	#get even or odd rows depending on start
	for row in np.arange(start,board.shape[0], 2):
		even_odd_row = list(board[row,:])
		row_score = even_odd_row.count(player)
		score += row_score * 2

	return score

def adjacent_score(adjacent_four: list, player: BoardPiece) -> int:
	"""
	Counts how many pieces of a specified player are adjacent in all directions and assigns score
	:param adjacent_four: four adjacent spots on the board in any direction (horizontal, vertial, diagonal)
	:param player: agent
	:return: score for playing piece at spot within adjacent_four
	"""
	score = 0

	#check which player score to maximize and which player to block
	if player == PLAYER1:
		opponent_player = PLAYER2
	else:
		opponent_player = PLAYER1

	#check if agent (player) is close to getting a win by placing 4 adjacent pieces
	if adjacent_four.count(player) == 4:
		score += 10000
	elif adjacent_four.count(player) == 3 and adjacent_four.count(NO_PLAYER) == 1:
		score += 40
	elif adjacent_four.count(player) == 2 and adjacent_four.count(NO_PLAYER) == 2:
		score += 5

	#block opponent from getting a win
	if adjacent_four.count(opponent_player) == 3 and adjacent_four.count(NO_PLAYER) == 1:
		score -= 60

	return score

def heuristic(board: np.ndarray, player: BoardPiece) -> int:
	'''
	Calculates score considering 4 adjacent spots of the board in each row, column, and diagonal
	(checks how many empty and filled spots there are in 4 adjacent spots in all directions)
	:param board: current state of board
	:param player: player who wants to maximize score
	:return: score that can be achieve by playing open position
	'''

	num_rows = board.shape[0]
	num_columns = board.shape[1]

	score = 0

	#Prefer moves in the center column
	score += center_column_score(board, player)

	#Prefer moves in even or odd rows depending on player
	score += even_odd_row_scores(board, player)

	# first check horizontal
	for row in range(num_rows): #loop through every row
		row = board[row,:]
		#now loop through each column and check 4 adjacent spots
		for col in range(num_columns-3): #only need to loop through first 4 cols, since adjacent_4 would touch (j+3)
			adjacent_four = list(row[col:col+4]) #convert to list to apply count() lalter
			#now count the number of pieces for each player
			score += adjacent_score(adjacent_four,player)

	# vertical
	for col in range(num_columns):
		col = board[:,col]
		for row in range(num_rows-3):
			adjacent_four = list(col[row:row+4])
			score += adjacent_score(adjacent_four, player)

	#score positive sloped diagonal
	for row in range(num_rows-3): #rows
		for col in range(num_columns-3): # cols
			adjacent_four = [board[row+i,col+i] for i in range(4)]
			score += adjacent_score(adjacent_four,player)

	#score negative diagonal
	for row in range(num_rows-3): #rows
		for col in range(num_columns-3): # cols
			adjacent_four = [board[row + 3 - i, col + i] for i in range(4)] #col increases but row decreases
			score += adjacent_score(adjacent_four, player)

	return score

def minimax(board: np.ndarray, depth: int, alpha: int, beta: int, player: BoardPiece, maximizing_player: bool) -> Tuple[int, int]:
	'''
	Returns a column where action should be placed and the min and max score for GameState
	:param board: current state of board
	:param depth: depth of search tree
	:param maximizingPlayer: True if we want to max for player
	:return: min or max score for action of player
	'''

	#check which player is the agent so that we don't max/min for wrong player
	if player == PLAYER1:
		opponent_player = PLAYER2
	else:
		opponent_player = PLAYER1

	#check which columns are currently open
	open_cols = check_open_columns(board)

	#check if depth is 0
	if depth == 0:
		score = heuristic(board, player)
		return None, score

	#check if we're at a leaf/terminal node
	if check_end_state(board,player) != GameState.STILL_PLAYING:
		if connected_four(board, player): #agent won
			return None, 100000
		if connected_four(board, opponent_player): #opponent won
			return None, -100000
		else: #must be a draw
			return None, 0

	if maximizing_player: #get max score for agent
		score = -math.inf
		#column = np.random.choice(open_cols)
		for column in open_cols:
			#now simulate making a move and check what score it would get, save the original board in board
			board, board_copy = apply_player_action(board, column, player, True)
			# recursive call to minimax with depth-1 with board_copy so board isn't modified
			next_score = minimax(board_copy, depth-1, alpha, beta, player, False)[1] #get only the score
			#if the score is better save score and column
			if next_score > score:
				score = next_score
				action_column = column
			#evaluate alpha for early stopping
			alpha = max(alpha, score)
			if alpha >= beta: #don't evaluate more options down this path of tree
				break
		return action_column, score

	else:
		score = math.inf
		#column = np.random.choice(open_cols)
		for column in open_cols:
			board, action_board = apply_player_action(board, column, opponent_player, True)
			next_score = minimax(action_board, depth-1, alpha, beta, player, True)[1]
			if next_score < score:
				score = next_score
				action_column = column
			beta = min(beta, score) #here we wanna minimize since we're opponent player
			if alpha >= beta:
				break
		return action_column, score