import numpy as np
import math
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, GameState
from agents.common import initialize_game_state, pretty_print_board, string_to_board, connected_four, apply_player_action, check_board_full, check_end_state
from agents.agent_minimax.agent_minimax import *

still_playing_board = "|==============|\n" \
                  "|X   O     X O |\n" \
                  "|X O X O   X X |\n" \
                  "|O O X O   O O |\n" \
                  "|O O X X O X X |\n" \
                  "|X X O O O X O |\n" \
                  "|X X X O X X X |\n" \
                  "|==============|\n" \
                  "|0 1 2 3 4 5 6 |\n"

def test_check_open_columns():

	open_cols = string_to_board(still_playing_board)
	check_open_columns(open_cols)

	assert list(check_open_columns(open_cols)) == [1,3,4]

def test_generate_move_minimax():

	board = initialize_game_state()
	ret = generate_move_minimax(board, PLAYER1, board)

	#first move should be in column 3 and saved state not modified
	assert ret == (3,board)

def test_center_column_score():

	filled_board = string_to_board(still_playing_board)
	ret = center_column_score(filled_board,PLAYER1)

	#PLAYER1 has 1 piece in column so score should be 3
	assert ret == 3

	ret = center_column_score(filled_board,PLAYER2)

	#PLAYER2 has 4 pieces in column so score should be 15
	assert ret == 4*3

	empty_board = initialize_game_state()
	ret = center_column_score(empty_board,PLAYER1)

	#should be 0
	assert ret == 0

def test_even_odd_scores():

	filled_board = string_to_board(still_playing_board)
	ret = even_odd_row_scores(filled_board, PLAYER1)

	#PLAYER1 has 6 pieces in odd rows
	assert ret == 6*2

	#TODO: check whats up here
	ret = even_odd_row_scores(filled_board, PLAYER2)

	#PLAYER2 also has 6 pieces in even rows
	assert ret == 6

def test_adjacent_score():

	board = initialize_game_state()
	#test case 1: horizontal
	board[0, 0] = PLAYER1
	board[0, 1] = PLAYER1
	board[0, 2] = PLAYER1
	board[0, 3] = PLAYER1
	ret = adjacent_score(list(board[0,0:4]), PLAYER1)

	assert ret == 10000

	#test case 2: vertical
	board[0, 6] = PLAYER1
	board[1, 6] = PLAYER1
	board[2, 6] = PLAYER1
	board[3, 6] = PLAYER1
	ret = adjacent_score(list(board[0:4,6]), PLAYER1)

	assert ret == 10000

	#test case 3: postive diagonal
	board = initialize_game_state()
	board[0, 0] = PLAYER1
	board[1, 1] = PLAYER1
	board[2, 2] = PLAYER1
	pos_list = (board[0,0],board[1,1],board[2,2],board[3,3])
	ret = adjacent_score(pos_list, PLAYER1)

	assert ret == 40

	#test case 4: postive diagonal
	board[0, 6] = PLAYER1
	board[1, 5] = PLAYER1
	pos_list = (board[0,6],board[1,5],board[2,4],board[3,3])
	ret = adjacent_score(pos_list, PLAYER1)

	assert ret == 5

	#test case 5: block the opponent
	board[2,4] = PLAYER1
	pos_list = (board[0,6],board[1,5],board[2,4],board[3,3])
	ret = adjacent_score(pos_list, PLAYER2)

	assert ret == -60

def test_heuristic():

	board = initialize_game_state()

	#should return 3 for first move
	ret = heuristic(board,PLAYER1)

	assert ret == 0

	board[0,0] = PLAYER1
	board[0,1] = PLAYER1
	board[0,2] = PLAYER1

	ret = heuristic(board,PLAYER1)

	#this should be equal to output of vertical adjacent_score for 3 pieces

	#calling multiple times so should be 3 adjacent pieces in first call + 2 adjacent pieces in second call
	score = adjacent_score(list(board[0,0:4]),PLAYER1) + adjacent_score(list(board[0,1:5]),PLAYER1)

	assert score == ret

def test_minimax():

	board = initialize_game_state()

	#first move should be in center column 3
	assert minimax(board, 4, -math.inf, math.inf, PLAYER1, True) == (3,6)


