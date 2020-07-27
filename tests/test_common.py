import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, GameState
from agents.common import initialize_game_state, pretty_print_board, string_to_board, connected_four, apply_player_action, check_board_full, check_end_state, check_open_columns

#test cases

#empty board
empty_board = "|==============|\n" \
              "|              |\n" \
              "|              |\n" \
              "|              |\n" \
              "|              |\n" \
              "|              |\n" \
              "|              |\n" \
              "|==============|\n" \
              "|0 1 2 3 4 5 6 |\n"

#one piece of player1 (X) placed on board
one_piece_board = "|==============|\n" \
                  "|              |\n" \
                  "|              |\n" \
                  "|              |\n" \
                  "|              |\n" \
                  "|              |\n" \
                  "|X             |\n" \
                  "|==============|\n" \
                  "|0 1 2 3 4 5 6 |\n"

#full board in a draw
full_draw_board = "|==============|\n" \
                  "|X X O O X X O |\n" \
                  "|X O X O X X X |\n" \
                  "|O O X O X O O |\n" \
                  "|O X X X O X X |\n" \
                  "|X X O O O X O |\n" \
                  "|X X X O X X X |\n" \
                  "|==============|\n" \
                  "|0 1 2 3 4 5 6 |\n"

still_playing_board = "|==============|\n" \
                  "|X   O     X O |\n" \
                  "|X O X O   X X |\n" \
                  "|O O X O   O O |\n" \
                  "|O O X X O X X |\n" \
                  "|X X O O O X O |\n" \
                  "|X X X O X X X |\n" \
                  "|==============|\n" \
                  "|0 1 2 3 4 5 6 |\n"

def test_initialize_game_state():

	ret = initialize_game_state()

	assert isinstance(ret, np.ndarray)
	assert ret.dtype == BoardPiece
	assert ret.shape == (6, 7)
	assert np.all(ret == NO_PLAYER)

def test_pretty_print_board():

	#test case: empty board
	board = initialize_game_state()
	ret = pretty_print_board(board)

	#assert pretty_print_board  equal to the empty test board created above
	assert empty_board.__eq__(ret)

def test_string_to_board():

	#test case 1: empty board
	ret = string_to_board(empty_board)

	assert ret.dtype == BoardPiece
	assert isinstance(ret, np.ndarray)
	assert ret.shape == (6, 7)
	assert np.all(ret == NO_PLAYER)

	board = initialize_game_state()

	assert np.array_equal(ret, board)

	#test case 2: with one BoardPieces of player 1 placed on board
	board = initialize_game_state()
	board[0,0] = PLAYER1

	ret = string_to_board(one_piece_board)

	assert np.array_equal(ret, board)

def test_apply_player_action():

	action_board = initialize_game_state()
	ret = apply_player_action(action_board, 0, PLAYER1)

	board = string_to_board(one_piece_board)

	assert np.array_equal(board, ret)

def test_connected_four():
	#TODO: generalize (all possible wins)

	#test horizontal
	board = initialize_game_state()
	board[0, 0] = PLAYER1
	board[1, 0] = PLAYER1
	board[2, 0] = PLAYER1
	board[3, 0] = PLAYER1

	assert connected_four(board, PLAYER1)

	#test vertical
	board = initialize_game_state()
	board[0, 1] = PLAYER1
	board[0, 2] = PLAYER1
	board[0, 3] = PLAYER1
	board[0, 4] = PLAYER1

	assert connected_four(board, PLAYER1)

	#test diagonal positive
	board = initialize_game_state()
	board[0, 0] = PLAYER1
	board[1, 1] = PLAYER1
	board[2, 2] = PLAYER1
	board[3, 3] = PLAYER1

	assert connected_four(board, PLAYER1)

	#test diagonal negative
	board = initialize_game_state()
	board[0, 6] = PLAYER1
	board[1, 5] = PLAYER1
	board[2, 4] = PLAYER1
	board[3, 3] = PLAYER1

	assert connected_four(board, PLAYER1)

def test_check_board_full():

	ret = string_to_board(full_draw_board)
	ret_false = string_to_board(one_piece_board)

	assert check_board_full(ret)
	assert check_board_full(ret_false) == False

def test_check_end_state():

	# test case: win horizontal
	board = initialize_game_state()
	board[0, 0] = PLAYER1
	board[1, 0] = PLAYER1
	board[2, 0] = PLAYER1
	board[3, 0] = PLAYER1

	assert (check_end_state(board,PLAYER1) == GameState.IS_WIN)

	#test case: draw
	draw_board = string_to_board(full_draw_board)

	assert check_end_state(draw_board,PLAYER1) == GameState.IS_DRAW

	#test case: still playing
	playing_board = string_to_board(one_piece_board)

	assert check_end_state(playing_board,PLAYER1) == GameState.STILL_PLAYING


def test_check_open_columns():

	open_cols = string_to_board(still_playing_board)
	check_open_columns(open_cols)

	assert list(check_open_columns(open_cols)) == [1,3,4]