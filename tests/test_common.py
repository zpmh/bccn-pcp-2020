from unittest import TestCase
import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2

class TestGame(TestCase):

	def test_initialize_game_state(self):
		from agents.common import initialize_game_state

		ret = initialize_game_state()

		assert isinstance(ret, np.ndarray)
		assert ret.dtype == BoardPiece
		# assert ret.shape == (6, 7)
		assert ret.shape == (6, 7)
		assert np.all(ret == NO_PLAYER)

	def test_pretty_print_board(self):
		'check is board can be shown as a string'
		from agents.common import pretty_print_board

		#ret = pretty_print_board()

# if isString(True):


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    #assert ret.shape == (6, 7)
    assert ret.shape == (6,7)
    assert np.all(ret == NO_PLAYER)

def test_pretty_print_board():
	'check is board can be shown as a string'
	#from agents.common import pretty_print_board

	#ret = pretty_print_board()

	#if isString(True):


def test_player_action():
	'check whether board is different after player action is applied, put piece at the lowest slot in that column'