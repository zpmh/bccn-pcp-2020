disable_jit = True
if disable_jit:
	import os
	os.environ['NUMBA_DISABLE_JIT'] = '1'

import timeit
import numpy as np
from agents.common import connected_four, connected_four_convolve, connected_four_iter, initialize_game_state, BoardPiece, PlayerAction, NO_PLAYER, CONNECT_N

"""performance evaluation of connected 4 functions"""

board = initialize_game_state()

number = 10**4

res = timeit.timeit("connected_four_iter(board, player)",
					setup="connected_four_iter(board, player)",
					number=number,
					globals=dict(connected_four_iter=connected_four_iter,
								 board=board,
								 player=BoardPiece(1)))

print(f"Python iteration-based: {res/number*1e6 : .1f} us per call")

res = timeit.timeit("connected_four_convolve(board, player)",
					number=number,
					globals=dict(connected_four_convolve=connected_four_convolve,
								 board=board,
								 player=BoardPiece(1)))

print(f"Convolve2d-based: {res/number*1e6 : .1f} us per call")

res = timeit.timeit("connected_four(board, player)",
					setup="connected_four(board, player)",
					number=number,
					globals=dict(connected_four=connected_four,
								 board=board,
								 player=BoardPiece(1)))

print(f"My secret sauce: {res/number*1e6 : .1f} us per call")
