from agents.agent_mcts.agent_mcts import Node, MCTS, generate_move
from agents.common import *
from time import time
from copy import deepcopy

# CHECK NODE FUNCTIONS

def test_node():

	# check implementation of root Node
	board = initialize_game_state()
	root_node = Node(board, parent=None, col=-1, player=PLAYER1)

	assert root_node.unexpanded_moves == check_open_columns(board)
	assert np.array_equal(board, root_node.board)
	assert root_node.parent == None
	assert root_node.value == 0
	assert root_node.num_visits == 0
	assert root_node.is_root == True
	assert root_node.player == PLAYER1
	assert root_node.column_move == -1

	# check non-root Node
	non_root_node = Node(board, parent=root_node, col=-1, player=PLAYER2)

	assert non_root_node.unexpanded_moves == check_open_columns(board)
	assert np.array_equal(board, non_root_node.board)
	assert non_root_node.parent == root_node
	assert non_root_node.value == 0
	assert non_root_node.num_visits == 0
	assert non_root_node.is_root == False
	assert non_root_node.player == PLAYER2
	assert non_root_node.column_move == -1

def test_ucb_value():

	board = initialize_game_state()
	root_node = Node(board, parent=None, col=-1, player=PLAYER1)

	assert root_node.num_visits == 0
	# return 0 for no visit
	assert root_node.ucb_value() == np.inf

	# return correct value for number of visits
	root_node.num_visits = 1
	arb_node = Node(board, parent=root_node, col=-1, player=PLAYER1)
	for i in np.arange(1,5,1):
		arb_node.num_visits = i
		correct_res = (arb_node.value / arb_node.num_visits) + \
		              (np.sqrt(2) * (np.sqrt(np.abs(np.log(arb_node.parent.num_visits) / arb_node.num_visits))))
		# returns nan when parent node has 0 visits
		assert Node.ucb_value(arb_node) == correct_res

def test_expansion():
	move = 3
	board = initialize_game_state()
	root_node = Node(board, parent=None, col=-1, player=PLAYER1)
	#root_node.expansion(move, board, PLAYER1)

	# children are stored in a list format, expansion just returns the node (not in a list format)
	assert root_node.children == [root_node.expansion(move, board, PLAYER1)] # returns and appends child correctly
	assert root_node.unexpanded_moves == [0,1,2,4,5,6] # removed correct move

# CHECK MCTS FUNCTIONS

# test MCTS objects
player1 = MCTS(PLAYER1)
player2 = MCTS(PLAYER2)
# test board and NODE object
board = initialize_game_state()
root_node = Node(board, parent=None, col=-1, player=PLAYER1)

def test_mcts():
	assert player1.player == PLAYER1
	assert np.round(player2.start_time,0) == np.round(time(),0) # 3rd decimal place may differ
	assert player2.player == PLAYER2

def test_backpropagation():
	# return nothing for root_node
	assert player1.backpropagation(root_node, 0) == None

	# create some leaf nodes and check if they're backpropagated correctly
	first_node = Node(board, parent=root_node, col=3, player=PLAYER1)
	second_node = Node(board, parent=first_node, col=4, player=PLAYER1)
	leaf_node = Node(board, parent=second_node, col=2, player=PLAYER1)

	#call backpropagate
	player1.backpropagation(leaf_node, 1)
	assert first_node.num_visits == 1
	assert first_node.value == 1

def test_check_time():
	player1 = MCTS(PLAYER1)
	# check that check_time() returns True
	assert player1.check_time(5) == True
	# check that if time elapsed is larger than 5 returns false
	player1.start_time = MCTS(PLAYER1).start_time - 5
	assert player1.check_time(5) == False

def test_highest_ucb():
	board = initialize_game_state()
	current_node = Node(board, parent=None, col=-1, player=PLAYER1)

	# check if  children are 0 returns None
	assert len(current_node.children) == 0
	assert player1.highest_ucb(current_node) == None

	# add children, define max ucb val, check if its returned
	child_one = Node(board, parent=current_node, col=0, player=PLAYER1)
	child_two = Node(board, parent=current_node, col=1, player=PLAYER1)
	child_three = Node(board, parent=current_node, col=2, player=PLAYER1)
	child_one.num_visits = 1
	child_one.value = 9
	child_two.num_visits = 15
	child_two.value = 3
	child_three.num_visits = 2
	child_three.value = 20

	current_node.children.append(child_one)
	current_node.children.append(child_two)
	current_node.children.append(child_three)

	assert len(current_node.children) == 3
	assert player1.highest_ucb(current_node) == child_one

def test_result():

	mcts = MCTS(PLAYER1)

	from tests.test_common import full_draw_board, still_playing_board
	draw_board = string_to_board(full_draw_board)
	win_board = apply_player_action(deepcopy(board), 0, PLAYER1)
	win_board = apply_player_action(win_board, 1, PLAYER1)
	win_board = apply_player_action(win_board, 2, PLAYER1)
	win_board = apply_player_action(win_board, 3, PLAYER1)

	assert mcts.result(draw_board, PLAYER2) == 0.2
	assert mcts.result(board, PLAYER1) == 0
	assert mcts.result(win_board, PLAYER1) == 1
	assert mcts.result(win_board, PLAYER2) == -1

def test_select_random_child():
	mcts = MCTS(PLAYER1)

	root_node = Node(board, parent=None, col=-1, player=PLAYER1)
	child_one = Node(board, parent=root_node, col=0, player=PLAYER1)
	child_two = Node(board, parent=root_node, col=1, player=PLAYER1)
	child_three = Node(board, parent=root_node, col=2, player=PLAYER1)
	root_node.children.append(child_one)
	root_node.children.append(child_two)
	root_node.children.append(child_three)

	random_node = mcts.select_random_child(root_node.children)

	# check that returns (one) child
	assert random_node == child_one or random_node == child_two or random_node == child_three

def test_monte_carlo_tree_search():

	# monte_carlo_tree_search() depends on multiple other functions in MCTS:
	# selection(), simulation(), backpropagation(), best_child()
	# if monte_carlo_tree_search behaves as expected so should the functions it depends on

	# create a board where PLAYER1 is close to winning
	win_board = apply_player_action(deepcopy(board), 0, PLAYER1)
	win_board = apply_player_action(win_board, 1, PLAYER1)
	win_board = apply_player_action(win_board, 2, PLAYER1)

	# create root Node object
	root = Node(board_copy=deepcopy(win_board), parent=None, col=-1, player=PLAYER1)
	# create MCTS object for player
	mcts1 = MCTS(PLAYER1)  # to start the time
	# call monte carlo tree search starting from root node
	action = mcts1.monte_carlo_tree_search(root)

	assert action == 3 # mcts should choose an immediate win for PLAYER1

	mcts2 = MCTS(PLAYER2)
	action = mcts2.monte_carlo_tree_search(root)

	assert action == 3 # mcts should choose an immediate block for PLAYER2


def test_generate_move():
	# test that generate move plays in center on empty board
	assert generate_move(board, PLAYER1, False) == (3, False)