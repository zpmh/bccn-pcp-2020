import random
from copy import deepcopy
import numpy as np
from time import time
from typing import Optional, Tuple, List

from agents.common import check_board_full, check_open_columns, apply_player_action, check_end_state, connected_four
from agents.common import PLAYER1, PLAYER2, GameState, BoardPiece, SavedState, NO_PLAYER, PlayerAction

# Typical Python style is to put related classes in the same module. (no consensus - from stack overflow)
# https://stackoverflow.com/questions/2098088/should-i-create-each-class-in-its-own-py-file

# make sure player is assigned in generate_move
PLAYER = NO_PLAYER
OPPONENT = NO_PLAYER

def generate_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState])\
        -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    generates an optimal move/action using the Monte Carlo Tree Search strategy
    :param board: current state of board
    :param player: player whose move is optimized
    :param saved_state: saved state of board
    :return: move, saved_state (optional)
    """

    # turn on RuntimeWarning
    np.seterr(divide='warn')

    global PLAYER
    global OPPONENT

    # set player and opponent
    PLAYER = player
    OPPONENT = PLAYER1 if player == PLAYER2 else PLAYER2

    # if lowest board row is empty make first move in central col = 3
    if not board[0,:].any():
        action = 3

    else:
        # create root Node object
        root = Node(board_copy=deepcopy(board), parent=None, col=-1, player=PLAYER)
        # create MCTS object for player
        mcts = MCTS(PLAYER) #to start the time
        # call monte carlo tree search starting from root node
        action = mcts.monte_carlo_tree_search(root)

    # return optimal action for player
    return PlayerAction(action), saved_state


class Node:
    def __init__(self, board_copy: np.ndarray, parent: object, col: int, player: BoardPiece) -> object:
        self.board = deepcopy(board_copy)
        self.parent = parent
        self.column_move = col  # node belongs to move in this column
        self.player = player
        self.is_root = parent is None
        self.children = []
        self.value = 0
        self.num_visits = 0
        self.unexpanded_moves = check_open_columns(board_copy) # all possible moves for current board

    def expansion(self, move: int, state: np.ndarray, player: BoardPiece) -> object:
        """
        expands list of children of given node and removes child from unexpanded moves
        :param move: column move
        :param state: current board
        :param player: player
        :return: child Node
        """
        child = Node(board_copy=state, parent=self, col=move, player=player)
        self.unexpanded_moves.remove(move)
        self.children.append(child)
        return child

    def ucb_value(self) -> float:
        """
        strategy ucb1
        :return: upper confidence bound for selections & expansion of given node
        """
        if self.num_visits == 0:
            return np.inf
        np.seterr(divide='ignore') # turn off RuntimeWarning for possible division by 0
        return self.value/self.num_visits + np.sqrt(2) * np.sqrt(np.log(self.parent.num_visits) / self.num_visits)

class MCTS:
    def __init__(self, player: BoardPiece) -> object:
        self.player = player
        self.start_time = time()  # set a time limit for exploration

    def backpropagation(self, node: Node, simulation_result: int):
        """
        backpropagates value and number of vists
        :param node: leaf node
        :param result: game simulation result
        :return: recursive call until root node is hit (returns void/nothing)
        """
        # stop recursive call at root node
        if node.is_root:
            return
        # update node's value and number of visits
        node.value += simulation_result # value counts wins/losses
        node.num_visits += 1
        # recursive call to backpropagate result
        self.backpropagation(node.parent, simulation_result)

    def best_child(self, root: Node) -> Node:
        """
        finds the best (optimal) next move
        :param root: current game state/ node
        :return: optimal next node to make root node
        """
        # set the best ratio the a (large) negative number
        best_ratio = -np.infty
        # declare best action and urgent block
        best_action = None
        urgent_block = None
        # loop through child nodes
        for child in root.children:
            # create board for opponent move in child column
            opponent_board = apply_player_action(deepcopy(root.board), child.column_move, OPPONENT)
            # always return immediate wins
            if connected_four(child.board, child.player):
                return child
            # block immediate loss (if you don't play position and opponent can win by playing there next)
            elif connected_four(opponent_board, OPPONENT):
                urgent_block = child # you can only block one position at a time anyway
            # find child with highest value/visits ratio
            else:
                ratio = child.value / child.num_visits
                if ratio > best_ratio:
                    best_action = child
                    best_ratio = ratio
        if urgent_block != None:  # check blocks in the end to prefer wins over blocks
            return urgent_block
        else:
            return best_action

    def check_time(self, time_limit: int) -> bool:
        """
        :return: True if runtime still within time limit, False otherwise
        """
        return (time() - self.start_time) < time_limit

    def highest_ucb(self, node: Node) -> Node:
        """
        returns node (out of children) with the highest ucb value
        :param node: root node
        :return: child with highest ucb
        """
        # don't return anything for terminal/leaf node
        if len(node.children) == 0:
            return None
        # select child node with the max ucb value
        return max(node.children, key=Node.ucb_value)

    def monte_carlo_tree_search(self, root: Node) -> int:
        """
        returns column value of optimal move
        :param root: root Node
        :return: column that is the optimal move
        """
        root.num_visits += 1  # root node isn't 0, it's visited first to get the leaf node (otherwise I get nan values)
        while self.check_time(5):
            # selection and expansion
            node = self.selection(root, deepcopy(root.board), self.player)
            # simulate games
            simulation_score = self.simulation(node)
            # backpropagation scores (update value for each visited node)
            self.backpropagation(node, simulation_score)
        # now choose the best action (based on the ratio of node value and visits)
        chosen_node = self.best_child(root)
        return chosen_node.column_move

    def result(self, board: np.ndarray, player: BoardPiece) -> int:
        """
        returns value for the simulation result of the game for player
        :param board: current state of board
        :param player: agent
        :return: result for agent (won/lost/draw/still_playing) expressed in an int
        """
        OPPONENT = PLAYER1 if player==PLAYER2 else PLAYER2
        if check_end_state(board, OPPONENT) == GameState.IS_WIN:
            return -1
        elif check_end_state(board, player) == GameState.IS_WIN:
            return 1
        elif check_end_state(board, player) == GameState.IS_DRAW:
            return 0.2 # worked well (adjusted by playing many games)
        else:
            return 0 # for still playing

    def selection(self, node: Node, root_board: np.ndarray, player: BoardPiece) -> Node:
        """
        selects child node to expand and calls expansion
        :param node: node thats expanded
        :param root_board: copy of current state of the board
        :param player: player
        :return: expanded node
        """
        while node.children != [] and node.unexpanded_moves == []:
            # select best child for expansion
            node = self.highest_ucb(node)

        # unless we've already expanded all children, add new child node with best ucb
        if node.unexpanded_moves != []:
            # pick unexpanded child of node with best ucb
            move = self.select_random_child(node.unexpanded_moves)
            # create board for child
            child_board = apply_player_action(deepcopy(root_board), move, self.player)
            # add child
            node = node.expansion(move=move, state=child_board, player=player)
        return node

    def select_random_child(self, children: List) -> int:
        """
        returns random column value
        :param children: unexpanded moves (children)
        :return: random child's column value
        """
        return children[random.choice(range(len(children)))]

    def simulation(self, node: Node) -> int:
        """
        simulates game until board is full or either player won
        :param node: start node
        :return: result of the game simulation
        """
        simulation_board = deepcopy(node.board)
        player = original_player = node.player

        while not check_board_full(simulation_board) and len(check_open_columns(simulation_board)) > 0:
            avail_moves = check_open_columns(simulation_board)
            # switch between players
            player = PLAYER2 if player == PLAYER1 else PLAYER1  # opposite player makes a move first
            # simulate
            simulation_board = apply_player_action(simulation_board, avail_moves[random.choice(range(len(avail_moves)))],
                                   player=player)
            # early stopping in case a player won
            if connected_four(simulation_board, player):
                break
        # evaluate end state of the game after simulation for the original player
        return self.result(simulation_board, original_player)