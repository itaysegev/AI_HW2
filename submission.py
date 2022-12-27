import math
import random
import threading
import time

import numpy as np

import Gobblet_Gobblers_Env as gge

not_on_board = np.array([-1, -1])


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class RBMinMax(StoppableThread):
    def __init__(self, curr_state, agent_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.curr_state = curr_state
        self.best_action = None
        self.num_iter = 0

    def run(self) -> None:
        max_depth = 1
        neighbors = self.curr_state.get_neighbors()
        actions = [c[0] for c in neighbors]
        children = [c[1] for c in neighbors]
        self.best_action = actions[0]
        while not self.stopped():
            self.num_iter += 1
            values = [rb_min_max(child, self.agent_id, False, max_depth - 1) for child in children]
            best_value = max(values)
            self.best_action = actions[values.index(best_value)]
            max_depth += 1


class ExpectiMax(StoppableThread):
    def __init__(self, curr_state, agent_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.curr_state = curr_state
        self.best_action = None
        self.num_iter = 0

    def run(self) -> None:
        max_depth = 1
        neighbors = self.curr_state.get_neighbors()
        actions = [c[0] for c in neighbors]
        children = [c[1] for c in neighbors]
        self.best_action = actions[0]
        while not self.stopped():
            self.num_iter += 1
            values = [rb_expectimax(child, self.agent_id, False, max_depth - 1) for child in children]
            best_value = max(values)
            self.best_action = actions[values.index(best_value)]
            max_depth += 1


# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final is 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


# count the numbers of pawns that i have that aren't hidden
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns


# representation of the board and whose pieces are where (where there is blue and where there is orange)
def board_rep(state):
    coordinates = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    board = {}
    agent_id = 0
    for key, value in state.player1_pawns.items():
        if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
            board[(value[0][0], value[0][1])] = 0
            coordinates.remove((value[0][0], value[0][1]))
    agent_id = 1
    for key, value in state.player2_pawns.items():
        if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
            board[(value[0][0], value[0][1])] = 1
            coordinates.remove((value[0][0], value[0][1]))
    for c in coordinates:
        board[c] = None

    return board


def smart_heuristic(state, agent_id):
    opponent = 1 - agent_id
    h = 0
    board = board_rep(state)
    possible_wins = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
                     [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
                     [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]

    scores = [[0, -10, -100, -1000],
              [10, 0, 0, 0],
              [100, 0, 0, 0],
              [1000, 0, 0, 0]]

    for option in possible_wins:
        player = 0
        other = 0
        for i in range(3):
            piece = board[option[i]]  # will be 0 if is player1's piece, 1 if is player2's piece
            if piece == agent_id:
                player += 1
            elif piece == opponent:
                other += 1
        h += scores[player][other]
    return h


# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
def greedy(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


# TODO - add your code here
def greedy_improved(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = -2000
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        # curr_heuristic = super_heuristic(neighbor[1],agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    rb_minmax = RBMinMax(curr_state, agent_id)
    rb_minmax.start()
    rb_minmax.join(timeout=time_limit - 0.2)
    rb_minmax.stop()

    print(f"Number of iter: {rb_minmax.num_iter}")
    return rb_minmax.best_action


def rb_min_max(curr_state, agent_id, our_turn, depth):
        if gge.is_final_state(curr_state) is not None or depth == 0:
            return smart_heuristic(curr_state, agent_id)
        # Turn <- Turn(State)
        agent_to_play = agent_id if our_turn else ((agent_id + 1) % 2)
        # Children <- Succ(State)
        children = curr_state.get_neighbors()

        if our_turn:
            curr_max = -math.inf
            for _, c in children:
                v = rb_min_max(c, agent_to_play, False, depth - 1)
                curr_max = max(v, curr_max)
            return curr_max
        else:
            curr_min = math.inf
            for _, c in children:
                v = rb_min_max(c, agent_to_play, True, depth - 1)
                curr_min = min(v, curr_min)
            return curr_min


def alpha_beta(curr_state, agent_id, time_limit):
    raise NotImplementedError()


def expectimax(curr_state, agent_id, time_limit):
    expectimax = ExpectiMax(curr_state, agent_id)
    expectimax.start()
    expectimax.join(timeout=time_limit - 0.2)
    expectimax.stop()

    print(f"Number of iter: {expectimax.num_iter}")
    return expectimax.best_action


def rb_expectimax(curr_state, agent_id, our_turn, depth):
    if gge.is_final_state(curr_state) is not None or depth == 0:
        return smart_heuristic(curr_state, agent_id)
    # Turn <- Turn(State)
    agent_to_play = agent_id if our_turn else ((agent_id + 1) % 2)
    # Children <- Succ(State)
    children = curr_state.get_neighbors()
    if our_turn:
        curr_max = -math.inf
        for _, c in children:
            v = rb_expectimax(c, agent_to_play, False, depth - 1)
            curr_max = max(v, curr_max)
        return curr_max
    else:
        values = []
        u_val = 0
        opponent_id = (agent_id + 1) % 2
        # count the numbers of pawns that i have that aren't hidden
        curr_opponent_not_hidden_pawns = dumb_heuristic2(curr_state, opponent_id)
        for action, c in children:
            double_flag = False
            new_opponent_not_hidden_pawns = dumb_heuristic2(c, opponent_id)
            if new_opponent_not_hidden_pawns < curr_opponent_not_hidden_pawns:
                double_flag = True
                u_val += 2
            elif action[0][0] == 'S':
                double_flag = True
                u_val += 2
            else:
                u_val += 1
            values.append((rb_expectimax(c, agent_to_play, True, depth - 1),double_flag))
        v = 0
        p_val = 1/u_val
        for val, curr_flag in values:
            p = 2 * p_val * val if curr_flag else p_val * val
            v += p
        return v


# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = -2000
    max_neighbor = None
    for neighbor in neighbor_list:
        # curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        curr_heuristic = super_heuristic(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


def super_heuristic(state, agent_id):
    return 1000 * is_we_won(state,agent_id) + 25 * is_B_on_M(state, agent_id) +\
           2 * is_my_pawn_near_opponent(state, agent_id) + 10 * close_triple(state,agent_id) + \
           10 * is_both_B_on_board(state, agent_id) + 5 * is_B_on_middle(state, agent_id) +\
           is_both_M_on_board(state, agent_id) + is_M_on_board(state, agent_id) - \
           100 * opponent_one_step_from_win(state, agent_id)


def is_B_on_middle(state, agent_id):
    return gge.cor_to_num(gge.find_curr_location(state, 'B1', agent_id)) == 4 or \
           gge.cor_to_num(gge.find_curr_location(state, 'B2', agent_id)) == 4


def is_we_won(state, agent_id):
    return dumb_heuristic1(state, agent_id)


def is_both_B_on_board(state, agent_id):
    return (gge.cor_to_num(gge.find_curr_location(state, 'B1', agent_id)) > -1) and \
           (gge.cor_to_num(gge.find_curr_location(state, 'B2', agent_id)) > -1)


def is_M_on_board(state, agent_id):
    return (gge.cor_to_num(gge.find_curr_location(state, 'M1', agent_id)) > -1) or \
           (gge.cor_to_num(gge.find_curr_location(state, 'M2', agent_id)) > -1)


def is_both_M_on_board(state, agent_id):
    return (gge.cor_to_num(gge.find_curr_location(state, 'M1', agent_id)) > -1) and \
           (gge.cor_to_num(gge.find_curr_location(state, 'M2', agent_id)) > -1)


def is_B_on_M(state, agent_id):
    m_sum = 0
    opponent_id = 1 - agent_id
    agent_pawns_dict = {0: state.player1_pawns, 1: state.player2_pawns}
    pawn1_loc_row = agent_pawns_dict[opponent_id]['M1'][0][0]
    pawn2_loc_row = agent_pawns_dict[opponent_id]['M2'][0][0]
    if pawn1_loc_row > -1:
        m_sum += is_hidden(state, opponent_id, 'M1')
    if pawn2_loc_row > -1:
        m_sum += is_hidden(state, opponent_id, 'M2')
    return m_sum


def is_pawn_neighbors(state, pawn1, pawn2, agent_id1, agent_id2):
    if agent_id1 == agent_id2 and pawn1 == pawn2:
        return False
    agent_pawns_dict = {0:state.player1_pawns, 1:state.player2_pawns}
    pawn1_loc_row = agent_pawns_dict[agent_id1][pawn1][0][0]
    pawn1_loc_col = agent_pawns_dict[agent_id1][pawn1][0][1]
    pawn2_loc_row = agent_pawns_dict[agent_id2][pawn2][0][0]
    pawn2_loc_col = agent_pawns_dict[agent_id2][pawn2][0][1]
    if -1 in [pawn1_loc_row,pawn1_loc_col,pawn2_loc_row,pawn2_loc_col]:
        return False

    return abs(pawn1_loc_row - pawn2_loc_row) <= 1 and\
            abs(pawn1_loc_col - pawn2_loc_col) <= 1


def is_my_pawn_near_opponent(state, agent_id):
    sum_pawns = 0
    for pawn1 in state.player1_pawns.keys():
        for pawn2 in state.player2_pawns.keys():
            if is_pawn_neighbors(state, pawn1, pawn2, agent_id, 1 - agent_id) and\
                    not is_hidden(state, agent_id, pawn1) \
                    and not is_hidden(state, 1 - agent_id, pawn2):
                sum_pawns += 1
    return sum_pawns


def close_triple(state, agent_id):
    for pawn1 in state.player1_pawns.keys():
        num_of_neighbors = 0
        for pawn2 in state.player1_pawns.keys():
            num_of_neighbors += is_pawn_neighbors(state, pawn1, pawn2, agent_id, agent_id)
        if num_of_neighbors == 2:
            return True
    return False


def opponent_one_step_from_win(state, agent_id):
    opponent_id = str(1 -agent_id + 1)
    arr = gge.pawn_list_to_marks_array(state)
    one_step_from_win = None
    # check rows
    for i in range(3):
        if (arr[i][0] == arr[i][1] and arr[i][2] == " " and arr[i][0] == opponent_id) or (arr[i][1] == arr[i][2] and
                                                                                      arr[i][0] == " " and
                                                                                      arr[i][1] == opponent_id) or \
                (arr[i][0] == arr[i][2] and arr[i][1] == " " and arr[i][0] == opponent_id):
            return 1

    # check columns
    for j in range(3):
        if (arr[0][j] == arr[1][j] and arr[2][j] == " " and arr[0][j] == opponent_id) or (arr[1][j] == arr[2][j] and
                                                                                      arr[0][j] == " " and
                                                                                      arr[1][j] == opponent_id) or \
                (arr[0][j] == arr[2][j] and arr[1][j] == " " and arr[0][j] == opponent_id):
            return 1

    # check obliques
    if (arr[0][0] == arr[1][1] and arr[2][2] == " " and arr[0][0] == opponent_id) or (arr[1][1] == arr[2][2] and
                                                                                  arr[0][0] == " " and
                                                                                  arr[1][1] == opponent_id) or \
            (arr[0][0] == arr[2][2] and arr[1][1] == " " and arr[0][0] == opponent_id):
        return 1

    if (arr[0][2] == arr[1][1] and arr[2][0] == " " and arr[1][1] == opponent_id) or (arr[0][2] == arr[2][0] and
                                                                                  arr[1][1] == " " and
                                                                                  arr[2][0] == opponent_id) or \
            (arr[1][1] == arr[2][0] and arr[0][2] == " " and arr[2][0] == opponent_id):
        return 1
    return 0





