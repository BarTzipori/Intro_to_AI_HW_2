import math
import multiprocessing
import time

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, board_size
import random
from multiprocessing import Process


def min_distance(env: WarehouseEnv, robot, enemy):
    min_val = math.inf
    # the robot didn't pick a package - calculate the minimum distance from a package that is further from the enemy
    if robot.package is None:
        packages_on_board = [p for p in env.packages if p.on_board]
        min_val = board_size * board_size
        for p in packages_on_board:
            distance = manhattan_distance(robot.position, p.position)
            enemy_distance = manhattan_distance(enemy.position, p.position)
            if distance < min_val and distance <= enemy_distance:
                min_val = distance
                min_destination = manhattan_distance(p.position, p.destination)

    # the robot pick a package - calculate the distance from the destination
    else:
        min_val = manhattan_distance(robot.position, robot.package.destination)

    # if the robot is losing and can't get credit points, go to charge station
    if robot.credit > 0 and robot.battery == min_val and enemy.battery != 0:
        charges_list = [manhattan_distance(robot.position, c.position) for c in env.charge_stations]
        return min(charges_list)
    else:
        return min_val


def utility(env: WarehouseEnv, robot_id: int):
    # print("UTILITY GOES HERE")
    if env.done():
        robot = env.robots[robot_id]
        enemy_id = (robot_id + 1) % 2
        enemy = env.robots[enemy_id]
        if robot.credit > enemy.credit:
            return math.inf
        elif robot.credit < enemy.credit:
            return -math.inf
        return -1000
    return None


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.robots[robot_id]
    enemy_id = (robot_id + 1) % 2
    enemy = env.robots[enemy_id]

    # distance from the closest package/ package's destination (if picked)
    robot_distance = min_distance(env, robot, enemy)
    enemy_distance = min_distance(env, enemy, robot)

    # calculate the reward for a package (if picked)
    robot_reward = 0
    if not (robot.package is None):
        robot_reward = manhattan_distance(robot.package.position, robot.package.destination) * 2 + 100

    enemy_reward = 0
    if not (enemy.package is None):
        enemy_reward = manhattan_distance(enemy.package.position, enemy.package.destination) * 2 + 100
    # print("ROBOT CREDIT", robot.credit, "ENEMY CREDIT", enemy.credit, "ROBOT DISTANCE", robot_distance,
    # "ENEMY DISTANCE", enemy_distance)
    return (robot.credit * 1000 + robot_reward - robot_distance) - (enemy.credit * 1000 + enemy_reward - enemy_distance)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def check_time_limit(self, epsilon=5e-2):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit - epsilon:
            raise TimeoutError

    def minimax(self, env: WarehouseEnv, agent_id, depth, turn):
        self.check_time_limit()
        if env.done() or depth == 0:
            if env.done():
                return utility(env, agent_id), 'park'
            return smart_heuristic(env, agent_id), 'park'
        operators, children = self.successors(env, agent_id)

        # agent_id's turn - max problem
        if turn == agent_id:
            possible_moves_list = []
            # print("possible operators for agent", agent_id, operators)
            for operator, child in zip(operators, children):
                v, _ = self.minimax(child, agent_id, depth - 1, (turn + 1) % 2)
                possible_moves_list.append((operator, v))
            curr_max = max(v for _, v in possible_moves_list)
            filtered_data = [(operator, v) for operator, v in possible_moves_list if v == curr_max]
            # print("FILTERED DATA IS", filtered_data)

            # print("MAX SOLUTION", cur_op, cur_max, "agent id", agent_id, turn)
            return curr_max, random.choice(filtered_data)[0]

        # rival's turn - min problem
        else:
            # print("possible operators for agent", agent_id, operators)
            possible_moves_list = []
            for operator, child in zip(operators, children):
                v, _ = self.minimax(child, agent_id, depth - 1, (turn + 1) % 2)
                possible_moves_list.append((operator, v))
            curr_min = min(v for _, v in possible_moves_list)
            filtered_data = [(operator, v) for operator, v in possible_moves_list if v == curr_min]
            # print("MIN SOLUTION", curr_op, curr_min, "agent id", agent_id)
            return curr_min, random.choice(filtered_data)[0]

    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):

        self.start_time = time.time()
        self.time_limit = time_limit

        depth, operator = 0, 'park'
        while True:
            try:
                _, operator = self.minimax(env, agent_id, depth, 0)
                depth += 1
            except TimeoutError:
                return operator


class AgentAlphaBeta(Agent):
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def check_time_limit(self, epsilon=5e-2):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit - epsilon:
            raise TimeoutError

    def alphabetaminimax(self, env: WarehouseEnv, agent_id, l, alpha, beta, turn):
        self.check_time_limit()
        if env.done() or l == 0:
            if env.done():
                return utility(env, agent_id), 'park'
            return smart_heuristic(env, agent_id), 'park'

        operators, children = self.successors(env, agent_id)
        current_alpha = alpha
        current_beta = beta

        # agent_id's turn - max problem
        if turn == agent_id:
            cur_max = -1 * math.inf
            cur_op = ''
            for operator, new_env in zip(operators, children):
                v, _ = self.alphabetaminimax(new_env, agent_id, l - 1, current_alpha, current_beta,(turn + 1) % 2)
                if v > cur_max:
                    cur_max = v
                    cur_op = operator
                    current_alpha = max(cur_max, current_alpha)
                    if cur_max >= current_beta:
                        return math.inf, 'park'
            return cur_max, cur_op

        # rival's turn - min problem
        else:
            cur_min = math.inf
            cur_op = ''
            for operator, new_env in zip(operators, children):
                v, _ = self.alphabetaminimax(new_env, agent_id, l - 1, current_alpha, current_beta, (turn + 1) % 2)
                if v < cur_min:
                    cur_min = v
                    cur_op = operator
                    current_beta = min(cur_min, current_beta)
                    if cur_min <= current_alpha:
                        return -1 * math.inf, 'park'
            return cur_min, cur_op

    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):

        self.start_time = time.time()
        self.time_limit = time_limit

        depth, operator = 0, 'park'
        while True:
            try:
                _, operator = self.alphabetaminimax(env, agent_id, depth, -1 * math.inf, math.inf, 0)
                depth += 1
            except TimeoutError:
                return operator


def prob_choice(env: WarehouseEnv, robot_id):
    operators = env.get_legal_operators(robot_id)
    prob_operators = []
    for op in operators:
        if op == 'move north' or op == 'pick up':
            prob_operators.append(op)
        prob_operators.append(op)
    return random.choice(prob_operators)


def probability(env: WarehouseEnv, robot_id, operator):
    length = len(env.get_legal_operators(robot_id))
    if 'move north' in env.get_legal_operators(robot_id):
        length += 1
    if 'pick up' in env.get_legal_operators(robot_id):
        length += 1
    prob = 1.0 / length
    if operator == 'move north' or operator == 'pick up':
        return 2 * prob
    return prob


class AgentExpectimax(Agent):

    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def check_time_limit(self, epsilon=5e-2):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit - epsilon:
            raise TimeoutError

    def expectimax(self, env: WarehouseEnv, agent_id, l, turn):

        if env.done() or l == 0:
            return smart_heuristic(env, agent_id), ''
        operators, children = self.successors(env, agent_id)

        # agent_id's turn - max problem
        if turn == agent_id:
            cur_max = -1 * math.inf
            cur_op = ''
            for operator, new_env in zip(operators, children):
                v, _ = self.expectimax(new_env, agent_id, l - 1, (turn + 1) % 2)
                if v > cur_max:
                    cur_max = v
                    cur_op = operator
            return cur_max, cur_op

        # rival's turn - min problem
        else:
            expectation = 0
            for operator, new_env in zip(operators, children):
                v, _ = self.expectimax(new_env, agent_id, l - 1, (turn + 1) % 2)
                expectation += probability(env, turn, operator) * self.expectimax(env, agent_id, l - 1, (turn + 1) % 2)
            return expectation, prob_choice(env, turn)

    # TODO: section d : 1

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit

        depth, operator = 0, None

        while True:
            try:
                _, operator = self.expectimax(env, agent_id, depth, 0)
                depth += 1
            except TimeoutError:
                return operator

# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move east", "move north", "pick up", "move east"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)
        return random.choice(operators)
