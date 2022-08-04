import networkx as nx


class Grid:
    def __init__(self, grid, agents, targets):
        self.grid = grid
        self.agents = agents  # containing all nodes preforming as agents
        self.targets = targets  # containing all nodes preforming as targets
        self.board_number = 1  # for printing use

    def print_board(self, h, goal_board_number):
        if self.board_number == 1:
            print('Board ' + str(self.board_number) + ' (starting position):')  # printing starting title
            self.board_number += 1
        elif self.board_number == goal_board_number and goal_board_number > 1:
            print('Board ' + str(self.board_number) + ' (goal position):')  # printing goal title
        else:
            print('Board ' + str(self.board_number) + ':')  # printing title
            self.board_number += 1
        title = '   '
        for k in range(len(self.grid)):  # creating the numbers above the board
            title += (' ' + str(k + 1))
        for i in range(len(self.grid)):  # responsible on printing the actual board
            line = ''
            if i == 0:
                print(title)
            for j in range(len(self.grid)):  # responsible on printing the actual board
                t = ''
                if self.get_grid()[i][j].get_state() == 1:
                    t = '@ '
                if self.get_grid()[i][j].get_state() == 2:
                    t = '* '
                if self.get_grid()[i][j].get_state() == 0:
                    t = '  '
                line += t
            print(i + 1, ':', line)
        if self.board_number == 3 and h > 0:  # in case bool is set to true> show heuristics
            print('Heuristic : ' + str(h))
        print('_____')

    def get_grid(self):
        return self.grid

    def get_agents(self):
        return self.agents

    def get_targets(self):
        return self.targets


class Node:
    def __init__(self, x, y, my_state, total_row):

        self.x = x
        self.y = y
        self.my_state = my_state
        self.total_row = total_row
        self.f = 0
        self.g = 0
        self.h = 0
        self.neighbors = []
        self.previous = None

    def update_neighbors(self, grid, targets):  # method is responsible of linking node to each other
        if self.x == len(grid) and self.y == len(grid):  # meaning were dealing with exit node
            # (exit node are generated in case there are more agents than targets.. meaning some of the agents
            # would have to use the exit node to leave the board)
          pass
        else:
            if (self.x < len(grid) - 1) and ((grid[self.x + 1][self.y].get_state() == 0) or
                                             (grid[self.x + 1][self.y].get_state() == 2)):  # down
                self.neighbors.append(grid[self.x + 1][self.y])

            if (self.x > 0) and ((grid[self.x - 1][self.y].get_state() == 0) or
                                 (grid[self.x - 1][self.y].get_state() == 2)):  # up
                self.neighbors.append(grid[self.x - 1][self.y])

            if (self.y < len(grid) - 1) and ((grid[self.x][self.y + 1].get_state() == 0) or
                                             (grid[self.x][self.y + 1].get_state() == 2)):  # right
                self.neighbors.append(grid[self.x][self.y + 1])

            if (self.y > 0) and ((grid[self.x][self.y - 1].get_state() == 0) or
                                 (grid[self.x][self.y - 1].get_state() == 2)):  # left
                self.neighbors.append(grid[self.x][self.y - 1])

            if self.x == len(grid) - 1:  # if the node were looking at is in the last row
                for j in range(len(targets)):  # for each target
                    if targets[j].get_x() == len(grid):  # if the target is a exit node (exit node.get_x is set to the
                        # length of the board)
                        self.neighbors.append(targets[j])  # connect this node to the exit node

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_state(self):
        return self.my_state

    def set_state(self, new_state):
        self.my_state = new_state

    def get_neighbors(self):
        return self.neighbors

    def get_total_row(self):
        return self.total_row

    def reset_previous(self):
        self.previous = None


def get_neighbors(grid, targets):  # method is responsible of iterating the grid and updating
    # each node possible neighbors
    for target in targets:
        if target.get_x() == len(grid):  # in case one of our targets is a exit node, its row and col would be set to
            # the length of the board (to identify them)
            target.update_neighbors(grid, targets)  # here were calling update neighbors inorder to connect the
            # exit node to the last row thus making them neighbors

    for i in range(len(grid)):
        for j in range(len(grid)):
            grid[i][j].update_neighbors(grid, targets)

    pass


def agent_counter(board):  # method counts the number of agents/targets for board checking
    agents_counter = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if isinstance(board[i][j], Node):
                if board[i][j].get_state() == 2:
                    agents_counter += 1
            else:
                if board[i][j] == 2:
                    agents_counter += 1

    return agents_counter


def board_is_legit(node_grid, goal_board):  # method is responsible of making sure that the forcefields are in the
    # same place and that the number of targets is less than the number of agents
    grid = node_grid
    start_agents_counter = agent_counter(node_grid)
    goal_agents_counter = agent_counter(goal_board)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (grid[i][j].get_state() == 1 and goal_board[i][j] != 1) or (
                    grid[i][j].get_state() != 1 and goal_board[i][j] == 1):
                # making sure that the forcefields are in the same place -
                # if the locations are not identical >> meaning the goal board isn't valid
                print("the forcefields spots between the two boards are not identical")
                return False

    return start_agents_counter >= goal_agents_counter


def make_grid(start_board, goal_board):  # method is responsible of generating Grid object
    grid = []
    agents = []
    targets = []
    for i in range(len(start_board)):
        grid.append([])
        for j in range(len(start_board)):
            spot_initial_value = start_board[i][j]  # getting values from starting board
            node = Node(i, j, spot_initial_value, len(start_board))  # creating nodes
            grid[i].append(node)  # filling the grid (2D array) in nodes
            if spot_initial_value == 2:
                agents.append(node)  # if the initial value at the stating board is 2 add the node into agents list
            if goal_board[i][j] == 2:
                targets.append(node)  # if the initial value at the goal board is 2 add the node into targets list
    exit_node = Node(len(start_board), len(start_board), 0, len(start_board))  # creating exit node in case of more
    # agents than targets
    for k in range(len(agents) - len(targets)):
        targets.append(exit_node)  # the diff between the two lists is the number of extra targets we need to add
        # into the targets list

    output = Grid(grid, agents, targets)  # finally > creating a grid object
    return output


def find_path(starting_board, goal_board, search_method, detail_output):
    node_grid = make_grid(starting_board, goal_board)  # creating grid object
    get_neighbors(node_grid.get_grid(), node_grid.get_targets())  # updates all nodes neighbors
    if board_is_legit(node_grid.get_grid(), goal_board):  # call for a simple board check
        if search_method == 1:
            multi_agent_a_star_handler(node_grid, detail_output)  # call for a solve

    else:
        print("No path found.")


class AStar:

    @staticmethod
    def clean_open_set(open_set, current_node):  # method is responsible of popping nodes out of the open-set as part of
        # A-star algorithm
        for i in range(len(open_set)):
            if open_set[i] == current_node:
                open_set.pop(i)
                break

        return open_set

    @staticmethod
    def h_score(current_node, end):  # calculating manhattan distance between two nodes
        if end.get_x() == current_node.get_total_row() and end.get_y() == current_node.get_total_row():  # if the end
            #  node were currently looking at is in fact a exit node don't mind the y coordinate when calculating the
            #  distance
            distance = abs(current_node.x - end.x)
        else:
            distance = abs(current_node.x - end.x) + abs(current_node.y - end.y)

        return distance

    @staticmethod
    def start_path(current_node, end):  # the actual a* algorithm - takes in 2 nodes.
        final_path = []
        current_node.reset_previous()
        if current_node.get_x() == end.get_x() and current_node.get_y() == end.get_y():  # if we already have a agent on
            # a target
            final_path.append(current_node)  # add the node into final path ( a list of nodes that symbolizes the path)
        else:
            open_set = [current_node]  # add the start node into the open set
            closed_set = []
            current_node.g = 0  # reset the start node g cost to 0
            while len(open_set) != 0:  # as long as the open set is not empty
                best_way = 0
                for i in range(len(open_set)):  # the power of the a-star algorithm is this line below
                    if open_set[i].f < open_set[best_way].f:  # best way gets the node in open set with the lowest
                        # f value (f= h+g) instead of using priority queue
                        best_way = i
                current_node = open_set[best_way]  # current node gets to be the node with the lowest f cost
                if current_node.get_x() == end.get_x() and current_node.get_y() == end.get_y():  # if the current node
                    # were looking at is actually the end node
                    temp = current_node
                    final_path.append(temp)  # add temp node into the list called final path
                    while temp.previous:  # while there is a node that led to the temp node
                        final_path.insert(0, temp.previous)  # insert the previous node into the list
                        temp = temp.previous
                    pass  # when the thread gets to this point is "breaks" the  "while len(open_set) != 0:" and
                    # final path gets returned
                open_set = AStar.clean_open_set(open_set, current_node)  # pop current node from open set
                closed_set.append(current_node)  # and append it to close set
                neighbors = current_node.neighbors  # gets the list of neighbor nodes
                for neighbor in neighbors:  # for each of the neighbors in that list
                    if neighbor in closed_set:  # if its already been evaluated continue
                        continue
                    else:
                        temp_g = current_node.g + 1  # the cost of another step
                        control_flag = 0  # reset control flag to 0
                        for k in range(len(open_set)):
                            if neighbor.x == open_set[k].x and neighbor.y == open_set[k].y:  # if the neighbor that were
                                # looking at is already in the open set
                                if temp_g < open_set[k].g:  # if the current temp_g is lower than the g of
                                    # that neighbor (meaning we found a chipper way) then update that node g h and
                                    # f values
                                    open_set[k].g = temp_g
                                    open_set[k].h = AStar.h_score(open_set[k], end)
                                    open_set[k].f = open_set[k].g + open_set[k].h
                                    open_set[k].previous = current_node  # update the previous node to the current_node
                                    # in order to reverse engineer the optimal path
                                else:  # if the neighbor that were looking at is already in the open set but the temp
                                    # value is not cheaper there is no need to update it >> control flag = 1 and we wont
                                    # enter the else that adds the node into the open set (its already there and the
                                    # values don't need to get updated)
                                    pass
                                control_flag = 1
                        if control_flag == 1:  # normally the flag is set to zero
                            pass
                        else:                # and we update the neighbor g h f and previous values
                            neighbor.g = temp_g
                            neighbor.h = AStar.h_score(neighbor, end)
                            neighbor.f = neighbor.g + neighbor.h
                            neighbor.previous = current_node
                            open_set.append(neighbor)  # and add the node to the open set
        return final_path  # finally return the optimal path between the start node and the end node


def multi_agent_a_star_handler(node_grid, detail_output):
    g = nx.Graph()  # creating a graph object from the networkx library > the aim is to create a bipartite graph
    # where one side represent the agents and the other the targets
    agents = node_grid.get_agents()
    targets = node_grid.get_targets()
    shortest_paths = [[[]] * len(targets) for i in range(len(agents))]  # creating a matrix where
    # shortest_paths[i][j] = shortest path from agent i to target j
    for i in range(len(agents)):  # this for loop is responsible of adding enough nodes (networkx type nodes)
        # representing agents and targets
        g.add_node(i)                 # agent node (numbered: 0..1..2..3..4)
        g.add_node(len(agents) + i)  # target node (numbered: 5..6..7..8..9)
    for i in range(len(agents)):
        for j in range(len(targets)):
            shortest_paths[i][j] = AStar.start_path(agents[i], targets[j])  # calling the start path function in A-star
            # object that output's the optimal path from agent i to target j
            if len(shortest_paths[i][j]) > 0:  # meaning there is actually a path from that agent to that target
                g.add_weighted_edges_from([(i, len(agents) + j, 1 / len(shortest_paths[i][j]))])  # add a weighted edge
                # between the two nodes in g (I choose the weight to be 1/len() because the function that I found knows
                # how to find a maximum weight match and by doing so I made sure that the shortest path of the
                # maximum weight)
    min_match = nx.algorithms.matching.max_weight_matching(g)  # black box returning the minimum match between
    # agents and targets
    print_solution(min_match, shortest_paths, node_grid, detail_output)  # method responsible of printing solution


def printing_assistance(board, path, i, h, goal_board_number):
    start = i - 1
    while i < len(path) and path[i].get_state() == 2:  # while the next step is occupied by a agent
        i += 1
    finish = i + 1
    while i > start:
        path[i - 1].set_state(0)  # making the step
        if path[i].get_x() != len(board.get_grid()) and path[i].get_y() != len(board.get_grid()):  # if the step we are
            # about to make is not a step towards a exit node (exit node are the way to disappear from the board and we
            # want to keep their state as 0)
            path[i].set_state(2)  # making the step
        board.print_board(h, goal_board_number)  # print the board after changes
        i -= 1
    return finish


def print_solution(optimal_matching, paths, node_grid, detail_output):  # handles solution print
    h = 0
    goal_board_number = 1
    node_grid.print_board(h, goal_board_number)  # print the starting board
    if len(node_grid.get_agents()) > len(optimal_matching):  # if agents list is bigger then the optimal match meaning
        # that there is at least one agent blocked from getting to his target >> no solution
        print('There is no possible solution..')
    else:
        for match in optimal_matching:  # iterate over all matches
            if match[0] < match[1]:  # sometimes the output of the "black box" function returns the tuple
                # representing the agents and targets backwards
                # the smaller number of the two represents the agent
                # after we know all the matches we know which agent goes for which target
                h += AStar.h_score(node_grid.get_agents()[match[0]],
                                   node_grid.get_targets()[match[1] - len(node_grid.get_agents())])  # adding the
                # h score from the agent[i] to the target[j]
                goal_board_number += len(paths[match[0]][match[1] - len(node_grid.get_agents())])-1
            else:
                h += AStar.h_score(node_grid.get_agents()[match[1]],
                                   node_grid.get_targets()[match[0] - len(node_grid.get_agents())])  # adding the
                # h score from the agent[i] to the target[j]
                goal_board_number += len(paths[match[1]][match[0] - len(node_grid.get_agents())])-1

        if not detail_output:  # if the h value is wanted in the output
            h = -1  # in case the h value is not needed
        for match in optimal_matching:
            if match[0] < match[1]:  # in case the tuple produced in the function gets returned backwards
                current_path = paths[match[0]][match[1] - len(node_grid.get_agents())]  # gets the shortest path from
                # agent[i] to the target[j]
            else:
                current_path = paths[match[1]][match[0] - len(node_grid.get_agents())]  # gets the shortest path from
                # agent[i] to the target[j]
            length_of_path = len(current_path)
            if length_of_path != 1:  # if the length_of_path is equal to 1 > meaning the node is already in the desired
                # target and so we don't need to print the board again if no move has been made
                i = 1  # starting in 1 because its actually the first step in the path (0 is the starting node of
                # the path)
                while i < length_of_path:
                    if current_path[i].get_state() == 2:  # if the step we are about to take is occupied by a agent
                        i = printing_assistance(node_grid, current_path, i, h, goal_board_number)  # call for assistance
                    else:  # meaning the next step is not occupied by a agent
                        current_path[i - 1].set_state(0)  # making the step
                        if current_path[i].get_x() != len(node_grid.get_grid()) or current_path[i].get_y() != len(
                                node_grid.get_grid()):  # if the step we are about to make is not a step towards a exit
                            # node (exit node are the way to disappear from the board and we want
                            # to keep their state as 0)
                            current_path[i].set_state(2)  # making the step
                        node_grid.print_board(h, goal_board_number)  # print board after changes
                        i += 1


def main():
    # starting_board = [[2, 0, 2, 0, 2, 0],  # the actual example from the ex 1.pdf
    #                   [0, 0, 0, 2, 1, 2],
    #                   [1, 0, 0, 0, 0, 0],
    #                   [0, 0, 1, 0, 1, 0],
    #                   [2, 0, 0, 0, 0, 0],
    #                   [0, 1, 0, 0, 0, 0]]
    # goal_board = [[2, 0, 2, 0, 0, 0],
    #               [0, 0, 0, 2, 1, 2],
    #               [1, 0, 0, 0, 0, 2],
    #               [0, 0, 1, 0, 1, 0],
    #               [0, 0, 0, 0, 0, 0],
    #               [0, 1, 0, 0, 0, 0]]

    starting_board = [[0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 1, 1, 0],
                      [0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0, 1],
                      [0, 1, 0, 1, 2, 2],
                      [0, 0, 0, 1, 2, 2]]
    goal_board = [[2, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0],
                  [2, 1, 2, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1],
                  [0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0]]
    find_path(starting_board, goal_board, 1, False)


main()
