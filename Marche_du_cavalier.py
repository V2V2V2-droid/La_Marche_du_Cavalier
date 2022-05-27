# Topic: Find an hamitonian graph for a knight on a chest board who needs to pass by all the cases without passing twice by any of them. 
import numpy as np
import random
import math
from itertools import chain

# list of vectors representing the possible moves of the knight.
# the use of tuple instead of np array has been made for convenience in the construction of dependable objects
knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-1, -2), (-2, -1), (1, -2), (2, -1)]


def is_square_number(n):
    """
    Helper function that checks if a number is a squared int.
    Necessary because we have a lack of precision: the sqrt function does not return an exact int for a squared
    number but a close approximation of this which will be a float
    :param n: int
    :return: boleean
    """
    root = np.sqrt(n)
    return n == int((root+0.5))**2


def matrix_mul(m, k=64):
    """
    Function to raise a square matrix to a power k
    :param m: matrix to raise to the k power
    :param k: power of the matrix multiplication
    :return: m^k
    """
    assert m.shape[0] == m.shape[1], "M must be a square matrix"
    if k <= 1:
        return m
    else:
        return np.matmul(matrix_mul(m, k=k-1), m)

# N0 return a huge nb of possible trajectories: max is 2,7 * 10^50


class Chess:
    """
    Instantiation of a chess board object with the following features:
    -a total number of cases: n from which we derive the size
    -the board : a square np array which shape is defined from the size
    -a dictionary listing all the cases (keys) and corresponding coordinate: (x,y) : row and col number (values)
    -a dictionary defining for all the cases (keys), a list of accessible cases for its next move (values)
    -the M0 adjacent matrix: a matrix of size : n x n showing 1 if there exists a possible path between
    two cases and 0 otherwise. By raising M0 to the power k: we obtain the number of possible paths of length k
    between two cases.
    Rk: we look at a square board but other forms can be considered
    Rk: colors are not looked at as we are interested in the knight move. The same code for the bishop or queen should
    take this into account
    """
    def __init__(self, n):
        self.n = n
        assert is_square_number(n) is True, "Chess board must be square"
        self.size = int(np.sqrt(self.n)+0.5)
        self.cases = np.array([c for c in range(int(self.n))])
        self.plateau = np.reshape(self.cases, (self.size, self.size))
        self.coordinate_dict = self.build_coordinate_dict()
        self.moves = self.possible_moves()
        self.M0 = self.build_adjacent_matrix()

    def build_coordinate_dict(self):
        i, j = np.indices((self.size, self.size))
        i_list, j_list = list(chain.from_iterable(i)), list(chain.from_iterable(j))
        a = [(l, k) for l, k in zip(i_list, j_list)]
        return dict(zip(list(self.cases), a))

    def possible_moves(self):
        all_moves = {k: [(v[0]+c[0], v[1]+c[1]) for c in knight_moves] for (k, v) in self.coordinate_dict.items()}
        return {k: [c for c in v if c[0] >= 0 if c[0] <= 7 if c[1] >= 0 if c[1] <= 7] for (k, v) in all_moves.items()}

    def build_adjacent_matrix(self):
        M0 = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            for j in range(0, self.n):
                M0[i][j] = 1 if self.coordinate_dict[j] in self.moves[i] else 0
        return M0


class Game(Chess):
    def __init__(self, p0, u=64):
        """
        The Game object is a series of u moves of a knight from a position p0. It inherits from the Chess class
        From the adjacent matrix M0, inherited from Chess, after each move from case pi, we define the Mi matrix
        by setting the i column and line to 0 (the knight does not have the right to go back to a case previously
        visited so the path is "condemned".
        ie with knight leaving the position pi: Mi+1 = Mi, with line i and column i = 0
        -> M64 will be the null matrix.
        From M0, we calculate N0 = M0^(64) raised to the power 64 : with coordinate (m,l) which counts the nb of
        possible paths of length 64 between case m and case l.
        Ni = Mi(64) at each iteration.
        From p0: initial position, we make a sequence of moves pi that we store in a list.
        Among the list of accessible cases, the choice of the next one follows the algorithms defined below:
        algo 1: we maximise the number of accessible cases
            (so make a count on the lines or columns of the Mi matrix and choose the highest for pi+1)
        algo 2: we maximise the nb of paths of length 64
            (so we take the max cell of matrix Ni among accessible cases and take the corresponding pi+1
        algo 3: same as algo 2 but minimizing the nb of path
        :param p0: Initial position on the board (case number): type int
        :param u: Number of moves in the game, type int
        """
        Chess.__init__(self, n=64)
        self.u = u
        self.p0 = p0
        self.N0 = matrix_mul(self.M0, k=self.n)
        self.position_sequence = []
        self.matrix_seq = []

    def p(self, i):
        """
        Function p(i) : calculate the position pi from pi-1, Mi-1 and Ni-1
        :param i: move index
        :return: pi
        """
        if i == 0:
            return self.p0
        else:
            M_ = self.M(i - 1)
            potential_next_p = [r for r, m in enumerate(M_[self.position_sequence[i-1]]) if m == 1]
            # print("Potential next positions after case: {}".format(str(self.position_sequence[i-1]))) # print statement unactivated because pretty heavy in the final loop: but one can look at one sequence precise construction with this  
            # print(potential_next_p)

            # Formula for algo 3: converges
            N_min = {k: int(self.N(i - 1)[k][k]) for k in potential_next_p}
            p_ = min(N_min, key=N_min.get)  

             # I list here the two other algorithm that were tested and do not converge: 

            # Formula for Algo 1: does not converge : brings us to 25-30 moves
            # M_counts = {k: int(self.M(i-1)[k].sum()) for k in potential_next_p if int(self.M(i-1)[k].sum()) !=0 }
            # Formula for algo 2: does not converge, brings us to about 40 moves
            # N_max = {k:int(self.N(i-1)[k][k]) for k in potential_next_p}
            

            return p_

    def M(self, i):
        """
        Function M(i) that calculate the matrix Mi from pi-1: we set the (i-1) column and row to 0 in Mi
        :param i: index move
        :return: Mi
        """
        if i == 0:
            M_ = self.M0
        else:
            M_ = self.matrix_seq[i-1].copy()
            M_[self.position_sequence[i-1], :] = 0
            M_[:, self.position_sequence[i-1]] = 0

        return M_

    def N(self, i):
        """
        Function N(i) which calculate Ni from Mi: Ni = Mi^(64)
        :param i:
        :return: Ni
        """
        if i == 0:
            return self.N0
        else:
            return matrix_mul(self.M(i))

    def make_ur_move(self):
        """
        Iteration function..
        :return:  position_sequence list to which one new pi is appended after each iteration
        """
        for i in range(0, self.u):
            p_ = self.p(i)
            M_ = self.M(i)
            self.position_sequence.append(p_)
            self.matrix_seq.append(M_)

        return self.position_sequence


# for a smaller board:
# with n = 16 (o 4*4) : the game is limited as instead of min potential next moves = 2 and max = 8,
# we have min = 2 and max = 4

if __name__ == '__main__':

    my_N = 64
    all_starting_positions = list(range(0, my_N))
  
    f = open("marche_du_cavalier.txt", "w+")

    for i in all_starting_positions:
        one_game = Game(p0=i)
        seq = one_game.make_ur_move()
        print("Sequence strating at position {} :".format(str(i)))
        print(seq)
        f.write("Position {}: ({})\n".format(i, '-'.join([str(j) for j in seq])))
    f.close()
