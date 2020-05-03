
import numpy as np
s_i = np.array([150, 155, 165, 123, 145, 156, 153, 100, 165])
SA_i = np.array([180, 170, 180, 140, 160, 180, 170, 140, 180])
p_i = np.array([3, 4, 1, 2, 1, 3, 4, 1, 5])
t_j = np.array([3, 4, 3, 5, 4, 3, 5, 4, 5, 6])

d_hat_j = np.array([[155, 189],[160, 165], [166, 200], [130, 185], [150, 195], [165, 200], [160, 200], [110, 150], [170, 180], [140, 165]])
d_ij = np.array([[1,3,5,6,8,10,12,15,18,22],
                     [5,16,15,3,4,6,18,12,22,15],
                     [22,18,5,12,8,6,1,4,15,12],
                     [18,12,22,7,5,8,6,8,4,15],
                     [15,8,10,3,22,5,18,4,6,18],
                     [15,12,6,5,3,4,14,8,18,20],
                     [12,22,4,5,3,8,12,15,6,14],
                     [6,4,15,3,8,11,18,5,9,12],
                     [24,8,5,15,4,6,12,15,3,18]])
t_hat_ij = np.array([[[3, 4],[4, 5],[4, 5],[4, 6],[4, 6],[4, 7],[4, 7],[5, 7],[5, 8],[5, 9]],
                    [[5, 6],[5, 7],[5, 7],[4, 7],[4, 7],[4, 5],[5, 8],[4, 7],[5, 9],[5, 7]],
                    [[5, 9],[5, 8],[4, 5],[4, 7],[4, 6],[4, 5],[3, 4],[4, 5],[5, 7],[4, 7]],
                    [[5, 8],[4, 7],[5, 9],[4, 6],[4, 6],[4, 7],[4, 5],[4, 6],[4, 5],[5, 7]],
                    [[5, 7],[4, 6],[4, 7],[4, 5],[5, 9],[5, 6],[5, 8],[4, 5],[4, 5],[5, 8]],
                    [[5, 7],[4, 7],[4, 7],[5, 6],[4, 5],[4, 5],[5, 7],[4, 6],[5, 8],[5, 9]],
                    [[4, 7],[5, 9],[4, 5],[5, 6],[4, 5],[4, 6],[4, 7],[5, 7],[4, 5],[5, 7]],
                    [[4, 5],[4, 5],[5, 7],[4, 5],[4, 6],[4, 7],[4, 5],[5, 6],[4, 6],[4, 7]],
                    [[5, 9],[4, 6],[5, 6],[5, 7],[4, 5],[4, 5],[4, 7],[5, 7],[4, 5],[5, 8]]])
e_ij = np.array([[1, 1, 2, 2, 1, 2, 2, 3, 1, 3],
                [4, 1, 3, 2, 2, 4, 3, 3, 1, 5],
                [3, 2, 5, 1, 6, 3, 5, 4, 6, 3],
                [3, 5, 4, 4, 3, 2, 4, 6, 3, 4],
                [3, 5, 5, 4, 6, 6, 3, 6, 3, 2],
                [2, 2, 3, 4, 3, 1, 4, 3, 2, 4],
                [3, 4, 5, 2, 3, 1, 4, 2, 4, 2],
                [4, 3, 6, 5, 4, 3, 5, 2, 5, 7],
                [3, 4, 2, 6, 8, 6, 2, 3, 6, 3]])
theat_ij = np.array([[4, 2, 3, 1, 2, 3, 4, 5, 2, 6],
                    [4, 5, 3, 2, 1, 3, 2, 1, 3, 4],
                    [3, 4, 6, 2, 3, 2, 4, 1, 3, 5],
                    [5, 6, 2, 3, 4, 1, 4, 5, 2, 3],
                    [3, 2, 5, 2, 3, 2, 2, 6, 4, 3],
                    [5, 6, 1, 3, 2, 4, 6, 5, 7, 2],
                    [8, 2, 2, 5, 3, 3, 6, 3, 4, 7],
                    [1, 3, 2, 4, 1, 2, 4, 3, 1, 4],
                    [3, 5, 4, 1, 3, 2, 3, 2, 5, 2]])
fr_ij = np.array([[3, 4, 2, 2, 7, 4, 6, 1, 2, 4],
                 [2, 3, 1, 3, 5, 1, 6, 3, 6, 1],
                 [2, 4, 9, 1, 8, 4, 8, 4, 7, 6],
                 [1, 5, 3, 6, 3, 5, 1, 5, 1, 5],
                 [1, 6, 3, 3, 3, 4, 9, 6, 2, 1],
                 [4, 3, 4, 1, 5, 4, 8, 3, 6, 3],
                 [7, 3, 4, 6, 1, 4, 1, 6, 3, 6],
                 [6, 3, 1, 4, 4, 5, 1, 6, 2, 5],
                 [6, 6, 4, 1, 3, 1, 1, 3, 6, 4]])
for i in range(len(p_i)):
    for j in range(len(t_j)):
        for k in range(2):
            t_hat_ij[i][j][k] = t_hat_ij[i][j][k] - t_j[j]

# num_s = 3 # 小于等于len(p_i) 大于0
# num_d = 2 # 小于len(t_j) 大于0
# s_i = s_i[:num_s]
# SA_i = SA_i[:num_s]
# p_i = p_i[:num_s]
# t_j = t_j[:num_d]
# d_hat_j = d_hat_j[:num_d,:]
# d_ij = d_ij[:num_s,:num_d]
# t_hat_ij = t_hat_ij[:num_s,:num_d,:]
# e_ij = e_ij[:num_s,:num_d]
# theat_ij = theat_ij[:num_s,:num_d]
# fr_ij = fr_ij[:num_s,:num_d]

def setSDNumber(num_s,num_d,s_i,SA_i, p_i,t_j,d_hat_j,t_hat_ij,d_ij,e_ij,theat_ij,fr_ij):
    s_i = s_i[:num_s]
    SA_i = SA_i[:num_s]
    p_i = p_i[:num_s]
    t_j = t_j[:num_d]
    d_hat_j = d_hat_j[:num_d, :]
    d_ij = d_ij[:num_s, :num_d]
    t_hat_ij = t_hat_ij[:num_s, :num_d, :]
    e_ij = e_ij[:num_s, :num_d]
    theat_ij = theat_ij[:num_s, :num_d]
    fr_ij = fr_ij[:num_s, :num_d]
    return s_i,SA_i, p_i,t_j,d_hat_j,t_hat_ij,d_ij,e_ij,theat_ij,fr_ij

from platypus import NSGAII, Problem, Real
from platypus import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from concurrent.futures import ProcessPoolExecutor

class EMSGMOP_problem(Problem):
    def __init__(self, n, m, s_i, SA_i, p_i,d_hat_j,t_hat_ij,d_ij,e_ij,theat_ij,fr_ij):
        # n*m = 90 decision variables, 2 objectives, and 119 constraints
        super(EMSGMOP_problem, self).__init__(n*m, 2, n*m+n+n+m+1)
        self.n = n
        self.m = m
        self.s_i = s_i
        self.SA_i = SA_i
        self.p_i = p_i
        self.d_hat_j = d_hat_j
        self.t_hat_ij = t_hat_ij
        self.d_ij = d_ij
        self.e_ij = e_ij
        self.theat_ij = theat_ij
        self.fr_ij = fr_ij

        self.types[:] = self.get_bounds()

        self.constraints[0: self.n*self.m] = ">=0"
        self.constraints[self.n*self.m: self.n*(self.m+1)] = "==0"
        self.constraints[self.n*(self.m+1):] = "<=0"

    def get_bounds(self):
        types = []
        bounds = np.zeros([self.n * self.m, 2])
        for i in range(self.n*self.m):
            bounds[i][0] = 0
            bounds[i][1] = 200
            types.append(Real(bounds[i][0], bounds[i][1]))
        return types


    def obj1_fun(self, X):
        tmp = 0
        for i in range(self.n):
            for j in range(self.m):
                tmp += (self.t_hat_ij[i][j]*self.e_ij[i][j]+self.theat_ij[i][j])*X[j+i*self.m]
        return tmp
    def obj2_fun(self,X):
        tmp = 0
        for i in range(self.n):
            for j in range(self.m):
                tmp += (self.fr_ij[i][j]*self.d_ij[i][j]+self.p_i[i])*X[j + i * self.m]
        return tmp

    def objectives_fun(self, X):
        return [self.obj1_fun(X), self.obj2_fun(X)]

    def constraints_fun(self, X):
        cons1 = []
        cons2 = []
        cons3 = []
        cons4 = []
        cons5 = []
        sum_x_ij = 0
        for i in range(self.n):
            sum_x_j = 0
            for j in range(self.m):
                cons1.append(X[j + i * self.m])
                sum_x_j += X[j + i * self.m]
                sum_x_ij += X[j + i * self.m]
            cons2.append(sum_x_j - self.s_i[i])
            cons3.append(sum_x_j - self.SA_i[i])
        sum_d_hat_j = 0
        for j in range(self.m):
            sum_x_i = 0
            for i in range(self.n):
                sum_x_i += X[j + i * self.m]
            cons4.append(sum_x_i - self.d_hat_j[j])
            sum_d_hat_j += self.d_hat_j[j]
        cons5.append(sum_x_ij - sum_d_hat_j)

        cons = cons1 + cons2 + cons3 + cons4 + cons5
        return cons

    def evaluate(self, solution):
        X = solution.variables
        solution.objectives[:] = self.objectives_fun(X)
        solution.constraints[:] = self.constraints_fun(X)

class EMSGSOP_problem(Problem):
    def __init__(self, n, m, s_i, SA_i, p_i,d_hat_j,t_hat_ij,d_ij,e_ij,theat_ij,fr_ij,mF1,mF2):
        # n*m = 90 decision variables, 1 objectives, and 119 constraints
        super(EMSGSOP_problem, self).__init__(n*m, 1, n*m+n+n+m+1)
        self.n = n
        self.m = m
        self.s_i = s_i
        self.SA_i = SA_i
        self.p_i = p_i
        self.d_hat_j = d_hat_j
        self.t_hat_ij = t_hat_ij
        self.d_ij = d_ij
        self.e_ij = e_ij
        self.theat_ij = theat_ij
        self.fr_ij = fr_ij
        self.mF1 = mF1
        self.mF2 = mF2

        self.types[:] = self.get_bounds()

        self.constraints[0: self.n*self.m] = ">=0"
        self.constraints[self.n*self.m: self.n*(self.m+1)] = "==0"
        self.constraints[self.n*(self.m+1):] = "<=0"

    def get_bounds(self):
        types = []
        bounds = np.zeros([self.n * self.m, 2])
        for i in range(self.n*self.m):
            bounds[i][0] = 0
            bounds[i][1] = 200
            types.append(Real(bounds[i][0], bounds[i][1]))
        return types


    def obj1_fun(self, X):
        tmp = 0
        for i in range(self.n):
            for j in range(self.m):
                tmp += (self.t_hat_ij[i][j]*self.e_ij[i][j]+self.theat_ij[i][j])*X[j+i*self.m]
        return tmp
    def obj2_fun(self,X):
        tmp = 0
        for i in range(self.n):
            for j in range(self.m):
                tmp += (self.fr_ij[i][j]*self.d_ij[i][j]+self.p_i[i])*X[j + i * self.m]
        return tmp

    def objectives_fun(self, X):
        return [self.obj1_fun(X)*self.mF1 + self.obj2_fun(X)*self.mF2]

    def constraints_fun(self, X):
        cons1 = []
        cons2 = []
        cons3 = []
        cons4 = []
        cons5 = []
        sum_x_ij = 0
        for i in range(self.n):
            sum_x_j = 0
            for j in range(self.m):
                cons1.append(X[j + i * self.m])
                sum_x_j += X[j + i * self.m]
                sum_x_ij += X[j + i * self.m]
            cons2.append(sum_x_j - self.s_i[i])
            cons3.append(sum_x_j - self.SA_i[i])
        sum_d_hat_j = 0
        for j in range(self.m):
            sum_x_i = 0
            for i in range(self.n):
                sum_x_i += X[j + i * self.m]
            cons4.append(sum_x_i - self.d_hat_j[j])
            sum_d_hat_j += self.d_hat_j[j]
        cons5.append(sum_x_ij - sum_d_hat_j)

        cons = cons1 + cons2 + cons3 + cons4 + cons5
        return cons

    def evaluate(self, solution):
        X = solution.variables
        solution.objectives[:] = self.objectives_fun(X)
        solution.constraints[:] = self.constraints_fun(X)

def EMSG(Algorithm_Run_NUM,n,m,s_i,SA_i, p_i,d_hat_j_tmp,t_hat_ij_tmp,d_ij,e_ij,theat_ij,fr_ij):
    ProcessPoolExecutor(max_workers=8)
    print("-------EMSGMOP:")
    problem = EMSGMOP_problem(n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij, e_ij, theat_ij, fr_ij)

    # 默认求最小目标值
    # problem.directions[:] = Problem.MINIMIZE

    # algorithm = EpsMOEA(problem, epsilons=0.5)
    # algorithm = MOEAD(problem)
    # algorithm = SMPSO(problem)
    # algorithm = OMOPSO(problem, epsilons=0.5)
    # algorithm  = EpsNSGAII(problem, epsilons=0.5)
    # algorithm = SPEA2(problem)
    # algorithm = GDE3(problem)
    algorithm = NSGAII(problem)
    # algorithm = NSGAIII(problem,divisions_outer=25)
    algorithm.population_size = 10
    algorithm.run(Algorithm_Run_NUM)

    # feasible_solutions = [s for s in algorithm.result if s.feasible]
    # if len(feasible_solutions) > 0:
    #     print(feasible_solutions.objectives)
    # print("------------------")

    # print("Population size: ")
    # print(len(algorithm.result))
    # # The output shows on each line the objectives for a Pareto optimal solution:
    # for solution in algorithm.result:
    #     print(solution.variables)

    objectives_result = algorithm.result[0].objectives
    x_result = algorithm.result[0].variables
    x_matrix = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            x_matrix[i][j] = round(x_result[j + i * m])
    r_s_i = x_matrix.sum(axis=1)
    r_d_j = x_matrix.sum(axis=0)
    print("objectives_result:")
    print(objectives_result)
    # print("x_matrix:")
    # print(x_matrix)
    print("r_s_i:")
    print(r_s_i)
    print("std of r_s_i-s_i:")
    print(np.std(r_s_i - s_i))
    print("r_d_j:")
    print(r_d_j)

    # plot the results using matplotlib
    import matplotlib.pyplot as plt

    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])
    plt.xlabel("$f_1(x)$")
    plt.ylabel("$f_2(x)$")
    plt.show()

    return objectives_result, x_matrix
    '''
    mf1 = 1 / objectives_result[0]
    mf2 = 1 / objectives_result[1]
    if flg_EMSGSOP:
        print("-------EMSGSOP:")
        # mf1 = 1/objectives_result[0]
        # mf2 = 1/objectives_result[1]
        problem = EMSGSOP_problem(n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij, e_ij, theat_ij, fr_ij, mf1,
                                  mf2)
        # 单目标 求最大目标值
        problem.directions[:] = Problem.MAXIMIZE
        # 指定算法
        # algorithm = NSGAIII(problem,divisions_outer=25)
        algorithm = NSGAII(problem)
        algorithm.population_size = 10
        algorithm.run(Algorithm_Run_NUM)

        objectives_result = algorithm.result[0].objectives
        x_result = algorithm.result[0].variables
        x_matrix = np.zeros([n, m])
        for i in range(n):
            for j in range(m):
                x_matrix[i][j] = round(x_result[j + i * m])
        r_s_i = x_matrix.sum(axis=1)
        r_d_j = x_matrix.sum(axis=0)
        print("objectives_result:")
        print(objectives_result)
        # print("x_matrix:")
        # print(x_matrix)
        print("r_s_i:")
        print(r_s_i)
        print("std of r_s_i-s_i:")
        print(np.std(r_s_i - s_i))
        print("r_d_j:")
        print(r_d_j)

    #############多目标时（注意问题切换）算法试验对比##############
    if flg_expe and Expe_ON_Prob is 'EMSGMOP':
        problem_name = "EMSGMOP_problem"
        problem = EMSGMOP_problem(n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij, e_ij, theat_ij, fr_ij)
        problem.directions[:] = Problem.MINIMIZE
    elif flg_expe and Expe_ON_Prob is 'EMSGSOP':
        problem = EMSGSOP_problem(n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij, e_ij, theat_ij, fr_ij, mf1,
                                  mf2)
        problem.directions[:] = Problem.MAXIMIZE
        problem_name = "EMSGSOP_problem"
    else:
        return

    algorithms = [NSGAII,
                  (NSGAIII, {"divisions_outer": 25}),
                  (EpsNSGAII, {"epsilons": 0.5}),
                  GDE3,
                  (MOEAD, {"weight_generator": normal_boundary_weights, "divisions_outer": 25}),
                  (OMOPSO, {"epsilons": [0.5]}),
                  SMPSO,
                  SPEA2,
                  (EpsMOEA, {"epsilons": [0.5]})]

    # run the experiment using Python 3's concurrent futures for parallel evaluation
    with ProcessPoolEvaluator(processes=8) as evaluator:
        results = experiment(algorithms, problem, seeds=1, nfe=100000, evaluator=evaluator, display_stats=True)

    # display the results
    fig = plt.figure()
    for i, algorithm in enumerate(six.iterkeys(results)):
        result = results[algorithm][problem_name][0]

        for s0 in result:
            objectives_result = s0.objectives
            x_result = s0.variables
            x_matrix = np.zeros([n, m])
            for ii in range(n):
                for jj in range(m):
                    x_matrix[ii][jj] = round(x_result[jj + ii * m])
            r_s_i = x_matrix.sum(axis=1)
            r_d_j = x_matrix.sum(axis=0)
            print("objectives_result:")
            print(objectives_result)
            # print("x_matrix:")
            # print(x_matrix)
            print("r_s_i:")
            print(r_s_i)
            print("r_d_j:")
            print(r_d_j)

        ax = fig.add_subplot(2, 5, i + 1)
        ax.scatter([s.objectives[0] for s in result],
                   [s.objectives[1] for s in result])
        ax.set_title(algorithm)
        ax.locator_params(nbins=4)
    plt.show()
    '''

if __name__ == '__main__':

    num_s = 2  # 小于等于len(p_i) 大于0
    num_d = 2  # 小于len(t_j) 大于0
    s_i, SA_i, p_i, t_j, d_hat_j, \
    t_hat_ij, d_ij, e_ij, theat_ij, fr_ij \
        = setSDNumber(num_s, num_d, s_i, SA_i, p_i, t_j,
                      d_hat_j, t_hat_ij, d_ij, e_ij,
                      theat_ij, fr_ij)
    ##########################################################
    n = len(p_i)
    m = len(t_j)

    Algorithm_Run_NUM = 100000
    flg_EMSGSOP = True
    flg_expe = True
    Expe_ON_Prob = 'EMSGMOP'  # or 'EMSGSOP'

    ##########################################################
    tau_list = [0, 0.5, 1]
    mu_list = [0, 0.5, 1]
    '''
    objectives_result_list = []
    for i_TAU in range(len(tau_list)):
        for j_MU in range(len(mu_list)):
            tau = tau_list[i_TAU]
            mu = mu_list[j_MU]
            t_hat_ij_tmp = np.zeros([n, m])
            d_hat_j_tmp = np.zeros([m, ])
            for i in range(n):
                for j in range(m):
                    t_hat_ij_tmp[i][j] = t_hat_ij[i][j][1] - mu * (t_hat_ij[i][j][1] - t_hat_ij[i][j][0])
            for j in range(m):
                d_hat_j_tmp[j] = d_hat_j[j][0] + tau * (d_hat_j[j][1] - d_hat_j[j][0])

            objectives_result, x_matrix = EMSG(Algorithm_Run_NUM, n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij,
                 e_ij, theat_ij, fr_ij)
            objectives_result_list.append(objectives_result)
    objectives_results = np.array(objectives_result_list)
    print("---------------------------------------------")
    
    for i_TAU in range(len(tau_list)):
        for j_MU in range(len(mu_list)):
            tau = tau_list[i_TAU]
            mu = mu_list[j_MU]
            t_hat_ij_tmp = np.zeros([n, m])
            d_hat_j_tmp = np.zeros([m, ])
            for i in range(n):
                for j in range(m):
                    t_hat_ij_tmp[i][j] = t_hat_ij[i][j][1] - mu * (t_hat_ij[i][j][1] - t_hat_ij[i][j][0])
            for j in range(m):
                d_hat_j_tmp[j] = d_hat_j[j][0] + tau * (d_hat_j[j][1] - d_hat_j[j][0])

            if flg_EMSGSOP:
                print("-------EMSGSOP:")
                mf1 = 1 / np.amin(objectives_results[:, 0])
                mf2 = 1 / np.amin(objectives_results[:, 1])
                # mf1 = 1 / 4703.277687407595
                # mf2 = 1 / 45465.582073053236
                problem = EMSGSOP_problem(n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij, e_ij, theat_ij, fr_ij, mf1,
                                          mf2)
                # 单目标 求最大目标值
                problem.directions[:] = Problem.MAXIMIZE
                # 指定算法
                # algorithm = NSGAIII(problem,divisions_outer=25)
                algorithm = NSGAII(problem)
                algorithm.population_size = 10
                algorithm.run(Algorithm_Run_NUM)

                objectives_result = algorithm.result[0].objectives
                x_result = algorithm.result[0].variables
                x_matrix = np.zeros([n, m])
                for i in range(n):
                    for j in range(m):
                        x_matrix[i][j] = round(x_result[j + i * m])
                r_s_i = x_matrix.sum(axis=1)
                r_d_j = x_matrix.sum(axis=0)
                print("objectives_result:")
                print(objectives_result)
                # print("x_matrix:")
                # print(x_matrix)
                print("r_s_i:")
                print(r_s_i)
                print("std of r_s_i-s_i:")
                print(np.std(r_s_i - s_i))
                print("r_d_j:")
                print(r_d_j)
    '''
    ##########################################################

    TAU = 0.5
    MU = 1
    t_hat_ij_tmp = np.zeros([n, m])
    d_hat_j_tmp = np.zeros([m, ])
    for i in range(n):
        for j in range(m):
            t_hat_ij_tmp[i][j] = t_hat_ij[i][j][1] - MU * (t_hat_ij[i][j][1] - t_hat_ij[i][j][0])
    for j in range(m):
        d_hat_j_tmp[j] = d_hat_j[j][0] + TAU * (d_hat_j[j][1] - d_hat_j[j][0])

    ProcessPoolExecutor(max_workers=8)
    print("-------EMSGMOP:")
    problem = EMSGMOP_problem(n,m,s_i,SA_i, p_i,d_hat_j_tmp,t_hat_ij_tmp,d_ij,e_ij,theat_ij,fr_ij)

    # 默认求最小目标值
    problem.directions[:] = Problem.MINIMIZE

    # algorithm = EpsMOEA(problem, epsilons=0.5)
    # algorithm = MOEAD(problem)
    # algorithm = SMPSO(problem)
    # algorithm = OMOPSO(problem, epsilons=0.5)
    # algorithm  = EpsNSGAII(problem, epsilons=0.5)
    # algorithm = SPEA2(problem)
    # algorithm = GDE3(problem)
    algorithm = NSGAII(problem)
    # algorithm = NSGAIII(problem,divisions_outer=25)
    algorithm.population_size = 10
    algorithm.run(Algorithm_Run_NUM)

    # feasible_solutions = [s for s in algorithm.result if s.feasible]
    # if len(feasible_solutions) > 0:
    #     print(feasible_solutions.objectives)
    # print("------------------")

    # print("Population size: ")
    # print(len(algorithm.result))
    # # The output shows on each line the objectives for a Pareto optimal solution:
    # for solution in algorithm.result:
    #     print(solution.variables)

    objectives_result = algorithm.result[0].objectives
    x_result = algorithm.result[0].variables
    x_matrix = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            x_matrix[i][j] = round(x_result[j+i*m])
    r_s_i = x_matrix.sum(axis=1)
    r_d_j = x_matrix.sum(axis=0)
    print("objectives_result:")
    print(objectives_result)
    print("x_matrix:")
    print(x_matrix)
    print("r_s_i:")
    print(r_s_i)
    # print("std of r_s_i-s_i:")
    # print(np.std(r_s_i-s_i))
    print("r_d_j:")
    print(r_d_j)

    # plot the results using matplotlib
    import matplotlib.pyplot as plt

    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])
    plt.xlabel("$f_1(x)$")
    plt.ylabel("$f_2(x)$")
    plt.show()

    if flg_EMSGSOP:
        print("-------EMSGSOP:")
        mf1 = 1/objectives_result[0]
        mf2 = 1/objectives_result[1]
        problem = EMSGSOP_problem(n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij, e_ij, theat_ij, fr_ij, mf1, mf2)
        # 单目标 求最大目标值
        problem.directions[:] = Problem.MAXIMIZE
        # 指定算法
        # algorithm = NSGAIII(problem,divisions_outer=25)
        algorithm = NSGAII(problem)
        algorithm.population_size = 10
        algorithm.run(Algorithm_Run_NUM)

        objectives_result = algorithm.result[0].objectives
        x_result = algorithm.result[0].variables
        x_matrix = np.zeros([n, m])
        for i in range(n):
            for j in range(m):
                x_matrix[i][j] = round(x_result[j + i * m])
        r_s_i = x_matrix.sum(axis=1)
        r_d_j = x_matrix.sum(axis=0)
        print("objectives_result:")
        print(objectives_result)
        print("x_matrix:")
        print(x_matrix)
        print("r_s_i:")
        print(r_s_i)
        # print("std of r_s_i-s_i:")
        # print(np.std(r_s_i - s_i))
        print("r_d_j:")
        print(r_d_j)

    #############多目标时（注意问题切换）算法试验对比##############
    if flg_expe and Expe_ON_Prob is 'EMSGMOP':
        problem_name = "EMSGMOP_problem"
        problem = EMSGMOP_problem(n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij, e_ij, theat_ij, fr_ij)
        problem.directions[:] = Problem.MINIMIZE
    elif flg_expe and Expe_ON_Prob is 'EMSGSOP':
        problem = EMSGSOP_problem(n, m, s_i, SA_i, p_i, d_hat_j_tmp, t_hat_ij_tmp, d_ij, e_ij, theat_ij, fr_ij, mf1,
                                  mf2)
        problem.directions[:] = Problem.MAXIMIZE
        problem_name = "EMSGSOP_problem"
    else:
        exit(-1)

    algorithms = [NSGAII,
                  (NSGAIII, {"divisions_outer": 25}),
                  (EpsNSGAII, {"epsilons": 0.5}),
                  GDE3,
                  (MOEAD, {"weight_generator": normal_boundary_weights, "divisions_outer": 25}),
                  (OMOPSO, {"epsilons": [0.5]}),
                  SMPSO,
                  SPEA2,
                  (EpsMOEA, {"epsilons": [0.5]})]

    # run the experiment using Python 3's concurrent futures for parallel evaluation
    with ProcessPoolEvaluator(processes=8) as evaluator:
        results = experiment(algorithms, problem, seeds=1, nfe=100000, evaluator=evaluator, display_stats=True)

    # display the results
    fig = plt.figure()
    for i, algorithm in enumerate(six.iterkeys(results)):
        result = results[algorithm][problem_name][0]

        objectives_result = result[0].objectives
        x_result = result[0].variables
        x_matrix = np.zeros([n, m])
        for ii in range(n):
            for jj in range(m):
                x_matrix[ii][jj] = round(x_result[jj + ii * m])
        r_s_i = x_matrix.sum(axis=1)
        r_d_j = x_matrix.sum(axis=0)
        print("objectives_result:")
        print(objectives_result)
        # print("x_matrix:")
        # print(x_matrix)
        print("r_s_i:")
        print(r_s_i)
        print("r_d_j:")
        print(r_d_j)
        print("-------------------------------")

        ax = fig.add_subplot(2, 5, i + 1)
        ax.scatter([s.objectives[0] for s in result],
                   [s.objectives[1] for s in result])
        ax.set_title(algorithm)
        ax.locator_params(nbins=4)
    plt.show()