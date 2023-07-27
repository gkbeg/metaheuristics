import networkx as nx
import numpy as np
import random
import time
import math
from matplotlib import pyplot as plt

def create_adj_matrix(n):
    global Adj, G
    G = nx.grid_2d_graph(n,n)
    Adj = nx.convert_matrix.to_numpy_array(G)
    Adj = Adj.astype(int)
    print(Adj)

def shuffle(n):
    global combinations
    x = n * n
    nodes = list(range(x))
    random.shuffle(nodes)
    cut = int(x/2)
    A, B = np.sort(nodes[:cut]), np.sort(nodes[cut:])
    combinations = len(A) * len(B)
    A, B = test(n)
    print("Initial Random Solution")
    print("A: ", A+1)
    print("B: ", B+1)
    return A, B

def test(n):
    A = []
    B = []
    j = 0
    for i in range(n*n):
        if j % 2 == 0 :
            A.append(j)
        else:
            B.append(j)
        j+=1
    A = np.array(A)
    B = np.array(B)
    return A, B

def cost (A, B):
    cost = 0
    for i in range(len(A)):
        for j in range(len(B)):
            if Adj[A[i], B[j]] == 1:
                cost = cost + 1
    return cost

def simulated_annealing(A, B, T0, M, alpha, beta, end_criteria):
    global count, cutset
    cutset = []
    min = len(A) + len(B)
    count = 1
    T = T0
    runs = 0
    print("Annealing start: ")
    print("|\t\tCount", "\t\t|\t\t\talpha * T", "\t\t\t|\t\t\tRandom", "\t\t\t\t|\t\t\tCost(S)", "\t\t\t|\t\t\tCost(newS)", "\t\t\t|\t\t\tdelta", "\t\t\t|\t\t\te^-h/T")
    #while T > end_criteria*T0:
    while runs < end_criteria:
        A, B, S = metropolis(A, B, T, M)
        if S < min:
            min = S
        cutset.append(min)
        runs = runs + M
        T = alpha * T
        M = beta * M
        #print(A, B)

def neighbor(A, B):
    A2 = A.copy()
    B2 = B.copy()
    #print(A2 + 1, "|||| ", B2 + 1)
    a = random.choice(A2)
    b = random.choice(B2)
    #print("a: ", a+1, "b:", b+1)
    for i in range(len(A2)):
        if A2[i] == a:
           # print("Before | Ai = ", A[i]+1)
            A2[i] = b
           #print("After  | Ai = ", A[i]+1)
    for i in range(len(B2)):
        if B2[i] == b:
            #print("Before | Bi = ", B[i]+1)
            B2[i] = a
            #print("After  | Bi = ", B[i]+1)
    #print(A2+1,"|||| ", B2+1)
    return A2, B2

def metropolis(A, B, T, M):
    global count, end_criterion, t
    A_prime = A.copy()
    B_prime = B.copy()
    #A_prime, B_prime = copy(A, B)
    #print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    #print("|\t\tCount", "\t\t|\t\t\talpha * T", "\t\t\t|\t\t\tRandom", "\t\t\t\t|\t\t\tCost(S)",
         # "\t\t\t|\t\t\tCost(newS)", "\t\t\t|\t\t\tdelta", "\t\t\t|\t\t\te^-h/T")
    #print("Saved: ", save)
    while M != 0:
        #print(A+1,B+1)
        A_prime, B_prime = neighbor(A, B)
        #print(A_prime+1)#, B_prime)
        newS = cost(A_prime, B_prime)
        S = cost(A, B)
        delta_h =  newS - S
        random_number = random.uniform(0, 1)
        z = metropolis_criteria(-delta_h/T)
        if delta_h < 0 or random_number < math.exp(-delta_h/T):
            print("|\t\t%5.f" % count, "\t\t|\t\t\t%7.3f" % T, "\t\t\t|\t\t\t%7.3f" % random_number, "\t\t\t|\t\t\t%7.3f" % S, "\t\t\t|\t\t\t%7.3f" % newS, "\t\t\t|\t\t\t%7.3f" % delta_h, "\t\t\t|\t\t\t%7.3f" % z)
            #accepting the move
            t = newS
            A = A_prime.copy()
            B = B_prime.copy()
        cutset.append(t)
        count += 1
        end_criterion = end_criterion - 1
        M = M - 1
    return A, B, t

def metropolis_criteria(gamma):
    if gamma > 100:
        return 1
    return math.exp(gamma)

def main():
    global cutset, end_criterion
    start_time = time.time()
    n = 10
    create_adj_matrix(n)
    A, B = shuffle(n)
    print("Cost of initial partition: ", cost(A, B))
    T0 = 150
    alpha = 0.90
    beta = 1.0
    M = 15
    end_criterion = 2000 # if T < T * 0.3
    simulated_annealing(A, B, T0, M, alpha, beta, end_criterion)
    #print(count, cutset[-100:])

    #print("\nFinal Solution")
    #print("A: ", np.sort(A_swapped))
    #print("B: ", np.sort(B_swapped))
    #print("Iterations: ", iterations)
    print("Output cut size: ", min(cutset))
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))
    plt.plot(cutset)
    plt.xlabel('Iterations')
    plt.ylabel('Cutset cost')
    #plt.xticks(np.arange(0, count+1, 100))
    plt.show()

if __name__ == '__main__':
    main()