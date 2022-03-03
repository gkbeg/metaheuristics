import networkx as nx
import numpy as np
import random
import time
import math
from matplotlib import pyplot as plt

def create_adj_matrix(n):
    global Adj, G
    G = nx.grid_2d_graph(n,n)
    Adj = nx.convert_matrix.to_numpy_matrix(G)
    Adj = Adj.astype(int)
    print(Adj)

def shuffle(n, m):
    x = n * n
    nodes = list(range(x))
    random.shuffle(nodes)
    cut = int(x/2)
    A, B = np.sort(nodes[:cut]), np.sort(nodes[cut:])
    S = convert(A, B)
    A, B, S = test(n)
    print("Initial Random Solution")
    print("A: ", A)
    print("B: ", B)
    print("S: ", S)
    initial_pop = initial_population(S, m)
    return initial_pop

def initial_population(S, m):
    col = len(S)
    row = m
    population = np.zeros((row,col))
    for i in range(m):
        if i == 0:
            population[0] = S
        elif i > m:
            break
        else:
            population[i] = S
            random.shuffle(population[i])
    #print (population)
    #print (population.shape)
    return population

def selected_population (pop, pfit, off, ofit, m):
    combined = np.concatenate((pfit, ofit), axis=None)
    #print (combined)
    idx = np.argpartition(combined, m)
    #print(idx)
    #print(pop)
    #print(off)
    pop_off = np.vstack([pop, off])
    return pop_off[idx[:m]]

def evaluate_population (pop, m, Wc, Ws):
    fit = []
    for i in range(m):
        fit.append(fitness(pop[i], Wc, Ws))
    max = 0

    # max = np.max(fit)
    # #print(fit, max)
    # for i in range(len(fit)):
    #     if fit[i] < max:
    #         fit[i] -+ max
    #     else:
    #         fit[i] = 0

    return fit

def test(n):
    A = []
    B = []
    S = []
    loc = []
    j = 0
    for i in range(n*n):
        if j % 2 == 0 :
            A.append(j)
            loc.append(j)
            S.append(0)
        else:
            B.append(j)
            loc.append(j)
            S.append(1)
        j+=1
    A = np.array(A)
    B = np.array(B)
    S = np.array(S)
    loc = np.array(loc)
    return A, B, S#, loc

def convert (A, B):
    S = []
    loc = []
    a = 0
    b = 0
    for i in range(len(A)+len(B)):
        if a < len(A) and b < len(B):
            if A[a] == i:
                S.append(0)
                loc.append(i)
                a = a + 1
            elif B[b] == i:
                S.append(1)
                loc.append(i)
                b = b + 1
    S = np.array(S)
    loc = np.array(loc)
    return S#, loc

def cost(A, B, Wc, Ws):
    return Wc * normalize(cutset_weight(A,B), 1) + Ws * normalize(imbalance(A,B), 1)

def imbalance(A, B):
    return len(A) - len(B)

def cutset_weight(A, B):
    cost = 0
    for i in range(len(A)):
        for j in range(len(B)):
            if Adj[A[i], B[j]] == 1:
                cost = cost + 1
    return cost

def fitness (S, Wc, Ws):
    A = []
    B = []
    for i in range(len(S)):
        if S[i] == 0:
            A.append(i)
        elif S[i] == 1:
            B.append(i)
    return cost(A,B,Wc,Ws)

def normalize(S, norm):
    S = S/norm
    return S

def parents(fit, pop):
    #print(pop)
    #print(fit)
    x = []
    y = []
    min = 500
    for i in range(len(fit)):
        if fit[i] < min:
            x = pop[i]
            min = fit[i]
    min = 500
    for i in range(len(fit)):
        if pop[i] is x:
            continue
        elif fit[i] < min:
            y = pop[i]
            min = fit[i]
    #print("P1: ", x.T)
    #print("P2: ", y.T)
    return x, y

def crossover (x, y):
    n = random.randint(0, len(x))
    #print(x,y, fitness(x, 1, 0), fitness(y, 1, 0) )
    a = np.array(x[:n])
    b = np.array(y[n:])
    #print(a,b)
    offspring = np.concatenate((a, b), axis=None)
    #print(offspring, fitness(offspring, 1, 0))
    return offspring

def mutation(S, u):
    for i in range (len(S)):
        n = random.uniform(0, 1)
        if n < u:
            S[i] = 1 - S[i]
    return S

def genetic_algorithm (n, m, Wc, Ws, N_g, N_o, M_r):
    fit = []
    fit_o = []
    average = []
    count = 1
    offspring = np.zeros((N_o,n*n))
    pop = shuffle(n, m)

    fit = evaluate_population(pop, m, Wc, Ws)
    print(fit)

    for i in range(N_g):
        print ("-------------------------------Generation ", count, " -------------------------------------")
        for j in range(N_o):
             #print(fit)
             x, y = parents(fit, pop)
             offspring[j,:] = crossover (x, y)
             fit_o.append(fitness(offspring[j,:], Wc, Ws))
             for k in range(m):
                pop[k] = mutation(pop[k], M_r)

        pop = selected_population(pop, fit, offspring, fit_o, m)
        fit = evaluate_population(pop, m, Wc, Ws)
        print("Population fitness:", fit)
        print("Fitness offsprings:", fit_o)
        fit_o = []
        average.append(np.average(fit))
        count+=1

    return average, pop

def main():
    global cutset, end_criterion
    start_time = time.time()
    n = 10          # grid n*n
    m = 7           # size of initial population
    Wc = 0.7          # cutsetweight in cost
    Ws = 1 - Wc        # imbalance weight in cost
    N_g = 1000       # number of iterations
    N_o = 2         # number of offsprings per crossover
    M_r = 0.1     # mutation probability
    create_adj_matrix(n)
    cutset, pop = genetic_algorithm(n, m, Wc, Ws, N_g, N_o, M_r)
    print(pop)
    #print ("Initial cost: ", cost(A, B, Wc, Ws))
    plt.plot(cutset)
    plt.xlabel('Generations')
    plt.ylabel('Average population cost')
    #plt.xticks(np.arange(0, count+1, 100))
    plt.show()
if __name__ == '__main__':
    main()