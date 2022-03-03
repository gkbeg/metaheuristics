import networkx as nx
import numpy as np
import random
import time
import math
from matplotlib import pyplot as plt

def graph():
    global G, Adj
    G = nx.Graph()
    G.add_nodes_from([(1, {'pos': (0, 0)}), (2, {'pos': (0, 1)}), (3, {'pos': (1, 1)}), (4, {'pos': (1, 0)}),
                      (5, {'pos': (2, 1)}), (6, {'pos': (3, 0)}), (7, {'pos': (4, 1)}), (8, {'pos': (5, 0)}),
                      (9, {'pos': (2, 0)})])

    G.add_edges_from(
        [(1, 3, {'weight': 1}), (1, 4, {'weight': 3}), (2, 3, {'weight': 4}),
         (3, 4, {'weight': 1}), (3, 9, {'weight': 3}), (3, 5, {'weight': 7}),
         (4, 9, {'weight': 2}), (5, 6, {'weight': 6}), (6, 7, {'weight': 7}),
         (6, 8, {'weight': 8}), (6, 9, {'weight': 4})])
    print(G.number_of_nodes())
    print(G.number_of_edges())
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    Adj = nx.convert_matrix.to_numpy_matrix(G)
    Adj = Adj.astype(int)
    print(Adj)



def initialize():
    S = "aghcbidef"
    print("Initial Solution")
    print("S: ", S)
    initial_pop = initial_population(S, m)
    return initial_pop

def genetic_algorithm (n, m, Wc, Ws, N_g, N_o, M_r):
    fit = []
    fit_o = []
    average = []
    count = 1
    offspring = np.zeros((N_o,n*n))
    pop = initialize()

def main():
    global cutset, end_criterion
    start_time = time.time()
    m = 7           # size of initial population
    Wc = 0.4          # cutsetweight in cost
    Ws = 1 - Wc        # imbalance weight in cost
    N_g = 1000       # number of iterations
    N_o = 2         # number of offsprings per crossover
    M_r = 0.1     # mutation probability
    graph()
    cutset, pop = genetic_algorithm(m, Wc, Ws, N_g, N_o, M_r)
    print(pop)
    #print ("Initial cost: ", cost(A, B, Wc, Ws))
    plt.plot(cutset)
    plt.xlabel('Generations')
    plt.ylabel('Average population cost')
    #plt.xticks(np.arange(0, count+1, 100))
    plt.show()

if __name__ == '__main__':
    main()