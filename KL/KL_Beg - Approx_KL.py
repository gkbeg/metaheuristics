from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import random
import time
start_time = time.time()
length = 0

def create_adj_matrix(n):
    global Adj, G
    G = nx.grid_2d_graph(n,n)
    Adj = nx.convert_matrix.to_numpy_matrix(G)
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
    #%A = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    #B = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
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

def KL(A, B):
    gi, cuts = [], []
    queue_a = []
    queue_b = []
    max_gain = 0
    dPrimeFlag = False
    iteration = 0
    A_prime = A.copy()
    B_prime = B.copy()
    cut_size = cut_cost(A, B)
    print("Initial cut size: ", cut_size)
    cuts.append(cut_size)


    while True:
        print("==============Iteration #", iteration, "===============")
        while True:
            if dPrimeFlag == False:
                d = dValues(A, B)
                #print("A, B:", A, B)
                A_prime = A.copy()
                B_prime = B.copy()
                dPrimeFlag = True
                #print("d values: ", d)
            else:
                # I'm using a different function to update D-values as per Lemma 2.2 in the textbook
                d, dupdate = dPrime(d, A_prime, B_prime, queue_a, queue_b)
                #print("d_prime: ", d)
                print("D Values Updated: ", [x+1 for x in dupdate])
            i, j, g = gains(d, A_prime, B_prime)
            queue_a.append(A_prime[i])
            queue_b.append(B_prime[j])
            gi.append(g)
            A_prime, B_prime = remove(A_prime, B_prime, i, j)
            #print("A_prime, B_prime:", A_prime, B_prime)
            #print("Q: ", queue_a, queue_b)
            #print("gi: ", gi)
            if len(A_prime) == 0 and len(B_prime) == 0:
                break
        k, G = find_max_k(gi)
        max_gain += G
        print("Max k: ",k, "| G_k: ", G, "| Sum of G:", + sum(gi))
        if G > 0:
            A, B = swap(queue_a, queue_b, A, B, k)
            cut_size = cut_cost(A, B)
            print("A: ", [x+1 for x in A])
            print("B: ", [x+1 for x in B])
            print("Cut size after swap: ", cut_size)
            cuts.append(cut_size)
            queue_a = []
            queue_b = []
            gi = []
            dPrimeFlag = False
            iteration += 1
            #break
        else:
            break
    return iteration, cuts, A, B


def cut_cost (A, B):
    c = 0
    for i in range(len(A)):
        c = c + cost(i, B)
    return c


def swap(X, Y, A, B, k):
    Y = np.array(Y)
    X = np.array(X)
    #print("O: ", A, B)
    #print("Q: ", np.sort(X), np.sort(Y))
    A_swap = []
    B_swap = []
    remainA = len(A) - k
    remainB = len(B) - k
    for i in range(k+1):
        A_swap.append(Y[i])
        B_swap.append(X[i])

    #print("Moves: ", A_swap, B_swap)
    for i in range(remainA-1):
        A_swap.append(X[k+i+1])
        B_swap.append(Y[k+i+1])
    #A_swap = np.sort(A_swap)
    #B_swap = np.sort(B_swap)
    #print("Moves: ", A_swap, B_swap)

    return A_swap, B_swap


def find_max_k (G):
    sum = 0
    k = 0
    max = G[k]
    for i in range(len(G)):
        sum = sum + G[i]
        if sum > max:
            max = sum
            k = i
    return k, max


def remove(A, B, i, j):
    return np.delete(A, i), np.delete(B, j)


def gains(d, A, B):
    g = []
    max_ai = -1
    max_bj = -1
    max_gain = -100
    #print(d)
    #d = np.sort(d)[::-1]
    #print(d)
    for i in range(len(A)):
        for j in range(len(B)):
            #print("d_i, d_j, c_ij,", d[A[i]], d[B[j]], Adj[A[i], B[j]])
            g_ij = d[A[i]] + d[B[j]] #- (2 * (Adj[A[i], B[j]]))
            g.append(g_ij)
            if g_ij > max_gain:
                max_ai = i
                max_bj = j
                max_gain = g_ij
            #else:
            #    break

    #print("gains: ", g, max_gain)
    #print("i, j:", max_ai, max_bj)
    return max_ai, max_bj, max_gain


def dValues(A,B):
    global length
    d = []
    length = len(A) + len(B)
    for element in range(length):
        if element in A:
           dval = cost(element, B) - cost(element, A)
        else:
           dval = cost(element, A) - cost(element, B)
        d.append(dval)
    return d


def dPrime(d, A, B, qA, qB):
    # Update D-values of all nodes connected to max gain pair, ai, bi via Lemma 2
    ai = qA[-1]
    neighbors = []
    bi = qB[-1]
    x, y = [], []
    #print("Which nodes are attached to a_i and b_i? ", ai, bi)
    #length = len(A) + len(B)
    for element in range(length):
        if checkAdj(element, ai, length):
            neighbors.append(element)
        elif checkAdj(element, bi, length):
            neighbors.append(element)
    #print("Neighbors: ", neighbors)
    #print("A, B: ", A, B)
    for element in range(len(neighbors)):
        if neighbors[element] in A:
            x.append(neighbors[element])
        else:
            y.append(neighbors[element])
    #print(x, y)
    #print("d0:", d)
    update = []
    for i in range(len(x)):
        dxp = d[x[i]] + 2 * cAB(x[i], ai) - 2 * cAB(x[i], bi)
        #print("x: ", x[i], "\t d_x: ", d[x[i]], "\t d_x_p: ", dxp)
        d[x[i]] = dxp
        update.append(x[i])
    for i in range(len(y)):
        dyp = d[y[i]] + 2 * cAB(y[i], bi) - 2 * cAB(y[i], ai)
        #print("y: ", y[i], "\t d_y: ", d[y[i]], "\t d_y_p: ", dyp)
        d[y[i]] = dyp
        update.append(y[i])
    return d, update


def checkAdj(a, b, l):
    for i in range(l):
        if Adj[a, b] == 1:
            return True
    return False


def cAB(a, b):
    if Adj[a, b] == 1:
        return 1
    else:
        return 0


def cost (element, partition):
    cost = 0
    for i in range(len(partition)):
        if Adj[element, partition[i]] == 1:
            cost += 1
    #print("Element cost: ", element, cost)
    return cost


def main():
    n = 10
    create_adj_matrix(n)
    A, B = shuffle(n)

    iterations, cuts, A_swapped, B_swapped = KL(A, B)
    A = A + 1
    B = B + 1
    print("\nFinal Solution")
    print("A: ", np.sort(A_swapped))
    print("B: ", np.sort(B_swapped))
    print("Iterations: ", iterations)
    #print("Cut size: ", cut_cost(A_swapped, B_swapped))
    print("Cut size: ", min(cuts))
    #print("Lowest cost of cut is = ", cost)
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
    # fig.suptitle('KL Heuristic by Beg')
    # pos = {(x, y): (y, -x) for x, y in G.nodes()}
    #
    # nx.draw(G, pos=pos,
    #         node_color='lightblue',
    #         with_labels=True,
    #         node_size=450,
    #         ax=ax1)
    # ax2.set_xlabel('Iterations')
    # ax2.set_ylabel('Cost of cut')
    # ax2.plot(cuts, label = "KL - Iter/Cost")
    plt.plot(cuts)
    plt.xlabel('Iterations')
    plt.ylabel('Cutset cost')
    plt.xticks(np.arange(0, iterations+1, 1))
    plt.show()


if __name__ == '__main__':
    main()