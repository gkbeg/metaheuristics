import networkx as nx
import numpy as np
import random
import time
import math
from collections import deque
from matplotlib import pyplot as plt

def create_adj_matrix(n):
    global Adj, G
    G = nx.grid_2d_graph(n,n)
    #random.seed(201703830)
    for (u, v) in G.edges():
        if u[1] == 4 and v[1] ==5:
             weight = 3
        elif u[0] == 4 and v[0] == 5:
             weight = 3
        else:
            weight = random.randint(2, 10)
        #print(u, v, rand)
        G.edges[u, v]['weight'] = weight
    Adj = nx.convert_matrix.to_numpy_matrix(G)
    Adj = Adj.astype(int)
    #np.set_printoptions(threshold=np.inf)
    pos = {(x,y):(y,-x) for x,y in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', font_weight='bold')
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
    plt.show()
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
            if Adj[A[i], B[j]] != 0:
                cost = cost + Adj[A[i], B[j]]
    return cost

def goodness(i, A, B):
    deg = degree(i, A, B)
    weight = external(i, A, B)
    #print ("Degree if %s is %s" % (i, deg), "| Ext. Weight of %s is %s" % (i, weight))
    deg, weight = normalize(deg, weight)
    #print("Degree of %s is %s" % (i, deg))
    #print("External of %s is %s" % (i, weight))
    value = 1 - weight / deg
    return value

def normalize(deg, weight):
    deg = deg / 4
    weight = weight / 40
    return deg, weight

def degree (i, A, B):
    deg  = 0
    for j in range(len(A)):
        if Adj[i, A[j]] != 0:
            deg  = deg  + 1
    for j in range(len(B)):
        if Adj[i, B[j]] != 0:
            deg  = deg  + 1
    return deg

def external(i, A, B):
    deg  = 0
    if i in B:
        for j in range(len(A)):
            if Adj[i, A[j]] != 0:
                deg  = deg  + Adj[i, A[j]]
    else:
        for j in range(len(B)):
            if Adj[i, B[j]] != 0:
                deg  = deg  + Adj[i, B[j]]
    return deg

def evaluate_goodness(A, B):
    goodA = []
    goodB = []
    for i in range(len(A)):
        goodA.append(goodness(A[i], A, B))
    for i in range(len(B)):
        goodB.append(goodness(B[i], A, B))
    return goodA, goodB

def tabu (A, B, k, iterations):
    # ---------------------Initialization --------------------#
    tabu = np.zeros((7, 2), dtype = np.int32)
    tabu = tabu.tolist()        #tabu list
    best_solution = 1000        #initial aspiration
    count_tabu = 0              #tabu check counter
    count_moves = 0             #moves counter
    count_asp = 0
    cutset = []
    T = 0
    A = A.tolist()
    B = B.tolist()
    while T < iterations:
        print("************************** Iteration %s ****************************" % T)
        goodA, goodB = evaluate_goodness(A, B)
        print ("A", goodA)
        print ("B", goodB)
        #max_valueA = max(goodA)
        #max_indexA = goodA.index(max_valueA)
        #max_valueB = max(goodB)
        #max_indexB= goodB.index(max_valueB)
        #print("A: Max Value %s at %s" % (max_valueA, max_indexA))
        #print("B: Max Value %s at %s" % (max_valueB, max_indexB))

        # ---------------------SELECTION A, B --------------------#
        #Selection A
        print("============= Selection A ==============")
        Ps_A, Pr_A = selectionloop(A, goodA)
        if len(Ps_A) == 0:
            continue
        print("Ps_A: ", Ps_A,"| Pr_A: ",  Pr_A)
        print("------------Sort by Goodness (Ascending)---------")
        Ps_A, goodPsA_desc = sortbygoodness (Ps_A, goodA, A)
        Pr_A, goodPrA_desc = sortbygoodness (Pr_A, goodA, A)
        print("Ps_A: ", Ps_A,"| Pr_A: ",  Pr_A)
        #Selection B
        print("============= Selection B ==============")
        Ps_B, Pr_B = selectionloop(B, goodB)
        print("Ps_B: ", Ps_B,"| Pr_A: ",  Pr_B)
        if len(Ps_B) == 0:
            continue
        print("------------Sort by Goodness (Ascending)---------")
        Ps_B, goodPsB_desc = sortbygoodness (Ps_B, goodB, B)
        Pr_B, goodPrB_desc = sortbygoodness (Pr_B, goodB, B)
        print("Ps_B: ", Ps_B,"| Pr_B: ",  Pr_B)


        #---------------------Candidate List-----------------#
        print("=======Candidate List k = %s ==========" % (k))
        candidate = np.zeros((k,2), dtype = np.int32)
        for i in range(k):
            #print("Cand. A: ", Ps_A[i],"| Cand. B: ", Ps_B[i])
            if i <= len(Ps_A) - 1 and i <= len(Ps_B) - 1:
                candidate[i,0] = Ps_A[i]
                candidate[i,1] = Ps_B[i]
            else:
                continue
        candidates = candidate.tolist()
        print(candidates)
        mincost = 1000
        minindex = -1
        for i in range(k):
            if candidates[i][0] == 0 and candidates [i][1] == 0:
                continue
            else:
                Aprime, Bprime = move (i, candidates, A, B)
                movecost = cost(Aprime,Bprime)
                print("Cost of move %s is %s." % (i, movecost))
                if movecost <= mincost:
                    mincost = movecost
                    minindex = i
                    BestA = Aprime
                    BestB = Bprime
        print("Lowest cost %s at candidate list at location %s, pairs (%s, %s)"%(mincost,minindex, candidates[minindex][0],candidates[minindex][1]))

        #------------------TABU CHECK -------------------#
        if not tabu_check(tabu, candidates[minindex]):
            A, B = move (minindex, candidates, A, B)
            best_solution = mincost
            print("Best so far: ", best_solution)
            tabu = add_attribute(tabu, candidates[minindex])
            print(tabu)
            count_moves += 1
        elif mincost < best_solution:
            print("Tabu -> Aspiration check passed")
            A, B = move(minindex, candidates, A, B)
            best_solution = mincost
            print("Best so far: ", best_solution)
            tabu = add_attribute(tabu, candidates[minindex])
            print(tabu)
            count_asp += 1
            count_moves += 1
        else:
            print("Tabu -> Aspiration not passed")
            count_tabu += 1

        cutset.append(best_solution)
        T+=1
        # ------------------End of Iteration-----------------------#
    print("Tabu moves --> Aspiration failed ", count_tabu)
    print("Tabu moves --> Aspiration passed: ", count_asp)
    print("Total moves: ", count_moves)
    return BestA, BestB, cutset

def add_attribute(tabu, pair):
    #shift
    queue = deque(tabu)
    queue.rotate(1)
    tabu = list(queue)
    tabu[0] = pair[::-1] #saving reverse of the pair
    return tabu

def tabu_check(tabu, candidate):
    for i in range(len(tabu)):
        if tabu[i] == candidate:
            return True
    return False

def move (i, candidates, A, B):
    Aprime = A.copy()
    Bprime = B.copy()
    Ai = candidates[i][0]
    Bi = candidates[i][1]
    #print (Aprime)
    #print (Bprime)
    if Ai in Aprime:
        indexA = Aprime.index(Ai)
        #print("%s is in A at index %s" % (Ai, indexA))
    if Bi in Bprime:
        indexB = Bprime.index(Bi)
        #print("%s is in B at index %s" % (Bi, indexB))
    Aprime[indexA] = Bi
    Bprime[indexB] = Ai
    #print(Aprime)
    #print(Bprime)
    return Aprime, Bprime

def sortbygoodness(Ps, goodness, part):
    temp = []
    for i in range(len(Ps)):
        index = part.index(Ps[i])
        temp.append(goodness[index])
    #print(Ps)
    #print(temp)
    temp, Ps = (list(t) for t in zip(*sorted(zip(temp, Ps), reverse=False)))
    #print(Ps)
    #print(temp)
    return Ps, temp

def selectionloop (part, good):
    bias = 0
    Ps = []
    Pr = []
    for i in range(len(part)):
        if selection(part[i], bias, part, good):
           Ps.append(part[i])
        else:
           Pr.append(part[i])
    return Ps, Pr

def selection (m, bias, part, good):
    rand = random.uniform(0,1)
    #A = part.tolist()
    if rand <= 1 - good[part.index(m)] + bias:
        #print("Ps", rand, m, good[A.index(m)], 1 - good[A.index(m)] + bias)
        return True
    #print("Pr", rand, m, good[A.index(m)], 1 - good[A.index(m)] + bias)
    return False


def main():
    global cutset, end_criterion
    start_time = time.time()
    n = 10
    create_adj_matrix(n)
    A, B = shuffle(n)
    print("Cost of initial partition: ", cost(A, B))
    #print(goodness(3, A, B))
    #----------Parameters-----------#
    k = 5               # candidate list
    iterations = 150    # end criterion
    # ------------------------------#
    Aprime, Bprime, cutset = tabu(A, B, k, iterations)
    print("\nBest Solution")
    print("A: ", np.sort(Aprime) + 1)
    print("B: ", np.sort(Bprime) + 1)
    print("Iterations: ", iterations)
    print("Min. cut size: ", min(cutset))
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))
    plt.plot(cutset)
    #plt.xlabel('Iterations')
    #plt.ylabel('Cutset cost')
    #plt.xticks(np.arange(0, count+1, 100))
    plt.show()

if __name__ == '__main__':
    main()