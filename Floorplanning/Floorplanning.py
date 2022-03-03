import numpy as np

def initialize ():
    S = np.array([
        [[16,1], [16,1]],
        [[6,6], [6,6]],
        [[2,9], [2,9]],
        [[8,8], [8,8]],
        [[4,4], [4,4]],
        [[12,3], [12,3]]
                  ])
    flip = np.array([1,2,3,4,5,6])
    return S, flip

def slicingfloorplan (E):
    global tree
    cost = 0
    pairs = []
    if E[-1] == 'V':
       print("V cut | EXP: ", E[:])
       if len(E[:-1]) > 3:
            i, j = findSubPolish(E[:-1], 'H')
            print(i, j)
            pairs_i = slicingfloorplan(i)
            pairs_j = slicingfloorplan(j)
            pairs = findminpairs(pairs_i, pairs_j, 'V')
            print(pairs)
            #tree.append(pairs.tolist())
       else:
           pairs = findmin(E[:])
           print(pairs)
           #tree.append(pairs.tolist())
    elif E[-1] == 'H':
       print("H cut | EXP: ", E[:])
       if len(E[:-1]) > 3:
           i, j = findSubPolish(E[:-1], 'V')
           print(i, j)
           pairs_i = slicingfloorplan(i)
           pairs_j = slicingfloorplan(j)
           pairs = findminpairs(pairs_i, pairs_j, 'H')
           print(pairs)
           #tree.append(pairs.tolist())
       else:
           pairs = findmin(E[:])
           print(pairs)
           #tree.append(pairs.tolist())
    return pairs

def findSubPolish(E, operator):
    P1 = ''
    mark = 0
    P2 = ''
    count = 0
    for i in range(len(E)):
        if E[i] == operator and count == 0:
           P1 = E[:i+1]
           mark = i+1
           count += 1
           #print(P1)
        elif E[i] == operator:
           P2 = E[mark:i + 1]
           #print(P2)
    return P1, P2

def findmin(E):
    if E[-1] == 'V':
        i = E[0]
        j = E[1]
        print(i, j, 'V')
        i_pairs, j_pairs = preprocess(i, j)
        pairs = findminpairs(i_pairs, j_pairs, 'V')
    else:
        i = E[0]
        j = E[1]
        print(i, j, 'H')
        i_pairs, j_pairs = preprocess(i, j)
        pairs = findminpairs(i_pairs, j_pairs, 'H')
    return pairs

def preprocess(i, j):
    global S
    global flip
    i_Flip = False
    j_Flip = False
    pairs = []
    for k in range(len(flip)):
        i = int(i)
        j = int(j)
        if i == flip[k]:
            i_Flip = True
        elif flip[k] == j:
            j_Flip = True
    # print(i_Flip, j_Flip)
    i_pairs = S[i - 1, :]
    j_pairs = S[j - 1, :]
    if i_Flip:
        i_pairs = rotatepairs(i_pairs)
    if j_Flip:
        j_pairs = rotatepairs(j_pairs)
    return i_pairs, j_pairs

def rotatepairs(i_pairs):
    rotate = []
    for i in range(len(i_pairs)):
        x = i_pairs[i][0]
        y = i_pairs[i][1]
        rotate.append([x, y])
        if x != y:
            rotate.append([y, x])
        else:
            continue
    i_pairs = np.array(rotate)
    return i_pairs

def findminpairs(i_pairs, j_pairs, op):
    #print(i_pairs)
    #print(j_pairs)
    pairs = []
    if op == 'H':
        #max x
        #sum y
        temp = []
        for i in range(len(i_pairs)):
            for j in range(len(j_pairs)):
                x1 = i_pairs[i][0]
                x2 = j_pairs[j][0]
                x = max(x1, x2)
                y = i_pairs[i][1] + j_pairs[j][1]
                #print(x, y)
                temp.append([x,y])
        #print(temp)
        first = True
        flag = False
        for i in range(len(temp)):
            for j in range(len(temp)):
                x1 = temp[i][0]
                y1 = temp[i][1]
                x2 = temp[j][0]
                y2 = temp[j][1]
                if (x2 < x1 and y2 <= y1) or (x2 <= x1 and y2 < y1):
                   # print("Add", x2, y2)
                    if first:
                        pairs.append([x2, y2])
                        first = False
                    else:
                        for k in range(len(pairs)):
                            if x2 == pairs[k][0] and y2 == pairs[k][1]:
                                flag = True
                        if not flag:
                            pairs.append([x2, y2])
                    flag = False
                    break
                elif x2 == x1 and y2 == y1:
                    #print("Add this", x2, y2)
                    for k in range(len(pairs)):
                        if x2 == pairs[k][0] and y2 == pairs[k][1]:
                            flag = True
                    if not flag:
                        pairs.append([x2, y2])
                    flag = False

       # print("Round 2")
        first = True
        flag = False
        temp = pairs.copy()
        #print(temp)
        for i in range(len(temp)):
            for j in range(len(temp)):
                x1 = temp[i][0]
                y1 = temp[i][1]
                x2 = temp[j][0]
                y2 = temp[j][1]
                if (x2 < x1 and y2 <= y1) or (x2 <= x1 and y2 < y1):
                    #print("Remove", x1, y1)
                    index = 0
                    for m in range(len(pairs)):
                        if pairs[m][0] == x1 and pairs[m][1] == y1:
                            index = m
                    pairs = np.delete(pairs, index, axis=0)
                    break
        #print(pairs)
    else:
        #sum x
        #max y
        temp = []
        for i in range(len(i_pairs)):
            for j in range(len(j_pairs)):
                x1 = i_pairs[i][0]
                x2 = j_pairs[j][0]
                x = x1 + x2
                y = max(i_pairs[i][1],j_pairs[j][1])
                #print(x, y)
                temp.append([x,y])
        #print(temp)
        first = True
        flag = False
        for i in range(len(temp)):
            for j in range(len(temp)):
                x1 = temp[i][0]
                y1 = temp[i][1]
                x2 = temp[j][0]
                y2 = temp[j][1]
                #print (x1,y1, x2,y2)
                if (x2 < x1 and y2 <= y1) or (x2 <= x1 and y2 < y1):
                    #print("Add", x2, y2)
                    if first:
                        pairs.append([x2, y2])
                        first = False
                    else:
                        for k in range(len(pairs)):
                            if x2 == pairs[k][0] and y2 == pairs[k][1]:
                                flag = True
                        if not flag:
                            pairs.append([x2, y2])
                    flag = False
                    break
                elif x2 == x1 and y2 == y1:
                    #print("Add this", x2, y2)
                    for k in range(len(pairs)):
                        if x2 == pairs[k][0] and y2 == pairs[k][1]:
                            flag = True
                    if not flag:
                        pairs.append([x2, y2])
                    flag = False
        #print("Round 2")
        first = True
        flag = False
        temp = pairs.copy()
        #print(temp)
        for i in range(len(temp)):
            for j in range(len(temp)):
                x1 = temp[i][0]
                y1 = temp[i][1]
                x2 = temp[j][0]
                y2 = temp[j][1]
                if (x2 < x1 and y2 <= y1) or (x2 <= x1 and y2 < y1):
                    #print("Remove", x1, y1)
                    index = 0
                    for m in range(len(pairs)):
                        if pairs[m][0] == x1 and pairs[m][1] == y1:
                            index = m
                    pairs = np.delete(pairs, index, axis=0)
                    #print(pairs)
                    break
        #print(pairs)
    return pairs

def mincost (pairs):
    index = 0
    min = pairs[index][0] * pairs[index][1]
    for i in range(len(pairs)):
        area = pairs[i][0] * pairs [i][1]
        if area < min:
            min = area
            index = i
    return index, min

def main ():
    global flip, S, tree, dt
    tree = []
    E = "21H43H56HHH"
    S, flip = initialize()
    print(S)
    n, m, s = np.shape(S)
    cuts = n - 1
    print("E: ", E)
    print ("Rectangles: ", n)
    print ("Cut lines:  " ,n-1)
    print ("Rotatable: ", flip)
    pairs = slicingfloorplan (E)
    i, area = mincost (pairs)
    print("Minimum area of bounding box: ", area)
    print("Pairs: ", pairs[i])
    #print(tree[:])
if __name__ == '__main__':
    main()