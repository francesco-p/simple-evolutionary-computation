#!/usr/bin/python3.5
# Coding: utf-8
# Indentation : 4 spaces


import scipy as sp
import numpy as np
import random
import matplotlib.pyplot as plt

# COSTANTS OF THE PROBLEM
ROWS = 6
COLS = 16
VAR_LEVEL_STEP = 0.2



# TO TUNE               Best
REPETITIONS = 3         #40
GENERATIONS = 10        #10
INITIAL_PHEROMONE = 1   #1
N_ANTS = 100            #100
EVAPORATION = 0.5       #0.5
PH_INIT = 20            #20
PH_BESTSOL = 20         #20
PH_SOL = 2              #2

def createDictionary(dataset):
    """ Given our dataset creates dictionary of [x1,x2,...,x16]:y"""
    dictionary = dict()
    for row in dataset:
        dictionary[keyGen(row[:COLS])] = row[COLS]
    return dictionary

def keyGen(path):
    """ Convertes path in a string form for dictrionary lookup """
    key = ""
    for node in path:
        key += str(node)
    return key

def buildPath(pheromones):
    """ Simulates an Ant given a pheromone matrix"""
    tot_ph = 0
    curr_sum = 0
    idx_path = []
    lvl_path = []
    for ph_col in pheromones.T:
        max_level = 0
        tot_ph = 0
        for ph_value in ph_col:
            if (curr_sum + (max_level*VAR_LEVEL_STEP)) <= 1:
                tot_ph += ph_value
                max_level += 1

        feasible_ph = ph_col[:max_level]
        pr = feasible_ph / tot_ph
        next_vertex = np.random.choice(max_level, 1, p=pr)
        idx_path.append(next_vertex[0])
        lvl_path.append(float("{0:.1f}".format(next_vertex[0] * VAR_LEVEL_STEP)))
        curr_sum += (next_vertex*VAR_LEVEL_STEP)

    return lvl_path,idx_path

def putPherom(matrix, path, qty):
    """ Releases pheromone over the path in the pheromone matrix"""
    for i in range(COLS):
        matrix[path[i],i] += qty

def genRandomPath():
    """ Generates a random path which respect the constrains [0, 0.2, ..., 0.4] sums to 1"""
    lvl_path = [0.0] * 16
    idx_path = [0] * 16

    total = 0
    while (total != 1):
        aux = float("{0:.1f}".format(random.randint(1,5) * 0.2))
        aux2 = random.randint(0,15)
        if (total + aux <= 1) and ( lvl_path[aux2] == 0.0 ):
            lvl_path[aux2] = aux
            total += aux
            idx_path[aux2] = int((aux / VAR_LEVEL_STEP) + 0.5)

    return lvl_path, idx_path

def validPath(path):
    """ Checks if a path respect the constrain """
    summa = 0
    for level in path:
        summa += level
    if (summa == 1.0):
        return True
    return False


#########################################################################################

def main():
    """ Main function code """

    # Read .csv and remove index column
    data = sp.genfromtxt("./space_response.csv", delimiter=";")

    # Create dictionary to lookup the response (simulstes the experiment)
    dataDict = createDictionary(data)

    #%%

    for i in range(REPETITIONS):

        all_solutions_dict = dict()
        pherom = np.ones((ROWS,COLS)) * INITIAL_PHEROMONE

        # Simulates N_ANTS that randomly explored the search space
        for a in range(N_ANTS):
            path, init_path = genRandomPath()
            y = dataDict[keyGen(path)]
            putPherom(pherom, init_path, PH_INIT)

        # Starts generation of ants
        for t in range(GENERATIONS):
            gen_solutions = dict()

            for ant in range(N_ANTS):

                # Kills invalid path of ant (it does not sum up to 1)
                valid = False;
                while(not valid):
                    lvl_path, idx_path = buildPath(pherom)
                    if (validPath(lvl_path)):
                        valid = True

                y = dataDict[keyGen(lvl_path)]
                gen_solutions[y] = idx_path

                # To keep track of global optimum
                all_solutions_dict[y] = idx_path

            # Find best path of current generation
            gen_best = max(gen_solutions.keys())
            gen_best_idx_path = gen_solutions[gen_best]

            # Recompute global best path
            all_best = max(all_solutions_dict.keys())
            all_best_idx_path = all_solutions_dict[all_best]

            # Dbg
            if(t == GENERATIONS-1):
                print(">>> Best path {0} : {1:.3f}".format(gen_best_idx_path,gen_best))
                print(">>> Unique experiments : {0}".format(len(all_solutions_dict)))

            # Initialize matrix of zeros to sum the paths for each ant at current generation
            gen_paths = np.zeros((ROWS,COLS))
            for key in gen_solutions.keys():
                empty = np.zeros((ROWS,COLS))
                if( key == gen_best ):
                    putPherom(empty, gen_solutions[key], PH_BESTSOL)
                else:
                    putPherom(empty, gen_solutions[key], PH_SOL)

                gen_paths += empty

            # Put pheromone also on the global best
            empty = np.zeros((ROWS,COLS))
            putPherom(empty, all_best_idx_path, PH_BESTSOL)
            gen_paths += empty

            # Let everything evaporate and then adds all the generation pheromones
            pherom *= EVAPORATION
            pherom += gen_paths


if __name__ == "__main__":
    main()
