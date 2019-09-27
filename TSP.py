from GA import GA
from TravelSalesPerson import TravelSalesPerson
from Logger import Logger
import numpy as np
import datetime
log = Logger('C-PR2392-10.log', level='info')       # set the name of the log

CROSS_RATE = 0.1        # crossover probability
MUTATE_RATE = 0.02      # mutation probability
POP_SIZE = 10           # the size of population
N_GENERATIONS = 20000   # generations, that is, the number of iterations
DATA_SET = 'PR2392'     # the input city data, chosen from the ten data sets

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    env = TravelSalesPerson(DATA_SET)   # get the data of all cities
    N_CITIES = env.N_CITIES
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)     # initialize the population
    print('Completion of population generation. Population Size: {}. DNA Length of Each Individual: {}'.format(ga.pop_size, ga.DNA_size))

    for generation in range(N_GENERATIONS):
        lx, ly = ga.translateDNA(ga.pop, env.city_position)     # get the DNA  of the population
        fitness, total_distance = ga.get_fitness(lx, ly)        # get the fitness and the travel distance of the population
        distance_list = total_distance.tolist()
        ga.evolve_elitism(fitness)                              # for each generation, do the evolution algorithm
        if generation % 100 == 99:                              # log the best fitness for every 100 generations
            print('Current generation: {}'.format(str(generation + 1)))
            log.log_write('Generation {}, the best fitness in the current population is {}'.format(str(generation + 1), str(np.max(fitness))))
        if generation == 4999 or generation == 9999 or generation == 19999:     # log the outcome of 5000, 10000, 20000 generations
            best_distance = np.min(total_distance)
            best_route = ga.pop[np.argmin(total_distance)]
            out_best_route = '-'.join(map(str, best_route.tolist()))
            print('The shortest distance of current population: {}'.format(str(best_distance)))
            print('The best route of current population{}'.format(out_best_route))
        if generation == 9999:
            endtime = datetime.datetime.now()
        if (datetime.datetime.now() - starttime).seconds > 8 * 60 * 60:
            best_distance = np.min(total_distance)
            best_route = ga.pop[np.argmin(total_distance)]
            out_best_route = '-'.join(map(str, best_route.tolist()))    # output the best optimal path
            print('The program ran for 8 hours. Stop running.')
            print('Current Generations: {}, Shortest Distance{}, Best Fitness{}'.format(str(generation), str(best_distance), str(np.max(fitness))))
            print('Current Optimal Path: {}'.format(out_best_route))
            exit()

    print('Time for 10,000 generations of program running: {} seconds'.format((endtime - starttime).seconds))