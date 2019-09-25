from GA import GA
from TravelSalesPerson import TravelSalesPerson
from Logger import Logger
import numpy as np
log = Logger('all.log', level='info')

CROSS_RATE = 0.1        # 交叉概率
MUTATE_RATE = 0.02      # 突变概率
POP_SIZE = 20          # 种群大小
N_GENERATIONS = 5000     # 迭代轮数
DATA_SET = 'eil51'      # 从10个数据集中选择使用其中一个，在这里写数据集的名字即可


env = TravelSalesPerson(DATA_SET)
N_CITIES = env.N_CITIES
ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
log.logger.info('种群产生完成，种群大小为{}，每个个体的DNA长度为{}'.format(ga.pop_size, ga.DNA_size))

for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)
    fitness, total_distance = ga.get_fitness(lx, ly)
    distance_list = total_distance.tolist()
    ga.evolve_elitism(fitness)
    if generation % 100 == 99:
        log.logger.info('正在进行第{}轮迭代'.format(str(generation + 1)))
        log.logger.info('The best fitness in the current population is {}'.format(str(np.max(fitness))))
    if generation == N_GENERATIONS - 1:
        best_distance = np.min(total_distance)
        best_route = ga.pop[np.argmin(total_distance)]
        out_best_route = '-'.join(map(str, best_route.tolist()))    # 加上分隔符输出
        print('当前种群中表现最佳的总路程{}'.format(str(best_distance)))
        print('当前种群中表现最佳的路径{}'.format(out_best_route))