import numpy as np


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):     # get cities' coord in order
        """
        给城市编码
        :param DNA: 种群矩阵，ndarray，（POP_SIZE， N_CITIES）
        :param city_position: 城市的横纵坐标矩阵, ndarray, (N_CITIES, 2)
        :return line_x: 种群走过路线的各城市的横坐标，ndarray，（POP_SIZE， N_CITIES）
        :return line_y: 种群走过路线的各城市的横坐标，ndarray，（POP_SIZE， N_CITIES）
        """
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select_fittness(self, fitness):
        """
        轮盘赌选择策略
        :param fitness:
        :return:
        """
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def select_tournament(self, fitness):
        """
        锦标赛选择策略
        :param fitness:
        :return:
        """
        # TODO Zongwei
        pass

    def select_elitism(self, fitness):
        """
        精英主义选择策略
        :param fitness:
        :return:
        """
        # TODO Zongwei
        pass

    def crossover_n_points(self, parent, pop):
        """
        n-points交叉策略
        :param parent:
        :param pop:
        :return:
        """
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            keep_city = parent[~cross_points]                                       # find the city number
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def crossover_order(self, parent, pop):
        """
        顺序交叉策略
        :param parent:
        :param pop:
        :return:
        """
        # TODO Xunshuai
        return parent

    def crossover_PMX(self, parent, pop):
        """
        PMX交叉策略
        :param parent:
        :param pop:
        :return:
        """
        # TODO Yanmei
        return parent

    def crossover_cycle(self, parent, pop):
        """
        循环交叉策略
        :param parent:
        :param pop:
        :return:
        """
        # TODO Peiyu
        return parent

    def crossover_edge(self, parent, pop):
        """
        边缘组合策略
        :param parent:
        :param pop:
        :return:
        """
        # TODO Lu
        return parent

    def mutate_swap(self, child):
        """
        交换突变策略
        :param child:
        :return:
        """
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def mutate_insert(self, child):
        """
        插入突变策略
        :param child:
        :return:
        """
        # TODO Xunshuai
        return child

    def mutate_inversion(self, child):
        """
        反转突变策略
        :param child:
        :return:
        """
        # TODO Xunshuai
        return child

    def mutate_scramble(self, child):
        """
        征用突变策略
        :param child:
        :return:
        """
        # TODO Xunshuai
        return child

    def evolve(self, fitness):
        pop = self.select_fittness(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover_n_points(parent, pop_copy)
            child = self.mutate_swap(child)
            parent[:] = child
        self.pop = pop
