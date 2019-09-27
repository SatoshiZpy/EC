import numpy as np
import copy


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size    # 等于城市数
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        # pop为种群矩阵，大小为（POP_SIZE， N_CITIES），城市编号从1开始
        self.pop = np.vstack([np.random.permutation([__ for __ in range(1, DNA_size + 1)]) for _ in range(pop_size)])

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
            city_coord = city_position[d - 1]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        """
        获取当前群体的适应度
        :param line_x:
        :param line_y:
        :return fitness: ndarray, (pop_SIZE, )
        :return total_distance:  ndarray, (pop_SIZE, )
        """
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        line_x = np.column_stack([line_x, line_x[:, 0]])
        line_y = np.column_stack([line_y, line_y[:, 0]])
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
        :param fitness: ndarray, (pop_SIZE, )
        :return: ndarray ,(pop_SIZE, DNA_SIZE)
        """
        # TODO Zongwei
        selected_pop = np.empty_like(self.pop, dtype=np.uint8)
        for tour_time in range(self.pop_size):
            compete_idx = np.random.choice(np.arange(self.pop_size), size=2, replace=True)
            compete_idx1, compete_idx2 = compete_idx[0], compete_idx[1]
            if fitness[compete_idx1] > fitness[compete_idx2]:
                winner_idx = compete_idx1
            else:
                winner_idx = compete_idx2
            selected_pop[tour_time, :] = self.pop[winner_idx, :]
        return selected_pop

    def select_elitism(self, fitness):
        """
        精英主义选择策略
        :param fitness:
        :return best: 适应度最高的个体，不参与这代的遗传操作  (1, DNA_SIZE)
        :return other_pop: 其他个体，参与这代的遗传操作 (POP_SIZE - 1, DNA_SIZE)
        """
        # TODO Zongwei
        best_idx = np.argmax(fitness)
        worst_idx = np.argmin(fitness)
        best = self.pop[best_idx, :]
        self.pop[worst_idx, :] = best
        other_pop = np.delete(self.pop, best_idx, axis=0)
        return best, other_pop

    def crossover_n_points(self, parent, pop):
        """
        n-points交叉策略
        :param parent:
        :param pop:
        :return:
        """
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, pop.shape[0], size=1)                         # select another individual from pop
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
        child = parent
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size)
            parent2 = pop[i_]
            point1 = np.random.randint(0, self.DNA_size)
            point2 = np.random.randint(point1, self.DNA_size)
            child = parent2[point1: point2]
            for i in range(point2, self.DNA_size):
                tmpA = parent2[i]
                if tmpA in child:
                    i += 1
                else:
                    child = np.append(child, tmpA)
            for j in range(0, point2):
                tmpB = parent2[j]
                if tmpB in child:
                    j += 1
                else:
                    child = np.append(child, tmpB)
        return child

    def crossover_PMX(self, parent, pop):
        """
        PMX交叉策略
        :param parent:
        :param pop:
        :return:
        """
        # TODO Yanmei
        mum = copy.deepcopy(parent)
        np.random.shuffle(copy.deepcopy(pop))
        dad = copy.deepcopy(pop[1])

        if np.random.random() > self.cross_rate or (mum == dad).all():
            return mum

        begin = np.random.randint(0, len(mum) - 2)
        end = np.random.randint(begin + 1, len(mum) - 1)
        for pos in range(begin, end):
            gene1 = mum[pos]
            gene2 = dad[pos]
            if gene1 != gene2:
                posGene1 = np.where(mum == gene1)
                posGene2 = np.where(mum == gene2)
                mum[posGene1], mum[posGene2] = mum[posGene2], mum[posGene1]
        return mum

    def crossover_cycle(self, parent, pop):
        """
        循环交叉策略
        :param parent:
        :param pop:
        :return:
        """
        # TODO Peiyu
        children = parent
        if np.random.rand() < self.cross_rate:
            children = [0] * self.DNA_size
            i_ = np.random.randint(0, pop.shape[0])
            tmpA = {}
            tmpB = {}
            cycles = []

            for i in range(0, self.DNA_size):
                tmpA[i] = False
                tmpB[i] = False
            cycle = []
            cycleComplete = False
            position_control = 0
            # two while loops, inner look to create a cycle, then outter loops controls a new cycle loop,
            # The inner while loop will get run in effect first on the first time processing.
            # End result is that we should have a list of lists containing  cycles
            while False in tmpA.values():
                cycleComplete = False
                if position_control == -1:
                    position_control = 0
                    for key, values in tmpA.items():
                        if values == False:
                            break
                        else:
                            position_control += 1
                while not cycleComplete:
                    if not tmpA[position_control]:
                        cycle.append(parent[position_control])
                        tmpA[position_control] = True
                        tmpB[position_control] = True
                        array = pop[i_].tolist()
                        position_control = array.index(parent[position_control])
                    else:
                        cycleComplete = True
                        position_control = -1
                        cycles.append(cycle.copy())
                        cycle = []
            # Now to cross over , to do this we will loop on out cycles and
            # alternating between A to A, B to B and A to B and B to A copies.
            a_to_a_crossover = True
            for cycle_to_process in cycles[:]:
                if a_to_a_crossover:
                    for key in cycle_to_process[:]:
                        array = parent.tolist()
                        insert_position = array.index(key)
                        del children[insert_position]
                        children.insert(insert_position, key)
                        a_to_a_crossover = False
                else:
                    for key in cycle_to_process[:]:
                        array = pop[i_].tolist()
                        insert_position = array.index(key)
                        del children[insert_position]
                        children.insert(insert_position, key)
                        a_to_a_crossover = True
        return children

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
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                insert_point = np.random.randint(point, self.DNA_size)
                tmp = child[insert_point]
                child = np.delete(child, [insert_point])
                child = np.insert(child, point, [tmp])
        return child

    def mutate_inversion(self, child):
        """
        反转突变策略
        :param child:
        :return:
        """
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                point2 = np.random.randint(point, self.DNA_size)
                c = child[point: point2]
                c = c[::-1]
                child[point: point2] = c[:]
        return child

    def mutate_scramble(self, child):
        """
        征用突变策略
        :param child:
        :return:
        """
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                point2 = np.random.randint(point, self.DNA_size)
                c = child[point: point2]
                np.random.shuffle(c)
                child[point: point2] = c[:]
        return child

    def evolve(self, fitness):
        # pop = self.select_fittness(fitness)
        pop = self.select_fittness(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover_order(parent, pop_copy)
            child = self.mutate_swap(child)
            parent[:] = child
        self.pop = pop

    def evolve_elitism(self, fitness):
        best, other_pop = self.select_elitism(fitness)
        pop_copy = other_pop.copy()
        for parent in other_pop:  # for every parent
            child = self.crossover_cycle(parent, pop_copy)
            child = self.mutate_scramble(child)
            parent[:] = child
        self.pop = np.vstack([best, other_pop])
