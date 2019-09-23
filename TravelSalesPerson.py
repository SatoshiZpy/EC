import numpy as np
from Logger import Logger
log = Logger('all.log', level='info')


class TravelSalesPerson(object):
    def __init__(self, DATA_SET):
        with open('./Lab file/tsp/' + DATA_SET + '.tsp', 'r') as f:
            data_begin = False
            city_position = []
            for line in f.readlines():
                if 'EOF' in line:
                    break
                if data_begin:  # 记录城市坐标
                    x_pos = float(line.strip().split(' ')[1])
                    y_pos = float(line.strip().split(' ')[2])
                    city_position.append([x_pos, y_pos])
                if 'NAME' in line.strip():
                    self.name = line.strip().split(' ')[-1]
                    log.logger.info('正在读取{}数据集'.format(self.name))
                if 'DIMENSION' in line.strip():     # 读取维度
                    dimension = line.strip().split(' ')[-1]
                    if not dimension.isdigit():
                        raise Exception('文件中DIMENSION读取失败')
                    dimension = int(dimension)
                elif 'NODE_COORD_SECTION' in line.strip():      # 开始读取城市坐标
                    data_begin = True
            if not len(city_position) == dimension:
                raise Exception('读取的城市坐标数据与文件指定维度不一致')
            log.logger.info('{}数据集读取完成，城市数{}'.format(self.name, dimension))
        self.city_position = np.array(city_position)
        self.N_CITIES = int(dimension)

# if __name__ == '__main__':
#     a = 45
#     raise Exception('文件中DIMENSION读取失败')
#     print(666)