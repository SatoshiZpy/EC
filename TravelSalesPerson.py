import numpy as np


class TravelSalesPerson(object):
    def __init__(self, DATA_SET):
        with open('./Lab file/tsp/' + DATA_SET + '.tsp', 'r') as f:
            data_begin = False
            city_position = []
            for line in f.readlines():
                if 'EOF' in line:
                    break
                if data_begin:  # record the coordinate of each city
                    x_pos = float(line.strip().split(' ')[1])
                    y_pos = float(line.strip().split(' ')[2])
                    city_position.append([x_pos, y_pos])
                if 'NAME' in line.strip():
                    self.name = line.strip().split(' ')[-1]
                    print('Starting reading the data set: {}'.format(self.name))
                if 'DIMENSION' in line.strip():     # get the dimension
                    dimension = line.strip().split(' ')[-1]
                    if not dimension.isdigit():
                        raise Exception('Failed to get the DIMENSION in file')
                    dimension = int(dimension)
                elif 'NODE_COORD_SECTION' in line.strip():      # begin to read the coordinates of cities
                    data_begin = True
            if not len(city_position) == dimension:
                raise Exception('the city quantity is not equal to the DIMENSION specified in file')
            print('The data set {} has been read successfully，totally {} cities'.format(self.name, dimension))
        self.city_position = np.array(city_position)
        self.N_CITIES = int(dimension)
