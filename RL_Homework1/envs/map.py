class Map:
    def __init__(self):
        self.x_range = 1.0  # size of background from -0.5 to 0.5
        self.y_range = 1.0  # size of background from -0.5 to 0.5

    def limit(self):

        x = self.x_range
        y = self.y_range

        # 地图边界，范围从-0.5到0.5
        margin = [[[-0.5, -0.5], [0.5, -0.5]], [[0.5, -0.5], [0.5, 0.5]], [[0.5, 0.5], [-0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]
        # 障碍物（线段），按比例缩放到[-0.5,0.5]范围
        barriers = [[[-0.3, 0.0], [-0.1, 0.0]],
                    [[-0.1, 0.0], [-0.1, -0.5]],
                    [[0.1, 0.0], [0.1, 0.5]],
                    [[0.3, -0.5], [0.3, 0.0]]]

        margin.extend(barriers)
        map_limit = margin
        return map_limit
