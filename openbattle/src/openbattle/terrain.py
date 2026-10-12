class Terrain:
    def __init__(self, tid="plains_tile", name="Plains", fatigue=1.0, mobility_mod = 1.0, cover=0.0, is_passable=True):
        self.id = tid
        self.name = name
        self.fatigue_mod = fatigue
        self.is_passable = is_passable
        self.mobility_mod = 1.0
        self.cover = cover

class Map:
    def __init__(self, terrain_matrix):
        self.map_data = terrain_matrix
        self.height = len(terrain_matrix)
        self.width = len(terrain_matrix[0])
