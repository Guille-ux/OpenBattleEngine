from openbattle.terrain import *

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, _pos):
        if _pos.x == self.x and _pos.y == self.y:
            return True
        return False
    def __hash__(self):
        return hash((self.x, self.y))

plains = Terrain("plains_tile", "Plains", fatigue=1.0, cover=0.0, is_passable=True)
forest = Terrain("forest_tile", "Forest", fatigue=1.5, cover=0.4, is_passable=True)
mountain = Terrain("mountain_tile", "Mountain", fatigue=2.5, cover=0.7, is_passable=True)
road = Terrain("road_tile", "Road", fatigue=0.6, cover=0.5, is_passable=True)
water = Terrain("plains_tile", "Water", fatigue=99.0, cover=0.0, is_passable=False)


# 2. Matriz 12x12 (12 filas de Y, 12 columnas de X)

terrain_matrix_12x12 = [
    [plains, plains, forest, forest, plains, plains, road,   plains, plains, mountain, mountain, plains],
    [plains, forest, forest, plains, plains, road,   road,   plains, mountain, mountain, plains,    plains],
    [forest, forest, plains, plains, road,   road,   plains, plains, mountain, plains,    plains,    plains],
    [forest, plains, plains, road,   road,   plains, plains, plains, plains,    plains,    plains,   plains],
    [plains, plains, road,   road,   plains, plains, plains, plains,    plains,    plains,   plains,   forest],
    [plains, road,   road,   plains, plains, plains, plains,    plains,    plains,   plains,   forest,   forest],
    [road,   road,   plains, plains, plains, plains,    plains,    plains, plains,   forest,   forest,   plains],
    [road,   plains, plains, plains, plains,    plains,    plains, plains, forest,   forest,   plains,   plains],
    [plains, plains, plains, plains,    plains,    plains, plains, forest, forest,   plains,   plains,   road],
    [plains, plains, plains,    plains,    plains, plains, forest, forest, plains,   plains,    road,     road], # Ajustado al ancho de 12
    [plains, plains,    plains,    plains, plains, forest, forest, plains, plains,   road,     road,     plains],
    [plains,    plains,    plains, plains, forest, forest, plains, plains, road,     road,     plains,   plains]
]

ter = [[plains] * 12 for i in range(12)]

class GameData:
    ticks_per_hour = 0.25
    map_x_size = 12 #
    map_y_size = 12 # COSAS
    map_data = Map(terrain_matrix_12x12)
    def vec2cell(x, y):
        return self.get_cell_id(x, y)
    def get_cell_id(x, y):
        return y+x*GameData.map_y_size+1
    def cell2vec(cell_id):
        return Position(cell_id//GameData.map_y_size, cell_id % GameData.map_y_size)
    def get_pos_bias(x, y):
        tile = GameData.map_data.map_data[y][x]
        if not tile.is_passable:
            return -1
        return tile.fatigue_mod # tile.mobility_mod
    def get_pos_combat_mod(x, y):
        return 1.0 - GameData.map_data.map_data[y][x].cover
# nota esto se cambia fácilito, solo modifica map_data y map_x_size y map_y_size
