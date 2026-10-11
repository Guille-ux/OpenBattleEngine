from common import GameData, Position

class Path:
    def __init__(self, p_coste=100000000000, p_pasos=[]):
        self.cost = p_coste
        self.steps = p_pasos
    def clone_advance(self, new_pos, cost):
        new_path = Path(self.cost+cost, self.steps[:])
        new_path.steps.append(new_pos)
        return new_path

class Pathfinder:
    def find_path(start_pos, end_pos):
        total_costes = {}
        caminos = []

        mejor_camino = Path()

        for start in Pathfinder.get_directions(start_pos):
            coste = GameData.get_pos_bias(start.x, start.y)
            if coste < 0:
                continue
            caminos.append(Path(coste, [start]))
            total_costes[start] = coste

        while len(caminos) > 0:
            current = caminos.pop(0)
            if current.cost > mejor_camino.cost:
                continue
            ultima_casilla = current.steps[-1]
            if ultima_casilla == end_pos:
                mejor_camino = current
                continue
            for vecino in Pathfinder.get_directions(ultima_casilla):
                if vecino == start_pos or vecino in current.steps:
                    continue
                peso = GameData.get_pos_bias(vecino.x, vecino.y)
                if peso < 0:
                    continue # no podemos pasar, bruh
                if not vecino in total_costes or (current.cost + peso) < total_costes[vecino]:
                    # parece q x aqui no pasamos, UwU
                    total_costes[vecino] = current.cost + peso
                else:
                    continue
                caminos.append(current.clone_advance(vecino, peso))

        return mejor_camino.steps


    def get_directions(pos):
        direcciones = []
        if pos.x > 0:
            direcciones.append(Position(pos.x-1, pos.y))
        if pos.x < GameData.map_x_size-1:
            direcciones.append(Position(pos.x+1, pos.y))
        if pos.y > 0:
            direcciones.append(Position(pos.x, pos.y-1))
        if pos.y < GameData.map_y_size-1:
            direcciones.append(Position(pos.x, pos.y+1))
        return direcciones

