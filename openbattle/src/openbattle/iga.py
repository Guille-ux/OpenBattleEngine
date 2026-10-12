import random
from math import sqrt
from openbattle.common import GameData, Position
from openbattle.order import Order, OrderType
from openbattle.engine import BattleState

# mi ia, porteada a python por gemini
class ModelGenome:
    def __init__(self, name="Modelo_Táctico"):
        self.name = name
        self.weight_damage = 2.5
        self.weight_finish_off = 4.0
        self.weight_supression = 1.5
        self.weight_flanking = 3.5
        self.weight_cover = 2.0
        self.weight_optimal_range = 2.5
        self.penalty_crowding = 1.5
        self.aggression_bias = 1.0
        self.positional = 1.0
        self.health_bias = 1.0
        self.cohesion_bias = 1.0
        self.morale_bias = 1.0
        self.speed_bias = 1.0
        self.range_bias = 1.0

    @staticmethod
    def create_random(p_name="Random_Tactician"):
        g = ModelGenome(p_name)
        g.weight_damage = random.uniform(0.5, 5.0)
        g.weight_finish_off = random.uniform(0.0, 8.0)
        g.weight_supression = random.uniform(0.0, 3.0)
        g.weight_flanking = random.uniform(0.5, 6.0)
        g.weight_cover = random.uniform(0.0, 5.0)
        g.weight_optimal_range = random.uniform(0.0, 4.0)
        g.penalty_crowding = random.uniform(0.2, 3.0)
        g.aggression_bias = random.uniform(2.0, 10.0)
        g.positional = random.uniform(0.1, 5.0)
        g.health_bias = random.uniform(0.1, 1.0)
        g.cohesion_bias = random.uniform(0.1, 1.0)
        g.morale_bias = random.uniform(0.1, 1.0)
        g.speed_bias = random.uniform(0.1, 1.0)
        g.range_bias = random.uniform(0.1, 1.0)
        return g

    def to_dict(self):
        return {
            "name": self.name,
            "weight_damage": self.weight_damage,
            "weight_finish_off": self.weight_finish_off,
            "weight_supression": self.weight_supression,
            "weight_flanking": self.weight_flanking,
            "weight_cover": self.weight_cover,
            "weight_optimal_range": self.weight_optimal_range,
            "penalty_crowding": self.penalty_crowding,
            "aggression_bias": self.aggression_bias,
            "positional": self.positional,
            "health_bias": self.health_bias,
            "cohesion_bias": self.cohesion_bias,
            "morale_bias": self.morale_bias,
            "speed_bias": self.speed_bias,
            "range_bias": self.range_bias
        }

    def save_to_json(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load_from_json(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        g = ModelGenome(data.get("name", "Loaded_Model"))
        for key, val in data.items():
            if key != "name" and hasattr(g, key):
                setattr(g, key, val)
        return g

class IGA:
    def __init__(self, genome=None):
        self.genome = genome if genome else ModelGenome()

    def set_model(self, p_genome):
        self.genome = p_genome

    def get_unit_force(self, unit):
        base_force = unit.unit.speed_kmh * self.genome.speed_bias + unit.mean_range() * self.genome.range_bias
        health_ratio = unit.current_health / unit.unit.max_personnel
        cohesion_ratio = unit.current_cohesion / 100.0
        morale_ratio = unit.current_morale / 100.0
        base_force *= (health_ratio * self.genome.health_bias + 
                       cohesion_ratio * self.genome.cohesion_bias + 
                       morale_ratio * self.genome.morale_bias)
        return base_force

    def casilla_force(self, unit_force, distance):
        return unit_force / float(1 + distance)

    def new_matrix(self, x, y):
        matrix = []
        for i in range(x):
            matrix.append([0.0] * y)
        return matrix

    def analyze_position(self, unit, order, ai_units, p_units):
        map_matrix = self.new_matrix(GameData.map_x_size, GameData.map_y_size)
        final_result = 0.0

        for line in range(GameData.map_x_size):
            for row in range(GameData.map_y_size):
                force = 0.0
                for aunit in ai_units:
                    if aunit.name == unit.name:
                        real_pos = order.arg  # order.arg es un objeto Position
                        distance = abs(real_pos.x - line) + abs(real_pos.y - row)
                        if len(aunit.moving_steps) > 0:
                            distance = abs(aunit.moving_steps[0].x - line) + abs(aunit.moving_steps[0].y - row)
                        force += self.casilla_force(self.get_unit_force(aunit), distance)
                    else:
                        distance = abs(aunit.grid_position.x - line) + abs(aunit.grid_position.y - row)
                        if len(aunit.moving_steps) > 0:
                            distance = abs(aunit.moving_steps[0].x - line) + abs(aunit.moving_steps[0].y - row)
                        force += self.casilla_force(self.get_unit_force(aunit), distance)

                for aunit in p_units:
                    distance = abs(aunit.grid_position.x - line) + abs(aunit.grid_position.y - row)
                    if len(aunit.moving_steps) > 0:
                        distance = abs(aunit.moving_steps[0].x - line) + abs(aunit.moving_steps[0].y - row)
                    force -= self.casilla_force(self.get_unit_force(aunit), distance)

                map_matrix[line][row] = force

        for line in map_matrix:
            for row in line:
                final_result += row

        #final_result /= float(GameData.map_y_size * GameData.map_x_size)
        return final_result

    def get_best_five(self, d, limit=5):
        sorted_keys = sorted(d.keys(), reverse=True)
        best_five = {}
        limit = min(limit, len(sorted_keys))
        for i in range(limit):
            k = sorted_keys[i]
            best_five[k] = d[k]
        return best_five

    def seek_5_best_orders(self, unit, ai_units, p_units):
        possible_attacks = {}
        for enemy in p_units:
            a_score = self.get_attack_score(unit, enemy)
            order = Order(OrderType.ATTACK_ORDER, enemy.grid_position)
            possible_attacks[a_score] = order

        reachable_range = GameData.ticks_per_hour * 2 * unit.unit.speed_kmh
        for obj in self.get_reachable_tiles(unit, reachable_range):
            m_score = self.get_movement_score(unit, ai_units, p_units, obj)
            order = Order(OrderType.GOTO_ORDER, obj)
            possible_attacks[m_score] = order

        return self.get_best_five(possible_attacks, 5)

    def get_movement_score(self, unit, ai_units, p_units, objective):
        distancia_media_a = 0.0
        for aunit in ai_units:
            distancia_media_a += sqrt((objective.x - aunit.grid_position.x)**2 + (objective.y - aunit.grid_position.y)**2)
        distancia_media_a /= float(len(ai_units))

        distancia_media_e = float('inf')
        distancia_media_e_2 = float('inf')
        for aunit in p_units:
            d = sqrt((unit.grid_position.x - aunit.grid_position.x)**2 + (unit.grid_position.y - aunit.grid_position.y)**2)
            if d < distancia_media_e_2:
                distancia_media_e_2 = d

        for aunit in p_units:
            d = sqrt((objective.x - aunit.grid_position.x)**2 + (objective.y - aunit.grid_position.y)**2)
            if d < distancia_media_e:
                distancia_media_e = d

        cover = 1.0 + GameData.get_pos_combat_mod(objective.x, objective.y)
        health_ratio = unit.current_health / unit.unit.max_personnel
        
        final_score = cover * self.genome.weight_cover + (distancia_media_e_2 - distancia_media_e) * self.genome.aggression_bias * (1.5 * sqrt(health_ratio)) + sqrt(distancia_media_a)
        
        if unit.grid_position.x == objective.x and unit.grid_position.y == objective.y:
            final_score -= 10

        return final_score

    def get_attack_score(self, unit, p_unit):
        dict_res = unit.attack(p_unit, offensive=True, sim=True)
        dict_2 = p_unit.attack(unit, offensive=False, sim=True)

        cohesion_loss = dict_res["cohesion_loss"] - dict_2["cohesion_loss"] / self.genome.aggression_bias
        morale_loss = dict_res["morale_loss"] - dict_2["morale_loss"] / self.genome.aggression_bias
        
        # Usamos personnel_loss directamente como el daño
        final_damage = dict_res["personnel_loss"] - dict_2["personnel_loss"] / self.genome.aggression_bias

        bonuses = 0.0
        if (p_unit.current_health - dict_res["personnel_loss"] <= 0 or 
            (p_unit.current_cohesion - dict_res["cohesion_loss"] <= 0.0 and p_unit.current_morale - dict_res["morale_loss"] <= 0.0)):
            bonuses += 100 * self.genome.weight_finish_off
        if final_damage <= 0:
            bonuses -= 1000
        p_final = (cohesion_loss + morale_loss) * self.genome.weight_supression + final_damage * self.genome.weight_damage + bonuses
        return p_final

    def get_reachable_tiles(self, unit, _range):
        reachable_tiles = []
        r = int(_range)
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                pos = Position(unit.grid_position.x + x, unit.grid_position.y + y)
                if pos.x < 0 or pos.x >= GameData.map_x_size or pos.y < 0 or pos.y >= GameData.map_y_size:
                    continue
                distance = abs(x) + abs(y)
                if distance < r:
                    reachable_tiles.append(pos)
        return reachable_tiles

    def execute_orders(self, state, ally = True):
        if ally:
            ai_units = state.a_units
            player_units = state.e_units
        else:
            ai_units = state.e_units
            player_units = state.a_units
        for unit in ai_units:
            if unit.current_health <= 0:
                continue
            best_order = None
            best_score = -99999.0
            best_five = self.seek_5_best_orders(unit, ai_units, player_units)

            for order_score, order in best_five.items():
                current_score = float(order_score) + self.analyze_position(unit, order, ai_units, player_units)
                if current_score > best_score:
                    best_score = current_score
                    best_order = order

            if best_order:
                unit.do_order(best_order)
        return state

def mutate_genome(parent_genome):
    child = ModelGenome(parent_genome.name + "_mut")
    for key, val in parent_genome.to_dict().items():
        if key != "name":
            setattr(child, key, val)
            
    child.weight_damage += random.uniform(-0.4, 0.4)
    child.weight_finish_off += random.uniform(-0.6, 0.6)
    child.weight_supression += random.uniform(-0.3, 0.3)
    child.weight_flanking += random.uniform(-0.5, 0.5)
    child.weight_cover += random.uniform(-0.4, 0.4)
    child.weight_optimal_range += random.uniform(-0.3, 0.3)
    child.penalty_crowding += random.uniform(-0.2, 0.2)
    child.aggression_bias += random.uniform(-0.2, 0.2)
    
    # no se
    child.aggression_bias = max(0.2, min(3.0, child.aggression_bias))
    
    # lo puse asi en godot, seguro q aqui tmb funciona
    child.positional += random.uniform(-0.4, 0.4)
    child.health_bias += random.uniform(-0.4, 0.4)
    child.cohesion_bias += random.uniform(-0.4, 0.4)
    child.morale_bias += random.uniform(-0.4, 0.4)
    child.speed_bias += random.uniform(-0.4, 0.4)
    child.range_bias += random.uniform(-0.4, 0.4)
    
    return child

AIManager = IGA