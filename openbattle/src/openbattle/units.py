from openbattle.equipment import *
from openbattle.order import *
from openbattle.pathfinder import Pathfinder
from math import sqrt
from openbattle.common import GameData

class UnitData:
    uid = "ger_infantry"
    name = "German Infantry Line"
    max_personnel = 800
    speed_kmh = 4.0
    base_morale = 100
    base_cohesion = 100
    detection_radius = 3
    softness = 1
    defence = 30
    inventory = [EquipmentSlot(BaseEquipment, 700)]

class UnitSlot:
    def __init__(self, unit, quantity):
        self.unit = unit
        self.quantity = quantity

class UnitInstance:
    def __init__(self, unit, name, position):
        self.unit = unit
        self.name = name
        self.grid_position = position
        self.current_morale = unit.base_morale
        self.current_cohesion = unit.base_cohesion
        self.current_health = unit.max_personnel
        self.objective = position # no se
        self.attacking = False
        self.moving_steps = []
        self.moving_progress = 0
    def do_order(self, order):
        text = f"""
        -------------------------------------------------------
        Unit Report from {self.name}
        Morale: {self.current_morale}
        Cohesion: {self.current_cohesion}
        Attacking: {self.attacking}
        Health: {self.current_health}/{self.unit.max_personnel}
        Position: {GameData.get_cell_id(self.grid_position.x, self.grid_position.y)}
        -------------------------------------------------------
        
        """
        self.attacking = False
        if order.type == OrderType.GOTO_ORDER:
            destino = order.arg
            self.moving_steps = Pathfinder.find_path(self.grid_position, destino)
            text += f"Now Moving to {destino.x}, {destino.y}\n"
        elif order.type == OrderType.ATTACK_ORDER:
            self.attacking = True
            self.objective = order.arg
            text += f"Now Attacking to {self.objective.x}, {self.objective.y}\n"
        elif order.type == OrderType.CANCEL_ORDER:
            self.moving_steps = []
            text += "All Orders Cancelled\n"
        elif order.type == OrderType.REPORT_ORDER:
            text += "This was the report\n"
        return text
    def check_for_enemies(self, enemy_units):
        text = f"""
        -------------------------------------------------------
        Unit Report from {self.name}
        Morale: {self.current_morale}
        Cohesion: {self.current_cohesion}
        Attacking: {self.attacking}
        Health: {self.current_health}/{self.unit.max_personnel}
        Position: {GameData.get_cell_id(self.grid_position.x, self.grid_position.y)}
        -------------------------------------------------------
        
        """
        detected_ones = []
        for enemy in enemy_units:
            delta_x = abs(self.grid_position.x-enemy.grid_position.x)
            delta_y = abs(self.grid_position.y-enemy.grid_position.y)
            distance = sqrt(delta_x**2 + delta_y**2)
            if distance <= self.detection_radius:
                # detectamos
                detected_ones.append(enemy)
                text += f"Unit Detected at {enemy.grid_position.x}, {enemy.grid_position.y}\n"
        
        if len(detected_ones) < 1:
            return ""
        else:
            return text
    def get_distance(self, pos):
        delta_x = abs(self.grid_position.x-pos.x)
        delta_y = abs(self.grid_position.y-pos.y)
        distance = sqrt(delta_x**2 + delta_y**2)
        return distance
    def mean_range(self):
        total_weapons = 0
        current_range = 0
        for equipment in self.unit.inventory:
            total_weapons += equipment.quantity
            current_range += equipment.quantity * equipment.equipment.max_range
        if total_weapons == 0:
            return 0
        return current_range/total_weapons
    def get_soft_attack(self, pos):
        total_damage = 0
        for eq in self.unit.inventory:
            if eq.equipment.max_range >= self.get_distance(pos):
                total_damage += eq.equipment.soft_attack * sqrt(eq.quantity)
        return total_damage
        
    def get_hard_attack(self, pos):
        total_damage = 0
        for eq in self.unit.inventory:
            if eq.equipment.max_range >= self.get_distance(pos):
                total_damage += eq.equipment.hard_attack * sqrt(eq.quantity)
        return total_damage
    def get_breakthrough(self):
        total_break = 0
        total_quantity = 0
        for eq in self.unit.inventory:
            total_break += eq.equipment.breakthrough * eq.quantity
            total_quantity += eq.quantity
        return total_break/total_quantity
    def get_supression(self, pos):
        total_damage = 0
        for eq in self.unit.inventory:
            if eq.equipment.max_range >= self.get_distance(pos):
                total_damage += eq.equipment.supression * sqrt(eq.quantity)
        return total_damage
    def attack(self, enemy, offensive = True, sim=False):
        soft_damage = self.get_soft_attack(enemy.grid_position)
        hard_damage = self.get_hard_attack(enemy.grid_position)
        supression = self.get_supression(enemy.grid_position)
        breakthrough = enemy.get_breakthrough()

        if offensive:
            defence = enemy.unit.defence
        else:
            defence = breakthrough

        # CONTINUAR POR AQUI
        total_damage = enemy.unit.softness * soft_damage + (1-enemy.unit.softness) * hard_damage

        vulnerability = 2 - enemy.current_cohesion/100
        efficiency = vulnerability * ((self.current_cohesion/100)*0.7+0.3*(self.current_morale/100))
        efficiency *= (self.current_health/self.unit.max_personnel)

        total_damage *= efficiency
        total_damage = total_damage/(100+defence)

        profile = sqrt(enemy.current_health/100)

        if profile < 0.1:
            profile = 0.1

        cover_mod = GameData.get_pos_combat_mod(enemy.grid_position.x, enemy.grid_position.y)

        final_damage = total_damage * profile * cover_mod

        personnel_damage = int(final_damage * 10)
        cohesion_loss = supression + (final_damage/enemy.unit.max_personnel) * 0.5
        morale_loss = (final_damage/enemy.unit.max_personnel) + ((100-cohesion_loss)/100) *0.15

        

        if not sim:
            enemy.current_health -= personnel_damage
            enemy.current_health = max(0, enemy.current_health)
            enemy.current_cohesion -= cohesion_loss
            enemy.current_cohesion = max(0, enemy.current_cohesion)
            enemy.current_morale -= morale_loss
            enemy.current_morale = max(0, enemy.current_morale)

        return {"personnel_loss":personnel_damage,
                "cohesion_loss":cohesion_loss,
                "morale_loss":morale_loss}

