import random


class BattleState:
    def __init__(self, my_units, e_units):
        self.a_units = my_units
        self.e_units = e_units
        self.end = False
        self.who_won = False # false = ia1, true= ia2

from common import *
from units import *
from ai import *
from equipment import *
from terrain import *
from order import *
from pathfinder import *

class OpenBattleEngine:
    def run_tick(state, player1, player2):
		# ORDENES PRIMER MODELO IA
        state = player1.execute_orders(state, True)
		# ORDENES IA ENEMIGA
        state = player2.execute_orders(state, False)
		# MOVIMIENTOS IA1
        for a in state.a_units:
            if len(a.moving_steps) > 0:
                a.moving_progress += GameData.ticks_per_hour * a.unit.speed_kmh
            else:
                a.moving_progress = 0
            moved = False
            while a.moving_progress >= 1 and len(a.moving_steps) > 0:
                stop_mov = False
                for e in state.a_units:
                    if e.grid_position == a.moving_steps[0] and (e.name!=a.name):
                        stop_mov = True
                        break # LO SIENTO, HAY ALGIUEIN EN MEDIO
                for e in state.e_units:
                    if e.grid_position == a.moving_steps[0]:
                        a.moving_progress -= 3
                        if random.uniform(0.0, 1.0) > 0.7:
                            e.attack(a, False)
                if stop_mov:
                    break
                a.moving_progress -= 1
                a.moving_progress = max(0, a.moving_progress)
                step = a.moving_steps.pop(0)
                a.grid_position = step
                pos_bias = GameData.get_pos_bias(step.x, step.y)
                a.current_cohesion -= 2.5 * pos_bias
                a.current_morale -= 2.5 * pos_bias
                a.current_cohesion = max(0, a.current_cohesion)
                a.current_morale = max(0, a.current_morale)
                moved = True
            if not moved:
                # más vida, jejej, bno cohesion y lo otro
                a.current_cohesion += 2.5
                a.current_morale += 2.5
                a.current_cohesion = min(100, a.current_cohesion)
                a.current_morale = min(100, a.current_morale)
        # Moviminetos IA 2
        for a in state.e_units:
            if len(a.moving_steps) > 0:
                a.moving_progress += GameData.ticks_per_hour * a.unit.speed_kmh
            else:
                a.moving_progress = 0
            moved = False
            while a.moving_progress >= 1 and len(a.moving_steps) > 0:
                stop_mov = False
                for e in state.e_units:
                    if e.grid_position == a.moving_steps[0] and (a.name!=e.name):
                        stop_mov = True
                        break # LO SIENTO, HAY ALGIUEIN EN MEDIO
                for e in state.a_units:
                    if e.grid_position == a.moving_steps[0]:
                        a.moving_progress -= 3
                        if random.uniform(0.0, 1.0) > 0.7:
                            e.attack(a, False)
                if stop_mov:
                    break
                a.moving_progress -= 1
                a.moving_progress = max(0, a.moving_progress)
                step = a.moving_steps.pop(0)
                a.grid_position = step
                pos_bias = GameData.get_pos_bias(step.x, step.y)
                a.current_cohesion -= 2.5 * pos_bias
                a.current_morale -= 2.5 * pos_bias 
                a.current_cohesion = max(0, a.current_cohesion)
                a.current_morale = max(0, a.current_morale)
                moved = True
            if not moved:
                # más vida, jejej, bno cohesion y lo otro
                a.current_cohesion += 2.5
                a.current_morale += 2.5
                a.current_cohesion = min(100, a.current_cohesion)
                a.current_morale = min(100, a.current_morale)
		
		# COMBATE IA1
        for unit in state.a_units:
            attacked = False
            if unit.attacking:
                # aqui lógica de atacar con una orden
                for e in state.e_units:
                    if e.grid_position == unit.objective:
                        # PERFECTO ESTE ES A QN ATACAMOS
                        attacked = True
                        unit.attack(e, True)
                        e.attack(unit, False)
            for e in state.e_units:
                if e.grid_position == unit.grid_position:
                    attacked = True
                    unit.attack(e, True)
                    e.attack(unit, False)
            if not attacked and unit.current_health > 0:
                unit.current_health = min(unit.unit.max_personnel, unit.current_health * 1.01)
		# COMBATE IA2
        for unit in state.e_units:
            attacked = False
            if unit.attacking:
                # aqui lógica de atacar con una orden
                for e in state.a_units:
                    if e.grid_position == unit.objective:
                        # PERFECTO ESTE ES A QN ATACAMOS
                        attacked = True
                        unit.attack(e, True)
                        e.attack(unit, False)
            for e in state.a_units:
                if e.grid_position == unit.grid_position:
                    attacked = True
                    unit.attack(e, True)
                    e.attack(unit, False)
            if not attacked and unit.current_health > 0:
                unit.current_health = min(unit.unit.max_personnel, unit.current_health * 1.01) 
		# DISBANDING DE UNIDADES
        i = 0
        while i < len(state.a_units):
            unit = state.a_units.pop(0)
            if unit.current_health <= 0 or (unit.current_morale >= 0 and unit.current_cohesion <= 0):
                continue
            else:
                state.a_units.append(unit)
            i += 1
        i = 0
        while i < len(state.e_units):
            unit = state.e_units.pop(0)
            
            if unit.current_health <= 0 or (unit.current_morale >= 0 and unit.current_cohesion <= 0):
                continue
            else:
                state.e_units.append(unit)
		# COMPROBACIONES DE VICTORIA
            i += 1
        if len(state.a_units) <= 0:
            state.end = True
            state.who_won = True
        elif len(state.e_units) <= 0:
            state.end = True
            state.who_won = False

        return state
