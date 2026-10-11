"""Manticore: rival tactico-coordinado para OpenBattleEngine.

Disenada como oponente de una IA evolutiva, no como reemplazo de ella.

Rasgos:
- Coordina el fuego: evalua ataques en el orden real de combate y evita
  malgastar varios disparos sobre una unidad que ya deberia caer.
- Predice el movimiento que puede completar el enemigo en el tick actual.
- Simula ataque y respuesta usando UnitInstance.attack(..., sim=True).
- Busca posiciones de tiro usando distancias sobre el terreno transitable,
  cobertura, amenaza de fuego, dispersion y potencial de ataque.
- No importa engine.py ni ai.py, para evitar importaciones circulares.
- No modifica pathfinder.py ni necesita bibliotecas externas.

API: RivalAI().execute_orders(state, ally=True)
"""
from heapq import heappush, heappop
from math import atan2, pi, sqrt

from common import GameData, Position
from order import Order, OrderType


class RivalAI:
    """Rival que coordina objetivos y busca intercambios favorables."""

    def __init__(self, name="Manticore"):
        self.name = name
        self._last_orders = {}

    # ---------- utilidades basicas ----------

    @staticmethod
    def _dist(a, b):
        return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @staticmethod
    def _same_pos(a, b):
        return (a is not None and b is not None and
                getattr(a, "x", None) == getattr(b, "x", None) and
                getattr(a, "y", None) == getattr(b, "y", None))

    @staticmethod
    def _alive(unit):
        hp = float(getattr(unit, "current_health", 0.0))
        cohesion = float(getattr(unit, "current_cohesion", 0.0))
        morale = float(getattr(unit, "current_morale", 0.0))
        # Reproduce la condicion de retirada/desbandada del motor.
        return hp > 0.0 and not (morale >= 0.0 and cohesion <= 0.0)

    @staticmethod
    def _max_hp(unit):
        return max(1.0, float(getattr(getattr(unit, "unit", None),
                                     "max_personnel", 1.0)))

    @classmethod
    def _health_ratio(cls, unit):
        return max(0.0, min(1.0, float(unit.current_health) / cls._max_hp(unit)))

    @staticmethod
    def _range(unit):
        try:
            return max(0.0, float(unit.mean_range()))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _max_weapon_range(unit):
        inventory = getattr(getattr(unit, "unit", None), "inventory", None)
        ranges = []
        if inventory:
            for slot in inventory:
                try:
                    if slot.quantity > 0:
                        ranges.append(float(slot.equipment.max_range))
                except (AttributeError, TypeError, ValueError):
                    continue
        return max(ranges) if ranges else RivalAI._range(unit)

    @classmethod
    def _weapons_reach(cls, unit, pos):
        dist = cls._dist(unit.grid_position, pos)
        inventory = getattr(getattr(unit, "unit", None), "inventory", None)
        if not inventory:
            return dist <= cls._range(unit)
        for slot in inventory:
            try:
                if slot.quantity > 0 and float(slot.equipment.max_range) >= dist:
                    return True
            except (AttributeError, TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _cover(pos):
        try:
            return max(0.0, min(1.0,
                1.0 - float(GameData.get_pos_combat_mod(pos.x, pos.y))))
        except (AttributeError, IndexError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _passable(pos):
        try:
            return (0 <= pos.x < GameData.map_x_size and
                    0 <= pos.y < GameData.map_y_size and
                    GameData.get_pos_bias(pos.x, pos.y) >= 0)
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    @staticmethod
    def _neighbors(pos):
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p = Position(pos.x + dx, pos.y + dy)
            if (0 <= p.x < GameData.map_x_size and
                    0 <= p.y < GameData.map_y_size):
                yield p

    # ---------- prediccion de movimiento ----------

    def _predict_position(self, unit, opposing_units):
        """Predice el punto final siguiendo las reglas de movimiento del motor.

        El motor mueve una unidad aunque su casilla este ocupada por un enemigo;
        en ese caso reduce el progreso y puede aplicar un ataque aleatorio. Por
        eso aqui se mueve al primer paso antes de aplicar esa penalizacion.
        """
        pos = unit.grid_position
        route = list(getattr(unit, "moving_steps", []) or [])
        if not route:
            return Position(pos.x, pos.y)
        try:
            progress = float(getattr(unit, "moving_progress", 0.0)) + (
                float(GameData.ticks_per_hour) *
                float(getattr(unit.unit, "speed_kmh", 0.0)))
        except (AttributeError, TypeError, ValueError):
            return Position(pos.x, pos.y)

        while progress >= 1.0 and route:
            step = route.pop(0)
            collision = any(self._alive(other) and
                            self._same_pos(other.grid_position, step)
                            for other in opposing_units)
            pos = step
            progress -= 1.0
            if collision:
                progress -= 3.0
            progress = max(0.0, progress)
        return Position(pos.x, pos.y)

    # ---------- simulacion de combate ----------

    @staticmethod
    def _clamp_stat(value, maximum):
        return max(0.0, min(float(maximum), float(value)))

    def _simulate_hit(self, attacker, target, attacker_pos, target_pos,
                      target_stats=None):
        """Devuelve (golpe, respuesta) sin dejar mutaciones en las unidades."""
        old_attacker_pos = attacker.grid_position
        old_target_pos = target.grid_position
        old_stats = (target.current_health, target.current_cohesion,
                     target.current_morale)
        try:
            attacker.grid_position = attacker_pos
            target.grid_position = target_pos
            if target_stats is not None:
                target.current_health = target_stats[0]
                target.current_cohesion = target_stats[1]
                target.current_morale = target_stats[2]
            hit = attacker.attack(target, offensive=True, sim=True)
            counter = target.attack(attacker, offensive=False, sim=True)
            return hit, counter
        except (AttributeError, IndexError, TypeError, ValueError,
                ZeroDivisionError, OverflowError):
            return None, None
        finally:
            attacker.grid_position = old_attacker_pos
            target.grid_position = old_target_pos
            target.current_health, target.current_cohesion, target.current_morale = old_stats

    def _threat_value(self, enemy):
        # Valora eliminar una unidad por el fuego que deja de aportar.
        return (0.25 + 0.045 * self._max_weapon_range(enemy)
                + 0.40 * self._health_ratio(enemy))

    def _shot_score(self, attacker, target, target_pos, attacker_pos=None,
                    target_stats=None):
        if attacker_pos is None:
            attacker_pos = attacker.grid_position
        if not self._weapons_reach_at(attacker, attacker_pos, target_pos):
            return float("-inf"), None, None
        hit, counter = self._simulate_hit(attacker, target, attacker_pos,
                                          target_pos, target_stats)
        if hit is None or counter is None:
            return float("-inf"), None, None

        hp, cohesion, morale = (target_stats if target_stats is not None else
                                (target.current_health, target.current_cohesion,
                                 target.current_morale))
        damage = max(0.0, float(hit.get("personnel_loss", 0.0)))
        suppression = max(0.0, float(hit.get("cohesion_loss", 0.0)))
        morale_loss = max(0.0, float(hit.get("morale_loss", 0.0)))
        if damage <= 0.0 and suppression <= 0.0:
            return float("-inf"), hit, counter

        hp_after = max(0.0, float(hp) - damage)
        cohesion_after = max(0.0, float(cohesion) - suppression)
        morale_after = max(0.0, float(morale) - morale_loss)
        kill = hp_after <= 0.0 or (morale_after >= 0.0 and cohesion_after <= 0.0)

        counter_damage = max(0.0, float(counter.get("personnel_loss", 0.0)))
        counter_cohesion = max(0.0, float(counter.get("cohesion_loss", 0.0)))
        counter_morale = max(0.0, float(counter.get("morale_loss", 0.0)))

        # Da prioridad al impacto relativo y al remate, sin ignorar el coste del
        # contraataque. El valor de quitar una unidad enemiga es global al equipo.
        score = (
            1.55
            + 2.40 * damage / max(1.0, float(hp))
            + 1.15 * suppression / max(1.0, float(cohesion))
            + 0.35 * morale_loss / max(1.0, float(morale))
            + 0.22 * self._threat_value(target)
            - 1.20 * counter_damage / self._max_hp(attacker)
            - 0.45 * counter_cohesion / 100.0
            - 0.15 * counter_morale / 100.0
        )
        if kill:
            score += 3.25
        elif hp_after <= 0.16 * self._max_hp(target):
            score += 0.30
        return score, hit, counter

    @classmethod
    def _weapons_reach_at(cls, unit, attacker_pos, target_pos):
        dist = cls._dist(attacker_pos, target_pos)
        inventory = getattr(getattr(unit, "unit", None), "inventory", None)
        if not inventory:
            return dist <= cls._range(unit)
        for slot in inventory:
            try:
                if slot.quantity > 0 and float(slot.equipment.max_range) >= dist:
                    return True
            except (AttributeError, TypeError, ValueError):
                continue
        return False

    def _assign_attacks(self, allies, enemies, predicted):
        """Asignacion secuencial de fuego, aproximando el orden real del motor."""
        projected = {
            id(e): [float(e.current_health), float(e.current_cohesion),
                    float(e.current_morale)] for e in enemies
        }
        assignments = {}
        # El motor resuelve combate siguiendo el orden de las listas de unidades;
        # usar ese mismo orden mejora la estimacion del estado tras cada ataque.
        for unit in allies:
            best = None
            best_score = float("-inf")
            for enemy in enemies:
                stats = projected[id(enemy)]
                if stats[0] <= 0.0 or (stats[2] >= 0.0 and stats[1] <= 0.0):
                    continue
                target_pos = predicted[id(enemy)]
                score, hit, _counter = self._shot_score(
                    unit, enemy, target_pos, target_stats=tuple(stats))
                if hit is None or score <= best_score:
                    continue
                best = (enemy, target_pos, hit, score)
                best_score = score

            if best is None or best_score <= 0.15:
                continue

            enemy, target_pos, hit, _score = best
            assignments[id(unit)] = (enemy, target_pos)
            stats = projected[id(enemy)]
            stats[0] = max(0.0, stats[0] - max(0.0, float(hit.get("personnel_loss", 0.0))))
            stats[1] = max(0.0, stats[1] - max(0.0, float(hit.get("cohesion_loss", 0.0))))
            stats[2] = max(0.0, stats[2] - max(0.0, float(hit.get("morale_loss", 0.0))))
        return assignments

    # ---------- posicionamiento ----------

    def _distance_field(self, goal):
        """Dijkstra en el mapa pequeno; no cambia ni llama a pathfinder.py."""
        inf = float("inf")
        dist = {(x, y): inf for x in range(GameData.map_x_size)
                for y in range(GameData.map_y_size)}
        dist[(goal.x, goal.y)] = 0.0
        queue = [(0.0, goal.x, goal.y)]
        while queue:
            cost, x, y = heappop(queue)
            if cost != dist[(x, y)]:
                continue
            current = Position(x, y)
            # Reverse edges: el coste de ir de predecessor a current es el coste
            # de entrar en current, igual que en Pathfinder.find_path.
            for prev in self._neighbors(current):
                if not self._passable(prev):
                    continue
                step_cost = max(0.05, float(GameData.get_pos_bias(x, y)))
                new_cost = cost + step_cost
                key = (prev.x, prev.y)
                if new_cost < dist[key]:
                    dist[key] = new_cost
                    heappush(queue, (new_cost, prev.x, prev.y))
        return dist

    def _incoming_score(self, unit, candidate, enemies, predicted):
        danger = 0.0
        for enemy in enemies:
            enemy_pos = predicted[id(enemy)]
            if not self._weapons_reach_at(enemy, enemy_pos, candidate):
                continue
            hit, _counter = self._simulate_hit(enemy, unit, enemy_pos, candidate)
            if hit is None:
                continue
            damage = max(0.0, float(hit.get("personnel_loss", 0.0)))
            cohesion = max(0.0, float(hit.get("cohesion_loss", 0.0)))
            if damage <= 0.0 and cohesion <= 0.0:
                continue
            danger += 1.10 * damage / self._max_hp(unit) + 0.40 * cohesion / 100.0
        return danger

    def _flank_score(self, candidate, target_pos, allies, unit):
        dx = candidate.x - target_pos.x
        dy = candidate.y - target_pos.y
        if dx == 0 and dy == 0:
            return 0.0
        angle = atan2(dy, dx)
        nearest_separation = pi
        found = False
        for other in allies:
            if other is unit or not self._alive(other):
                continue
            ox = other.grid_position.x - target_pos.x
            oy = other.grid_position.y - target_pos.y
            if ox == 0 and oy == 0:
                continue
            delta = abs((angle - atan2(oy, ox) + pi) % (2.0 * pi) - pi)
            nearest_separation = min(nearest_separation, delta)
            found = True
        return nearest_separation / pi if found else 0.25

    def _position_score(self, unit, candidate, allies, enemies, predicted,
                        fields, reserved):
        if any(other is not unit and self._alive(other) and
               self._same_pos(other.grid_position, candidate) for other in allies):
            return float("-inf")
        if any(self._same_pos(enemy.grid_position, candidate) or
               self._same_pos(predicted[id(enemy)], candidate) for enemy in enemies):
            return float("-inf")
        if (candidate.x, candidate.y) in reserved and not self._same_pos(candidate, unit.grid_position):
            return float("-inf")

        cover = self._cover(candidate)
        incoming = self._incoming_score(unit, candidate, enemies, predicted)
        bias = max(0.0, float(GameData.get_pos_bias(candidate.x, candidate.y)))
        fire_best = 0.0
        fire_sum = 0.0
        approach_values = []
        flank_best = 0.0

        for enemy in enemies:
            target_pos = predicted[id(enemy)]
            field = fields[id(enemy)]
            path_cost = field.get((candidate.x, candidate.y), float("inf"))
            max_range = self._max_weapon_range(unit)
            remaining = max(0.0, path_cost - max_range)
            # Potencial de alcance. El camino por el terreno evita que se atasque
            # intentando atravesar agua o rodeando obstaculos de forma local.
            approach_values.append(-remaining * 0.24)

            score, hit, _counter = self._shot_score(
                unit, enemy, target_pos, attacker_pos=candidate)
            if hit is not None and score > 0.0:
                fire_best = max(fire_best, score)
                fire_sum += score
            flank_best = max(flank_best, self._flank_score(
                candidate, target_pos, allies, unit))

        # Fire potential is dominant near contact; path progress and cover drive
        # approach when no reliable shot exists yet.
        approach = max(approach_values) if approach_values else -100.0
        closest_now = min((fields[id(e)].get((unit.grid_position.x,
                                               unit.grid_position.y), 1e6)
                           - self._max_weapon_range(unit)) for e in enemies)
        closest_next = min((fields[id(e)].get((candidate.x, candidate.y), 1e6)
                            - self._max_weapon_range(unit)) for e in enemies)
        progress = max(-3.0, min(3.0, closest_now - closest_next))

        crowding = 0.0
        for other in allies:
            if other is unit or not self._alive(other):
                continue
            d = self._dist(candidate, other.grid_position)
            if d < 1.5:
                crowding += (1.5 - d) * 0.55

        return (0.95 * cover + 1.10 * fire_best + 0.12 * fire_sum
                + 0.90 * progress + 0.35 * approach
                + 0.32 * flank_best - 1.15 * incoming
                - 0.075 * bias - crowding)

    def _choose_move(self, unit, allies, enemies, predicted, fields, reserved):
        # Consider quedarse quieto: abandonar una buena posicion de fuego puede
        # ser peor que mantenerla un turno sin disparar.
        choices = [Position(unit.grid_position.x, unit.grid_position.y)]
        choices.extend(p for p in self._neighbors(unit.grid_position)
                       if self._passable(p))
        best_pos = choices[0]
        best_score = float("-inf")
        for pos in choices:
            score = self._position_score(unit, pos, allies, enemies, predicted,
                                         fields, reserved)
            # Pequeno desempate estable que no depende de aleatoriedad global.
            tie = ((unit.grid_position.x * 17 + unit.grid_position.y * 31
                    + pos.x * 7 + pos.y * 13) % 101) * 1e-6
            score += tie
            if score > best_score:
                best_score = score
                best_pos = pos
        return best_pos

    # ---------- emision de ordenes ----------

    def _issue_attack(self, unit, destination):
        key = id(unit)
        signature = (OrderType.ATTACK_ORDER, destination.x, destination.y)
        if (self._last_orders.get(key) == signature and
                getattr(unit, "attacking", False) and
                not getattr(unit, "moving_steps", None) and
                self._same_pos(getattr(unit, "objective", None), destination)):
            return
        # El motor conserva moving_steps cuando recibe ATTACK_ORDER; hay que
        # limpiarlos expresamente para que el ataque no arrastre una ruta vieja.
        unit.moving_steps = []
        unit.moving_progress = 0.0
        unit.do_order(Order(OrderType.ATTACK_ORDER, destination))
        self._last_orders[key] = signature

    def _issue_move(self, unit, destination):
        key = id(unit)
        signature = (OrderType.GOTO_ORDER, destination.x, destination.y)
        if (self._last_orders.get(key) == signature and
                getattr(unit, "moving_steps", None)):
            return
        if self._same_pos(unit.grid_position, destination):
            if getattr(unit, "moving_steps", None) or getattr(unit, "attacking", False):
                unit.moving_steps = []
                unit.moving_progress = 0.0
                unit.do_order(Order(OrderType.CANCEL_ORDER, None))
            self._last_orders[key] = (OrderType.CANCEL_ORDER, destination.x, destination.y)
            return
        unit.do_order(Order(OrderType.GOTO_ORDER, destination))
        self._last_orders[key] = signature

    def execute_orders(self, state, ally=True):
        allies_raw = state.a_units if ally else state.e_units
        enemies_raw = state.e_units if ally else state.a_units
        allies = [u for u in allies_raw if self._alive(u)]
        enemies = [u for u in enemies_raw if self._alive(u)]
        if not enemies or not allies:
            return state

        predicted = {id(enemy): self._predict_position(enemy, allies)
                     for enemy in enemies}
        # Un campo de distancias por enemigo permite planificar alrededor del
        # terreno sin invocar Pathfinder por cada una de las casillas candidatas.
        fields = {id(enemy): self._distance_field(predicted[id(enemy)])
                  for enemy in enemies}

        assignments = self._assign_attacks(allies, enemies, predicted)
        reserved_moves = set()
        for unit in allies:
            plan = assignments.get(id(unit))
            if plan is not None:
                _enemy, target_pos = plan
                self._issue_attack(unit, target_pos)
                continue

            destination = self._choose_move(unit, allies, enemies, predicted,
                                            fields, reserved_moves)
            # Reservar destinos de movimiento reduce acumulaciones de unidades.
            # No se reservan las posiciones actuales que se mantienen.
            if not self._same_pos(destination, unit.grid_position):
                reserved_moves.add((destination.x, destination.y))
            self._issue_move(unit, destination)

        return state


# Nombre explicito y alias corto para que se pueda importar de ambas maneras.
ManticoreAI = RivalAI

