"""Aegis 3.0: planificador de intercepción para OpenBattleEngine.

Mejoras frente a Aegis 2.0:
- Predice dónde terminarán los enemigos al ejecutar su ruta en este tick.
- Apunta a esa casilla prevista, importante porque el motor comprueba la
  coincidencia exacta entre objective y grid_position al resolver ataques.
- Cancela el movimiento residual antes de ordenar un ataque. UnitInstance.do_order
  no vacía moving_steps cuando recibe ATTACK_ORDER.
- Replanifica a un paso, con evaluación de fuego, cobertura, amenaza y flanqueo.
- Mantiene la interfaz execute_orders(state, ally=True); no cambia Pathfinder.
"""
from math import sqrt, atan2, pi
from common import GameData, Position
from order import Order, OrderType


class AegisAI:
    def __init__(self, name="Aegis 3.0"):
        self.name = name
        self._last_orders = {}

    @staticmethod
    def _dist(a, b):
        return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @staticmethod
    def _same_pos(a, b):
        return a is not None and b is not None and a.x == b.x and a.y == b.y

    @staticmethod
    def _alive(unit):
        return (getattr(unit, "current_health", 0) > 0 and
                getattr(unit, "current_cohesion", 0) > 0)

    @staticmethod
    def _health_ratio(unit):
        maximum = max(1.0, float(getattr(getattr(unit, "unit", None), "max_personnel", 1)))
        return max(0.0, min(1.0, float(getattr(unit, "current_health", 0)) / maximum))

    @staticmethod
    def _range(unit):
        try:
            return max(0.0, float(unit.mean_range()))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _cover(pos):
        try:
            return max(0.0, min(1.0, 1.0 - GameData.get_pos_combat_mod(pos.x, pos.y)))
        except (AttributeError, IndexError, TypeError):
            return 0.0

    @staticmethod
    def _passable(pos):
        try:
            return (0 <= pos.x < GameData.map_x_size and
                    0 <= pos.y < GameData.map_y_size and
                    GameData.get_pos_bias(pos.x, pos.y) >= 0)
        except (AttributeError, IndexError, TypeError):
            return False

    @staticmethod
    def _weapons_reach(unit, pos):
        """Comprueba alcances individuales; mean_range puede ocultar armas mixtas."""
        inventory = getattr(getattr(unit, "unit", None), "inventory", None)
        if not inventory:
            return AegisAI._dist(unit.grid_position, pos) <= AegisAI._range(unit)
        distance = AegisAI._dist(unit.grid_position, pos)
        for slot in inventory:
            try:
                if slot.quantity > 0 and slot.equipment.max_range >= distance:
                    return True
            except (AttributeError, TypeError):
                continue
        return False

    def _predict_position(self, enemy, blockers):
        """Simula el desplazamiento del enemigo con las reglas del motor este tick."""
        pos = enemy.grid_position
        route = list(getattr(enemy, "moving_steps", []) or [])
        if not route:
            return pos
        try:
            progress = float(getattr(enemy, "moving_progress", 0.0)) + (
                float(GameData.ticks_per_hour) *
                float(getattr(enemy.unit, "speed_kmh", 0.0))
            )
        except (AttributeError, TypeError, ValueError):
            return pos

        while progress >= 1.0 and route:
            next_pos = route.pop(0)
            # El motor resta 3 de progreso cuando entra en una casilla ocupada
            # por el rival, aunque todavía mueve la unidad a esa casilla.
            collisions = sum(1 for other in blockers
                             if self._alive(other) and other is not enemy and
                             self._same_pos(other.grid_position, next_pos))
            pos = next_pos
            progress -= 1.0 + 3.0 * collisions
            progress = max(0.0, progress)
        return pos

    def _simulate_hit(self, attacker, target, target_pos):
        old_pos = target.grid_position
        target.grid_position = target_pos
        try:
            hit = attacker.attack(target, offensive=True, sim=True)
            counter = target.attack(attacker, offensive=False, sim=True)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return None, None
        finally:
            target.grid_position = old_pos
        return hit, counter

    def _incoming_loss(self, unit, proposed_pos, enemies, predicted):
        """Estima fuego enemigo que el motor podría resolver contra esta casilla."""
        loss = 0.0
        own_hp = max(1.0, float(getattr(unit, "current_health", 1)))
        for enemy in enemies:
            enemy_pos = predicted.get(id(enemy), enemy.grid_position)
            objective = getattr(enemy, "objective", None)
            ordered_here = (getattr(enemy, "attacking", False) and
                            self._same_pos(objective, proposed_pos))
            collision_here = self._same_pos(enemy_pos, proposed_pos)
            if not ordered_here and not collision_here:
                continue
            old_own, old_enemy = unit.grid_position, enemy.grid_position
            unit.grid_position = proposed_pos
            enemy.grid_position = enemy_pos
            try:
                incoming = enemy.attack(unit, offensive=True, sim=True)
                loss += max(0.0, float(incoming.get("personnel_loss", 0))) / own_hp
                loss += 0.004 * max(0.0, float(incoming.get("cohesion_loss", 0)))
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                pass
            finally:
                unit.grid_position = old_own
                enemy.grid_position = old_enemy
        return loss

    def _attack_value(self, unit, enemy, target_pos, enemies, predicted):
        hit, counter = self._simulate_hit(unit, enemy, target_pos)
        if hit is None or counter is None:
            return -1000.0, None
        damage = max(0.0, float(hit.get("personnel_loss", 0)))
        cohesion = max(0.0, float(hit.get("cohesion_loss", 0)))
        morale = max(0.0, float(hit.get("morale_loss", 0)))
        counter_damage = max(0.0, float(counter.get("personnel_loss", 0)))
        target_hp = max(1.0, float(enemy.current_health))
        own_hp = max(1.0, float(unit.current_health))
        kill = (enemy.current_health - damage <= 0 or
                enemy.current_cohesion - cohesion <= 0)

        # El ataque se ejecuta contra cualquier enemigo que acabe en target_pos;
        # si no hay pérdida real ni supresión, no merece frenar para disparar.
        if damage <= 0 and cohesion <= 0 and morale <= 0:
            value = -3.0
        else:
            value = (
                5.0 * damage / target_hp
                + 0.025 * cohesion
                + 0.008 * morale
                - 2.0 * counter_damage / own_hp
                + (4.0 if kill else 0.0)
            )
        value -= 1.25 * self._incoming_loss(unit, unit.grid_position, enemies, predicted)
        return value, hit

    def _flank_bonus(self, pos, target_pos, allies, unit):
        """Prefiere ángulos de aproximación distintos a los de los aliados."""
        tx = pos.x - target_pos.x
        ty = pos.y - target_pos.y
        if tx == 0 and ty == 0:
            return 0.0
        angle = atan2(ty, tx)
        separations = []
        for other in allies:
            if other is unit or not self._alive(other):
                continue
            ox = other.grid_position.x - target_pos.x
            oy = other.grid_position.y - target_pos.y
            if ox == 0 and oy == 0:
                continue
            other_angle = atan2(oy, ox)
            delta = abs((angle - other_angle + pi) % (2 * pi) - pi)
            separations.append(delta / pi)
        return min(separations) if separations else 0.35

    def _move_score(self, unit, candidate, allies, enemies, predicted):
        cover = self._cover(candidate)
        mobility = 1.0
        try:
            mobility = max(0.0, float(GameData.get_pos_bias(candidate.x, candidate.y)))
        except (IndexError, TypeError):
            pass
        threat = 0.0
        fire_potential = 0.0
        best_target_dist = float("inf")
        flank = 0.0
        for enemy in enemies:
            enemy_pos = predicted.get(id(enemy), enemy.grid_position)
            dist = self._dist(candidate, enemy_pos)
            best_target_dist = min(best_target_dist, dist)
            weapon_range = self._range(enemy)
            if dist <= max(weapon_range, 1.5):
                threat += (0.45 + self._health_ratio(enemy)) / (0.75 + dist)
            old_own, old_enemy = unit.grid_position, enemy.grid_position
            unit.grid_position = candidate
            enemy.grid_position = enemy_pos
            try:
                # Si el movimiento permite disparar en la próxima evaluación,
                # el ataque potencial pesa más que una mejora pequeña de terreno.
                hit = unit.attack(enemy, offensive=True, sim=True)
                dmg = max(0.0, float(hit.get("personnel_loss", 0)))
                sup = max(0.0, float(hit.get("cohesion_loss", 0)))
                fire_potential = max(fire_potential,
                                     3.0 * dmg / max(1.0, enemy.current_health) + 0.018 * sup)
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                pass
            finally:
                unit.grid_position = old_own
                enemy.grid_position = old_enemy
            flank = max(flank, self._flank_bonus(candidate, enemy_pos, allies, unit))

        current_dist = min(self._dist(unit.grid_position, p)
                           for p in predicted.values()) if predicted else best_target_dist
        progress = current_dist - best_target_dist
        incoming = self._incoming_loss(unit, candidate, enemies, predicted)
        spacing = 0.0
        for other in allies:
            if other is unit or not self._alive(other):
                continue
            d = self._dist(candidate, other.grid_position)
            if d < 1.35:
                spacing += (1.35 - d) * 0.65

        return (1.7 * cover + 1.05 * fire_potential + 0.22 * progress
                + 0.55 * flank - 1.25 * threat - 1.10 * incoming
                - 0.16 * mobility - spacing)

    def _choose_move(self, unit, allies, enemies, predicted):
        start = unit.grid_position
        choices = []
        # Un paso real por tick: escoger vecinos evita puntuar un destino lejano
        # como si la unidad pudiera llegar antes de la fase de combate.
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            pos = Position(start.x + dx, start.y + dy)
            if not self._passable(pos):
                continue
            if any(a is not unit and self._alive(a) and
                   self._same_pos(a.grid_position, pos) for a in allies):
                continue
            # Evita colisionar con un enemigo en su casilla actual o prevista.
            if any(self._same_pos(e.grid_position, pos) or
                   self._same_pos(predicted.get(id(e)), pos) for e in enemies):
                continue
            choices.append(pos)

        if not choices:
            return start
        return max(choices, key=lambda pos: self._move_score(
            unit, pos, allies, enemies, predicted))

    def _select_attack(self, unit, enemies, predicted):
        best_enemy, best_pos, best_score = None, None, float("-inf")
        for enemy in enemies:
            target_pos = predicted.get(id(enemy), enemy.grid_position)
            if not self._weapons_reach(unit, target_pos):
                continue
            score, hit = self._attack_value(unit, enemy, target_pos, enemies, predicted)
            if hit is None:
                continue
            damage = max(0.0, float(hit.get("personnel_loss", 0)))
            suppression = max(0.0, float(hit.get("cohesion_loss", 0)))
            morale_loss = max(0.0, float(hit.get("morale_loss", 0)))
            # Rematar unidades debilitadas y reducir el número de armas enemigas.
            score += 0.75 * (1.0 - self._health_ratio(enemy))
            if enemy.current_health - damage <= 0 or enemy.current_cohesion - suppression <= 0:
                score += 2.0
            if damage <= 0 and suppression <= 0 and morale_loss <= 0:
                continue
            if score > best_score:
                best_enemy, best_pos, best_score = enemy, target_pos, score
        if best_score <= 0.0:
            return None
        return best_enemy, best_pos, best_score

    def _issue_attack(self, unit, destination):
        key = id(unit)
        previous = self._last_orders.get(key)
        if (previous == (OrderType.ATTACK_ORDER, destination.x, destination.y) and
                getattr(unit, "attacking", False) and
                not getattr(unit, "moving_steps", None) and
                self._same_pos(getattr(unit, "objective", None), destination)):
            return
        # Importante: do_order(ATTACK_ORDER) no limpia moving_steps en este motor.
        unit.moving_steps = []
        unit.moving_progress = 0.0
        unit.do_order(Order(OrderType.ATTACK_ORDER, destination))
        self._last_orders[key] = (OrderType.ATTACK_ORDER, destination.x, destination.y)

    def _issue_move(self, unit, destination):
        key = id(unit)
        previous = self._last_orders.get(key)
        if (previous == (OrderType.GOTO_ORDER, destination.x, destination.y) and
                getattr(unit, "moving_steps", None)):
            return
        unit.do_order(Order(OrderType.GOTO_ORDER, destination))
        self._last_orders[key] = (OrderType.GOTO_ORDER, destination.x, destination.y)

    def execute_orders(self, state, ally=True):
        my_units = state.a_units if ally else state.e_units
        enemy_units = state.e_units if ally else state.a_units
        allies = [u for u in my_units if self._alive(u)]
        enemies = [e for e in enemy_units if self._alive(e)]
        if not enemies:
            return state

        # Ai1 asigna sus órdenes antes que Ai2. Al ser el segundo, Aegis puede
        # leer moving_steps y anticipar la casilla final del rival en este tick.
        predicted = {id(enemy): self._predict_position(enemy, allies) for enemy in enemies}

        # Dar primero los ataques rentables para que la selección no dependa de
        # una unidad débil que aparezca al principio de la lista.
        attack_plans = []
        for unit in allies:
            plan = self._select_attack(unit, enemies, predicted)
            if plan is not None:
                attack_plans.append((plan[2], unit, plan[0], plan[1]))
        planned = {id(unit): (enemy, pos, score)
                   for score, unit, enemy, pos in sorted(attack_plans, reverse=True,
                                                          key=lambda item: item[0])}

        for unit in allies:
            plan = planned.get(id(unit))
            if plan is not None:
                enemy, target_pos, score = plan
                # Si la simulación anticipa un golpe útil, intercepta la posición
                # prevista en vez de perseguir la posición que ya va a abandonar.
                self._issue_attack(unit, target_pos)
            else:
                destination = self._choose_move(unit, allies, enemies, predicted)
                self._issue_move(unit, destination)

        return state
