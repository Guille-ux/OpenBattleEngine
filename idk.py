"""NeuralRival: rival basado en Deep Q-Learning (DQN) para OpenBattleEngine.

Dependencia: numpy

Acciones por unidad:
  0..2 atacar al enemigo mas cercano, segundo mas cercano o tercero mas cercano
  3..6 moverse una casilla al oeste, este, norte o sur
  7 cancelar orden y mantener la posicion

La politica se aprende con recompensas de equipo. No contiene pesos tacticos
hechos a mano; las unicas reglas externas son la legalidad de las acciones y
la representacion numerica del estado.
"""
from collections import deque
import os
import random

import numpy as np

from common import GameData, Position
from order import Order, OrderType

OBS_SIZE = 58
ACTION_COUNT = 8
ACTION_ATTACKS = 3
ACTION_MOVE_WEST = 3
ACTION_MOVE_EAST = 4
ACTION_MOVE_NORTH = 5
ACTION_MOVE_SOUTH = 6
ACTION_HOLD = 7
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _clip(value, low=-1.0, high=1.0):
    return max(low, min(high, float(value)))


def _safe_ratio(value, maximum):
    try:
        maximum = max(1.0, float(maximum))
        return _clip(float(value) / maximum, 0.0, 1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


class _MLP:
    """MLP 58 -> 96 -> 96 -> 8 con ReLU, Huber loss y Adam, usando NumPy."""

    def __init__(self, input_size=OBS_SIZE, hidden=96, output_size=ACTION_COUNT, seed=None):
        rng = np.random.default_rng(seed)
        self.weights = [
            (rng.standard_normal((input_size, hidden)) * np.sqrt(2.0 / input_size)).astype(np.float32),
            (rng.standard_normal((hidden, hidden)) * np.sqrt(2.0 / hidden)).astype(np.float32),
            (rng.standard_normal((hidden, output_size)) * np.sqrt(2.0 / hidden)).astype(np.float32),
        ]
        self.biases = [np.zeros(hidden, dtype=np.float32),
                       np.zeros(hidden, dtype=np.float32),
                       np.zeros(output_size, dtype=np.float32)]
        self.mw = [np.zeros_like(x) for x in self.weights]
        self.vw = [np.zeros_like(x) for x in self.weights]
        self.mb = [np.zeros_like(x) for x in self.biases]
        self.vb = [np.zeros_like(x) for x in self.biases]
        self.adam_t = 0

    def forward(self, x, cache=False):
        x = np.asarray(x, dtype=np.float32)
        z1 = x @ self.weights[0] + self.biases[0]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.weights[1] + self.biases[1]
        h2 = np.maximum(z2, 0.0)
        out = h2 @ self.weights[2] + self.biases[2]
        if cache:
            return out, (x, z1, h1, z2, h2)
        return out

    def train_batch(self, states, actions, targets, learning_rate=0.0005):
        q_values, cache = self.forward(states, cache=True)
        x, z1, h1, z2, h2 = cache
        rows = np.arange(len(actions))
        chosen = q_values[rows, actions]
        error = chosen - targets

        # Derivada de Huber; normalizada por el tamaño del minibatch.
        dq_selected = np.clip(error, -1.0, 1.0) / max(1, len(actions))
        dq = np.zeros_like(q_values)
        dq[rows, actions] = dq_selected

        gw2 = h2.T @ dq
        gb2 = dq.sum(axis=0)
        dh2 = dq @ self.weights[2].T
        dz2 = dh2 * (z2 > 0.0)

        gw1 = h1.T @ dz2
        gb1 = dz2.sum(axis=0)
        dh1 = dz2 @ self.weights[1].T
        dz1 = dh1 * (z1 > 0.0)

        gw0 = x.T @ dz1
        gb0 = dz1.sum(axis=0)
        gws = [gw0, gw1, gw2]
        gbs = [gb0, gb1, gb2]

        # Recorte global para reducir explosiones de gradiente.
        norm_sq = sum(float(np.sum(g * g)) for g in gws + gbs)
        norm = np.sqrt(norm_sq)
        if norm > 5.0:
            scale = 5.0 / (norm + 1e-8)
            gws = [g * scale for g in gws]
            gbs = [g * scale for g in gbs]

        self.adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        correction1 = 1.0 - beta1 ** self.adam_t
        correction2 = 1.0 - beta2 ** self.adam_t
        for i in range(3):
            self.mw[i] = beta1 * self.mw[i] + (1.0 - beta1) * gws[i]
            self.vw[i] = beta2 * self.vw[i] + (1.0 - beta2) * (gws[i] * gws[i])
            self.mb[i] = beta1 * self.mb[i] + (1.0 - beta1) * gbs[i]
            self.vb[i] = beta2 * self.vb[i] + (1.0 - beta2) * (gbs[i] * gbs[i])
            mw_hat = self.mw[i] / correction1
            vw_hat = self.vw[i] / correction2
            mb_hat = self.mb[i] / correction1
            vb_hat = self.vb[i] / correction2
            self.weights[i] -= learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
            self.biases[i] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

        return float(np.mean(np.where(np.abs(error) < 1.0,
                                      0.5 * error * error,
                                      np.abs(error) - 0.5)))

    def copy_from(self, other, include_optimizer=False):
        self.weights = [x.copy() for x in other.weights]
        self.biases = [x.copy() for x in other.biases]
        if include_optimizer:
            self.mw = [x.copy() for x in other.mw]
            self.vw = [x.copy() for x in other.vw]
            self.mb = [x.copy() for x in other.mb]
            self.vb = [x.copy() for x in other.vb]
            self.adam_t = other.adam_t


class NeuralRival:
    """Agente DQN para usar como player1 o player2 en OpenBattleEngine.

    training=True activa epsilon-greedy, replay buffer y aprendizaje en linea.
    Para combatir con un modelo ya entrenado, usar model_path y training=False.
    """

    def __init__(self, model_path=None, training=False, seed=None,
                 replay_size=30000, batch_size=64, gamma=0.96,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9997,
                 learning_rate=0.0005, target_sync=250):
        self.name = "NeuralRival-DQN"
        self.training = bool(training)
        self.online = _MLP(seed=seed)
        self.target = _MLP(seed=None if seed is None else seed + 1)
        self.target.copy_from(self.online)
        self.replay = deque(maxlen=int(replay_size))
        self.batch_size = max(8, int(batch_size))
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start if training else 0.0)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.learning_rate = float(learning_rate)
        self.target_sync = max(1, int(target_sync))
        self.learn_steps = 0
        self.last_loss = 0.0
        self._pending = []
        self._last_material = None
        self._initial_total = 6.0
        self._last_side = False
        self._orders = {}
        self.episodes = 0
        if model_path:
            self.load(model_path, inference=not self.training)

    @staticmethod
    def _alive(unit):
        return (float(getattr(unit, "current_health", 0.0)) > 0.0 and
                not (float(getattr(unit, "current_morale", 0.0)) >= 0.0 and
                     float(getattr(unit, "current_cohesion", 0.0)) <= 0.0))

    @staticmethod
    def _max_health(unit):
        return max(1.0, float(getattr(getattr(unit, "unit", None),
                                     "max_personnel", 1.0)))

    @staticmethod
    def _dist(a, b):
        dx = float(a.x) - float(b.x)
        dy = float(a.y) - float(b.y)
        return float(np.sqrt(dx * dx + dy * dy))

    @staticmethod
    def _cover(pos):
        try:
            return _clip(1.0 - float(GameData.get_pos_combat_mod(pos.x, pos.y)), 0.0, 1.0)
        except (AttributeError, IndexError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _bias(pos):
        try:
            # -1 identifica terreno intransitable; 0..3 escala el coste de movilidad.
            return _clip(float(GameData.get_pos_bias(pos.x, pos.y)) / 3.0, -1.0, 1.0)
        except (AttributeError, IndexError, TypeError, ValueError):
            return -1.0

    @classmethod
    def _passable(cls, pos):
        return (0 <= pos.x < GameData.map_x_size and
                0 <= pos.y < GameData.map_y_size and cls._bias(pos) >= 0.0)

    @staticmethod
    def _mean_range(unit):
        try:
            return max(0.0, float(unit.mean_range()))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 0.0

    @classmethod
    def _in_range(cls, unit, enemy_pos):
        dist = cls._dist(unit.grid_position, enemy_pos)
        inventory = getattr(getattr(unit, "unit", None), "inventory", None)
        if inventory:
            for slot in inventory:
                try:
                    if slot.quantity > 0 and float(slot.equipment.max_range) >= dist:
                        return True
                except (AttributeError, TypeError, ValueError):
                    continue
            return False
        return dist <= cls._mean_range(unit)

    @staticmethod
    def _sides(state, ally):
        if ally:
            return list(state.a_units), list(state.e_units)
        return list(state.e_units), list(state.a_units)

    def _material(self, state, ally):
        ours, theirs = self._sides(state, ally)

        def total(units):
            score = 0.0
            for u in units:
                hp = _safe_ratio(u.current_health, self._max_health(u))
                cohesion = _safe_ratio(u.current_cohesion, 100.0)
                morale = _safe_ratio(u.current_morale, 100.0)
                # El término fijo cuenta una unidad superviviente; los demás
                # términos reflejan su capacidad de combate restante.
                score += 0.30 + 0.45 * hp + 0.15 * cohesion + 0.10 * morale
            return score

        return (total(ours) - total(theirs)) / max(1.0, self._initial_total)

    def _obs(self, unit, allies, enemies):
        max_x = max(1.0, float(GameData.map_x_size - 1))
        max_y = max(1.0, float(GameData.map_y_size - 1))
        health_max = self._max_health(unit)
        speed = float(getattr(getattr(unit, "unit", None), "speed_kmh", 0.0))
        pos = unit.grid_position
        obs = [
            _safe_ratio(unit.current_health, health_max),
            _safe_ratio(unit.current_cohesion, 100.0),
            _safe_ratio(unit.current_morale, 100.0),
            _clip(pos.x / max_x, 0.0, 1.0),
            _clip(pos.y / max_y, 0.0, 1.0),
            _clip(speed / 10.0, 0.0, 2.0),
            _clip(self._mean_range(unit) / 16.0, 0.0, 1.0),
            float(bool(getattr(unit, "attacking", False))),
            float(bool(getattr(unit, "moving_steps", []))),
            _clip(float(getattr(unit, "moving_progress", 0.0)), 0.0, 1.0),
        ]

        sorted_enemies = sorted(enemies, key=lambda e: self._dist(pos, e.grid_position))
        for enemy in sorted_enemies[:3]:
            ep = enemy.grid_position
            dx, dy = ep.x - pos.x, ep.y - pos.y
            dist = self._dist(pos, ep)
            obs.extend([
                _clip(dx / max_x, -1.0, 1.0),
                _clip(dy / max_y, -1.0, 1.0),
                _clip(dist / 16.0, 0.0, 1.0),
                _safe_ratio(enemy.current_health, self._max_health(enemy)),
                _safe_ratio(enemy.current_cohesion, 100.0),
                _safe_ratio(enemy.current_morale, 100.0),
                float(self._in_range(unit, ep)),
                float(bool(getattr(enemy, "moving_steps", []))),
            ])
        for _ in range(3 - min(3, len(sorted_enemies))):
            obs.extend([0.0] * 8)

        sorted_allies = sorted((a for a in allies if a is not unit),
                               key=lambda a: self._dist(pos, a.grid_position))
        for friend in sorted_allies[:2]:
            fp = friend.grid_position
            dx, dy = fp.x - pos.x, fp.y - pos.y
            obs.extend([
                _clip(dx / max_x, -1.0, 1.0),
                _clip(dy / max_y, -1.0, 1.0),
                _clip(self._dist(pos, fp) / 16.0, 0.0, 1.0),
                _safe_ratio(friend.current_health, self._max_health(friend)),
                float(bool(getattr(friend, "attacking", False))),
            ])
        for _ in range(2 - min(2, len(sorted_allies))):
            obs.extend([0.0] * 5)

        obs.extend([self._cover(pos), self._bias(pos)])
        occupied = [a.grid_position for a in allies + enemies if self._alive(a)]
        for dx, dy in DIRECTIONS:
            p = Position(pos.x + dx, pos.y + dy)
            in_bounds = (0 <= p.x < GameData.map_x_size and
                         0 <= p.y < GameData.map_y_size)
            passable = in_bounds and self._passable(p)
            blocked = any(p.x == q.x and p.y == q.y for q in occupied)
            obs.extend([float(passable and not blocked),
                        self._cover(p) if in_bounds else 0.0,
                        float(blocked)])

        if len(obs) != OBS_SIZE:
            raise RuntimeError(f"Observacion DQN incorrecta: {len(obs)} != {OBS_SIZE}")
        return np.asarray(obs, dtype=np.float32)

    def _legal_mask(self, unit, allies, enemies):
        mask = np.zeros(ACTION_COUNT, dtype=bool)
        sorted_enemies = sorted(enemies, key=lambda e: self._dist(unit.grid_position, e.grid_position))
        for i, enemy in enumerate(sorted_enemies[:ACTION_ATTACKS]):
            if self._alive(enemy) and self._in_range(unit, enemy.grid_position):
                mask[i] = True

        occupied = [(u.grid_position.x, u.grid_position.y)
                    for u in allies + enemies if self._alive(u) and u is not unit]
        x, y = unit.grid_position.x, unit.grid_position.y
        for i, (dx, dy) in enumerate(DIRECTIONS):
            p = Position(x + dx, y + dy)
            in_bounds = (0 <= p.x < GameData.map_x_size and
                         0 <= p.y < GameData.map_y_size)
            if in_bounds and self._passable(p) and (p.x, p.y) not in occupied:
                mask[ACTION_MOVE_WEST + i] = True
        mask[ACTION_HOLD] = True
        return mask

    def _choose(self, obs, mask):
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return ACTION_HOLD
        if self.training and random.random() < self.epsilon:
            return int(random.choice(legal.tolist()))
        values = self.online.forward(obs.reshape(1, -1))[0]
        values = values.copy()
        values[~mask] = -1e9
        return int(np.argmax(values))

    def _issue(self, unit, action, allies, enemies):
        key = id(unit)
        if action < ACTION_ATTACKS:
            targets = sorted(enemies, key=lambda e: self._dist(unit.grid_position, e.grid_position))[:3]
            if action >= len(targets):
                return
            target = targets[action]
            signature = ("attack", target.grid_position.x, target.grid_position.y)
            if (self._orders.get(key) == signature and unit.attacking and
                    not getattr(unit, "moving_steps", None) and
                    getattr(getattr(unit, "objective", None), "x", None) == target.grid_position.x and
                    getattr(getattr(unit, "objective", None), "y", None) == target.grid_position.y):
                return
            # En el motor ATTACK_ORDER no borra moving_steps, asi que lo hacemos
            # aqui para que la accion aprendida tenga una semantica consistente.
            unit.moving_steps = []
            unit.moving_progress = 0.0
            unit.do_order(Order(OrderType.ATTACK_ORDER, Position(target.grid_position.x,
                                                                target.grid_position.y)))
            self._orders[key] = signature
        elif ACTION_MOVE_WEST <= action <= ACTION_MOVE_SOUTH:
            dx, dy = DIRECTIONS[action - ACTION_MOVE_WEST]
            dest = Position(unit.grid_position.x + dx, unit.grid_position.y + dy)
            signature = ("move", dest.x, dest.y)
            if self._orders.get(key) == signature and getattr(unit, "moving_steps", None):
                return
            unit.do_order(Order(OrderType.GOTO_ORDER, dest))
            self._orders[key] = signature
        else:
            signature = ("hold", unit.grid_position.x, unit.grid_position.y)
            if self._orders.get(key) != signature and (
                    getattr(unit, "moving_steps", None) or getattr(unit, "attacking", False)):
                unit.moving_steps = []
                unit.moving_progress = 0.0
                unit.do_order(Order(OrderType.CANCEL_ORDER, None))
            self._orders[key] = signature

    def _learn(self):
        if not self.training or len(self.replay) < self.batch_size:
            return
        batch = random.sample(self.replay, self.batch_size)
        states = np.stack([item[0] for item in batch]).astype(np.float32)
        actions = np.asarray([item[1] for item in batch], dtype=np.int64)
        rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
        next_states = np.stack([item[3] for item in batch]).astype(np.float32)
        dones = np.asarray([item[4] for item in batch], dtype=np.float32)
        next_masks = np.stack([item[5] for item in batch]).astype(bool)

        next_q = self.target.forward(next_states)
        next_q[~next_masks] = -1e9
        # Terminales no tienen bootstrap.
        next_q[dones > 0.5] = 0.0
        targets = rewards + self.gamma * (1.0 - dones) * np.max(next_q, axis=1)
        self.last_loss = self.online.train_batch(states, actions, targets,
                                                 learning_rate=self.learning_rate)
        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.copy_from(self.online)

    def _complete_pending(self, state, reward, terminal, ally):
        if not self._pending:
            return
        allies, enemies = self._sides(state, ally)
        alive_ids = {id(u) for u in allies if self._alive(u)}
        for unit, obs, action in self._pending:
            unit_terminal = bool(terminal or id(unit) not in alive_ids)
            if unit_terminal:
                next_obs = np.zeros(OBS_SIZE, dtype=np.float32)
                next_mask = np.zeros(ACTION_COUNT, dtype=bool)
                next_mask[ACTION_HOLD] = True
            else:
                next_obs = self._obs(unit, allies, enemies)
                next_mask = self._legal_mask(unit, allies, enemies)
            self.replay.append((obs, int(action), float(_clip(reward, -2.0, 2.0)),
                                next_obs, float(unit_terminal), next_mask))
        self._pending = []
        self._learn()

    def execute_orders(self, state, ally=True):
        allies, enemies = self._sides(state, ally)
        allies = [u for u in allies if self._alive(u)]
        enemies = [u for u in enemies if self._alive(u)]
        self._last_side = bool(ally)

        if self._initial_total <= 0 or self._initial_total == 6.0 and self._last_material is None:
            self._initial_total = float(max(1, len(allies) + len(enemies)))
        material_now = self._material(state, ally)
        if self._last_material is None:
            reward = 0.0
        else:
            reward = _clip((material_now - self._last_material) * 6.0, -1.0, 1.0)
            self._complete_pending(state, reward, bool(getattr(state, "end", False)), ally)
        self._last_material = material_now

        if not allies or not enemies:
            return state

        pending = []
        for unit in allies:
            obs = self._obs(unit, allies, enemies)
            mask = self._legal_mask(unit, allies, enemies)
            action = self._choose(obs, mask)
            self._issue(unit, action, allies, enemies)
            pending.append((unit, obs, action))
            if self.training:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.training:
            self._pending = pending
        return state

    def finish_episode(self, won, state=None, ally=None):
        """Cierra el episodio y acredita victoria/derrota a las acciones pendientes.

        won debe estar expresado desde la perspectiva de NeuralRival.
        En el motor actual, si NeuralRival es player2/Ai2, won = state.who_won.
        """
        if not self.training:
            self._pending = []
            self._last_material = None
            return
        if ally is None:
            ally = self._last_side
        terminal_reward = 1.25 if won else -1.25
        if state is not None:
            current_material = self._material(state, ally)
            delta = 0.0 if self._last_material is None else (current_material - self._last_material) * 6.0
            terminal_reward += _clip(delta, -1.0, 1.0)
            self._complete_pending(state, terminal_reward, True, ally)
        else:
            # Sin estado final, aun se puede cerrar el buffer, aunque se pierde
            # la recompensa material del ultimo tick.
            dummy_state = None
            for obs, action in []:  # no-op explicito: no inventar estados terminales
                pass
            if self._pending:
                for _unit, obs, action in self._pending:
                    self.replay.append((obs, int(action), float(terminal_reward),
                                        np.zeros(OBS_SIZE, dtype=np.float32), 1.0,
                                        np.eye(1, ACTION_COUNT, ACTION_HOLD, dtype=bool)[0]))
                self._pending = []
                self._learn()
        self.episodes += 1
        self._last_material = None
        self._initial_total = 6.0
        self._orders.clear()

    def save(self, path="neural_rival.npz"):
        arrays = {
            "w0": self.online.weights[0], "w1": self.online.weights[1], "w2": self.online.weights[2],
            "b0": self.online.biases[0], "b1": self.online.biases[1], "b2": self.online.biases[2],
            "tw0": self.target.weights[0], "tw1": self.target.weights[1], "tw2": self.target.weights[2],
            "tb0": self.target.biases[0], "tb1": self.target.biases[1], "tb2": self.target.biases[2],
            "epsilon": np.asarray([self.epsilon], dtype=np.float32),
            "learn_steps": np.asarray([self.learn_steps], dtype=np.int64),
            "episodes": np.asarray([self.episodes], dtype=np.int64),
            "adam_t": np.asarray([self.online.adam_t], dtype=np.int64),
        }
        for i in range(3):
            arrays[f"mw{i}"] = self.online.mw[i]
            arrays[f"vw{i}"] = self.online.vw[i]
            arrays[f"mb{i}"] = self.online.mb[i]
            arrays[f"vb{i}"] = self.online.vb[i]
        np.savez_compressed(path, **arrays)

    def load(self, path, inference=True):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            for i in range(3):
                self.online.weights[i] = data[f"w{i}"].astype(np.float32)
                self.online.biases[i] = data[f"b{i}"].astype(np.float32)
                self.target.weights[i] = data[f"tw{i}"].astype(np.float32) if f"tw{i}" in data else self.online.weights[i].copy()
                self.target.biases[i] = data[f"tb{i}"].astype(np.float32) if f"tb{i}" in data else self.online.biases[i].copy()
                for name in ("mw", "vw", "mb", "vb"):
                    key = f"{name}{i}"
                    if key in data:
                        getattr(self.online, name)[i] = data[key].astype(np.float32)
            if "adam_t" in data:
                self.online.adam_t = int(data["adam_t"][0])
            self.learn_steps = int(data["learn_steps"][0]) if "learn_steps" in data else 0
            self.episodes = int(data["episodes"][0]) if "episodes" in data else 0
            self.epsilon = 0.0 if inference else float(data["epsilon"][0]) if "epsilon" in data else 1.0
        if inference:
            self.training = False
        return self

    def diagnostics(self):
        return {
            "name": self.name,
            "training": self.training,
            "epsilon": round(float(self.epsilon), 4),
            "replay_size": len(self.replay),
            "episodes": self.episodes,
            "learn_steps": self.learn_steps,
            "last_loss": round(float(self.last_loss), 6),
        }


# Alias para que el nombre sea facil de cambiar en los scripts de torneo.
DQNCombatAI = NeuralRival


# ---------------------------------------------------------------------------
# Entrenamiento y evaluacion integrados en este mismo archivo.
# ---------------------------------------------------------------------------
import argparse
import time
from ai import AIManager, ModelGenome
from common import Position
from engine import BattleState, OpenBattleEngine
from units import UnitData, UnitInstance

DEFAULT_OPPONENT = {
    "name": "Gen0_Bot_23_mut_mut_mut_mut_mut",
    "weight_damage": 0.6462275643312343,
    "weight_finish_off": 6.99535251396952,
    "weight_supression": 2.2347552582584553,
    "weight_flanking": 1.665354917460618,
    "weight_cover": 4.273852360563043,
    "weight_optimal_range": 3.108534491599299,
    "penalty_crowding": 1.6983043740001817,
    "aggression_bias": 3.0,
    "positional": 4.602419557731989,
    "health_bias": -0.4020241084449241,
    "cohesion_bias": -0.08914261221592673,
    "morale_bias": -0.3907413852419418,
    "speed_bias": 1.3002492466968072,
    "range_bias": 0.571087799210872
}


def make_opponent(json_path=None):
    """Carga un genoma desde JSON o usa el campeón IGA integrado."""
    if json_path:
        return AIManager(ModelGenome.load_from_json(json_path))
    genome = ModelGenome(DEFAULT_OPPONENT["name"])
    for key, value in DEFAULT_OPPONENT.items():
        if key != "name" and hasattr(genome, key):
            setattr(genome, key, value)
    return AIManager(genome)


def build_state(rng):
    """Crea una batalla 3 contra 3 como la configuración de referencia."""
    left = [UnitInstance(UnitData(), f"IGA_{i}", Position(0, i)) for i in range(3)]
    right = [UnitInstance(UnitData(), f"DQN_{i}", Position(11, i)) for i in range(3)]
    rng.shuffle(left)
    rng.shuffle(right)
    return BattleState(left, right)


def remaining_strength(units):
    score = 0.0
    for unit in units:
        max_hp = max(1.0, float(unit.unit.max_personnel))
        score += (0.30 + 0.45 * max(0.0, min(1.0, unit.current_health / max_hp))
                  + 0.15 * max(0.0, min(1.0, unit.current_cohesion / 100.0))
                  + 0.10 * max(0.0, min(1.0, unit.current_morale / 100.0)))
    return score


def reset_agent_episode(agent):
    agent._pending = []
    agent._last_material = None
    agent._initial_total = 6.0
    agent._orders.clear()


def play_episode(opponent, agent, max_ticks, seed, train=True):
    random.seed(seed)
    np.random.seed(seed)
    reset_agent_episode(agent)
    state = build_state(random)
    ticks = 0
    while not state.end and ticks < max_ticks:
        state = OpenBattleEngine.run_tick(state, opponent, agent)
        ticks += 1
    if state.end:
        won = bool(state.who_won)  # NeuralRival ocupa Ai2.
    else:
        won = remaining_strength(state.e_units) > remaining_strength(state.a_units)
    if train:
        agent.finish_episode(won, state, ally=False)
    return won, ticks, len(state.e_units), len(state.a_units)


def train(args):
    if args.episodes < 1 or args.max_ticks < 1:
        raise ValueError("episodes y max-ticks deben ser mayores que cero")
    opponent = make_opponent(args.opponent)
    agent = NeuralRival(training=True, seed=args.seed, batch_size=args.batch_size)
    if args.resume:
        try:
            agent.load(args.model, inference=False)
            agent.training = True
            print(f"Continuando desde {args.model}; epsilon={agent.epsilon:.3f}")
        except FileNotFoundError:
            print(f"No existe {args.model}; se inicia un modelo nuevo.")
    recent, started = [], time.perf_counter()
    print("=== NEURALRIVAL DQN: entrenamiento contra IGA ===")
    print(f"Oponente: {opponent.genome.name} | episodios: {args.episodes} | ticks max.: {args.max_ticks}")
    print("La red comienza con parámetros aleatorios; no contiene pesos tácticos precargados.\n")
    for episode in range(1, args.episodes + 1):
        won, ticks, ours, theirs = play_episode(opponent, agent, args.max_ticks,
                                                args.seed + episode, train=True)
        recent.append(int(won))
        if len(recent) > 25:
            recent.pop(0)
        if episode % max(1, args.log_every) == 0:
            agent.save(args.model)
            rate = 100.0 * sum(recent) / len(recent)
            print(f"Ep {episode:4d}/{args.episodes} | victorias ventana={rate:5.1f}% "
                  f"({sum(recent)}/{len(recent)}) | ticks={ticks:3d} "
                  f"| supervivientes DQN/IGA={ours}/{theirs} | eps={agent.epsilon:.3f} "
                  f"| loss={agent.last_loss:.4f} | {time.perf_counter()-started:.1f}s")
    agent.save(args.model)
    print(f"\nEntrenamiento terminado. Modelo guardado en {args.model}")
    print(f"Diagnóstico: {agent.diagnostics()}")
    print("Valida con el comando 'evaluate': las victorias de entrenamiento incluyen exploración.")


def evaluate(args):
    if args.games < 1 or args.max_ticks < 1:
        raise ValueError("games y max-ticks deben ser mayores que cero")
    opponent = make_opponent(args.opponent)
    agent = NeuralRival(model_path=args.model, training=False)
    wins = ticks_total = ours_total = theirs_total = draws = 0
    for game in range(args.games):
        won, ticks, ours, theirs = play_episode(opponent, agent, args.max_ticks,
                                                args.seed + game, train=False)
        # Los supervivientes se usan para el informe; el resultado lo decide el motor o la fuerza final.
        wins += int(won)
        ticks_total += ticks
        ours_total += ours
        theirs_total += theirs
        print(f"Partida {game+1:3d}/{args.games}: {'VICTORIA' if won else 'derrota/empate'} "
              f"| ticks={ticks:3d} | supervivientes DQN/IGA={ours}/{theirs}")
    print("\n=== EVALUACIÓN SIN EXPLORACIÓN ===")
    print(f"IGA: {opponent.genome.name}")
    print(f"Victorias NeuralRival: {wins}/{args.games} ({100.0*wins/args.games:.1f}%)")
    print(f"Ticks promedio: {ticks_total/args.games:.1f}")
    print(f"Supervivientes promedio DQN/IGA: {ours_total/args.games:.2f}/{theirs_total/args.games:.2f}")
    print(f"Empates o derrotas: {args.games-wins}; empates exactos pueden depender del criterio del motor.")


def main():
    parser = argparse.ArgumentParser(description="NeuralRival: DQN en un solo archivo")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="Entrenar o continuar entrenando")
    p_train.add_argument("--opponent", default=None, help="JSON opcional del genoma IGA; por defecto usa Gen0_Bot_23 integrado")
    p_train.add_argument("--episodes", type=int, default=300)
    p_train.add_argument("--max-ticks", type=int, default=80)
    p_train.add_argument("--model", default="neural_rival.npz")
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--seed", type=int, default=20261011)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--log-every", type=int, default=1)
    p_train.set_defaults(func=train)
    p_eval = sub.add_parser("evaluate", help="Evaluar un modelo sin exploración")
    p_eval.add_argument("--opponent", default=None, help="JSON opcional del genoma IGA")
    p_eval.add_argument("--model", default="neural_rival.npz")
    p_eval.add_argument("--games", type=int, default=30)
    p_eval.add_argument("--max-ticks", type=int, default=80)
    p_eval.add_argument("--seed", type=int, default=90125)
    p_eval.set_defaults(func=evaluate)
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
