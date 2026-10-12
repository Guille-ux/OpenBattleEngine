### WARNING
this documentation was ai generated, i obviously checked for mistakes on it, but i can still fail

# OpenBattle Engine - Referencia de Clases Base y Estructurales

Este documento recopila la referencia técnica de las clases base y el núcleo funcional que sustentan el motor de simulación de combate y la IA genética (`IGA`).

---

## 1. Entorno y Coordenadas (`common.py`)

### `class Position`[cite: 4]
* **Atributos de instancia:**
  * `self.x`: Coordenada horizontal en la cuadrícula[cite: 4].
  * `self.y`: Coordenada vertical en la cuadrícula[cite: 4].
* **Métodos:**
  * `__init__(self, x, y)`: Inicializa la posición asignando `self.x` y `self.y`[cite: 4].
  * `__eq__(self, _pos)`: Comprueba si `_pos.x == self.x` y `_pos.y == self.y`[cite: 4].
  * `__hash__(self)`: Retorna un hash basado en la tupla `(self.x, self.y)`[cite: 4].

### `class GameData`[cite: 4]
* **Atributos de clase:**
  * `ticks_per_hour = 0.25`: Fracción de hora que avanza por cada tick[cite: 4].
  * `map_x_size = 12`: Ancho máximo del mapa en celdas[cite: 4].
  * `map_y_size = 12`: Alto máximo del mapa en celdas[cite: 4].
  * `map_data = Map(terrain_matrix_12x12)`: Instancia del mapa cargada con la matriz de 12x12[cite: 4].
* **Métodos estáticos:**
  * `vec2cell(x, y)`: Llama internamente a `get_cell_id(x, y)`[cite: 4].
  * `get_cell_id(x, y)`: Calcula y retorna el ID numérico único de la celda mediante la fórmula `y + x * GameData.map_y_size + 1`[cite: 4].
  * `cell2vec(cell_id)`: Transforma un ID de celda en un objeto `Position` usando división entera y módulo respecto a `map_y_size`[cite: 4].
  * `get_pos_bias(x, y)`: Obtiene el terreno en `map_data.map_data[y][x]`; si no es transitable devuelve `-1`, de lo contrario retorna `tile.fatigue_mod`[cite: 4].
  * `get_pos_combat_mod(x, y)`: Calcula el modificador de combate restando la cobertura del terreno de `1.0` (`1.0 - GameData.map_data.map_data[y][x].cover`)[cite: 4].

---

## 2. Terreno y Equipamiento (`terrain.py` y `equipment.py`)

### `class Terrain` (`terrain.py`)[cite: 10]
* **Atributos de instancia:**
  * `self.id`: Identificador único del tipo de terreno (ej. `"plains_tile"`)[cite: 10].
  * `self.name`: Nombre descriptivo (ej. `"Plains"`, `"Forest"`, `"Mountain"`, `"Road"`)[cite: 10].
  * `self.fatigue_mod`: Modificador de fatiga y coste de movimiento[cite: 10].
  * `self.is_passable`: Booleano que indica si las unidades pueden cruzar la casilla[cite: 10].
  * `self.mobility_mod`: Modificador de movilidad (fijado por defecto en `1.0`)[cite: 10].
  * `self.cover`: Valor decimal que representa el porcentaje de cobertura defensiva[cite: 10].
* **Métodos:**
  * `__init__(self, tid="plains_tile", name="Plains", fatigue=1.0, mobility_mod=1.0, cover=0.0, is_passable=True)`: Inicializa los atributos anteriores[cite: 10].

### `class Map` (`terrain.py`)[cite: 10]
* **Atributos de instancia:**
  * `self.map_data`: Matriz bidimensional que almacena los objetos `Terrain`[cite: 10].
  * `self.height`: Altura de la matriz obtenida mediante `len(terrain_matrix)`[cite: 10].
  * `self.width`: Ancho de la matriz obtenido mediante `len(terrain_matrix[0])`[cite: 10].
* **Métodos:**
  * `__init__(self, terrain_matrix)`: Configura el mapa y sus dimensiones espaciales[cite: 10].

### `class EquipmentSlot` (`equipment.py`)[cite: 6]
* **Atributos de instancia:**
  * `self.equipment`: Referencia a la clase base del equipamiento/arma[cite: 6].
  * `self.quantity`: Cantidad disponible de dicho equipo en el inventario[cite: 6].
* **Métodos:**
  * `__init__(self, equipment, quantity)`: Asigna el equipo y su cantidad[cite: 6].

### `class BaseEquipment` (`equipment.py`)[cite: 6]
* **Atributos de clase (Plantilla de Arma):**
  * `name = "Karabiner 98K"`[cite: 6]
  * `max_range = 1.5`[cite: 6]
  * `breakthrough = 3`[cite: 6]
  * `soft_attack = 8`[cite: 6]
  * `hard_attack = 1`[cite: 6]
  * `supression = 3`[cite: 6]

---

## 3. Órdenes y Navegación (`order.py` y `pathfinder.py`)

### `class OrderType` (`order.py`)[cite: 8]
* **Constantes estáticas:**
  * `GOTO_ORDER = 0`[cite: 8]
  * `ATTACK_ORDER = 1`[cite: 8]
  * `CANCEL_ORDER = 2`[cite: 8]
  * `REPORT_ORDER = 3`[cite: 8]

### `class Order` (`order.py`)[cite: 8]
* **Atributos de instancia:**
  * `self.type`: Tipo de orden basada en `OrderType`[cite: 8].
  * `self.arg`: Argumento o destino asociado a la orden (habitualmente un objeto `Position`)[cite: 8].
* **Métodos:**
  * `__init__(self, type, arg)`: Inicializa el tipo y el argumento[cite: 8].

### `class Path` (`pathfinder.py`)[cite: 9]
* **Atributos de instancia:**
  * `self.cost`: Coste acumulado de desplazamiento a lo largo del camino[cite: 9].
  * `self.steps`: Lista de objetos `Position` que componen los pasos del trayecto[cite: 9].
* **Métodos:**
  * `__init__(self, p_coste=100000000000, p_pasos=[])`: Inicializa el coste y los pasos[cite: 9].
  * `clone_advance(self, new_pos, cost)`: Duplica el camino actual sumando el nuevo `cost` y añadiendo `new_pos` a una copia de `steps`[cite: 9].

### `class Pathfinder` (`pathfinder.py`)[cite: 9]
* **Métodos estáticos:**
  * `find_path(start_pos, end_pos)`: Implementa el algoritmo de búsqueda de rutas óptimas utilizando un diccionario de costes totales y una lista de caminos abiertos[cite: 9].
  * `get_directions(pos)`: Examina las 4 direcciones adyacentes válidas respetando los límites del mapa[cite: 9].

---

## 4. Unidades y Balística (`units.py`)

### `class UnitData`[cite: 12]
* **Atributos de clase:**
  * `uid = "ger_infantry"`[cite: 12]
  * `name = "German Infantry Line"`[cite: 12]
  * `max_personnel = 800`[cite: 12]
  * `speed_kmh = 4.0`[cite: 12]
  * `base_morale = 100`[cite: 12]
  * `base_cohesion = 100`[cite: 12]
  * `detection_radius = 3`[cite: 12]
  * `softness = 1`[cite: 12]
  * `defence = 30`[cite: 12]
  * `inventory = [EquipmentSlot(BaseEquipment, 700)]`[cite: 12]

### `class UnitInstance`[cite: 12]
* **Atributos de instancia:**
  * `self.unit`: Referencia al objeto `UnitData` base[cite: 12].
  * `self.name`: Nombre identificativo de la instancia[cite: 12].
  * `self.grid_position`: Objeto `Position` con la ubicación actual en la rejilla[cite: 12].
  * `self.current_morale`: Nivel actual de moral[cite: 12].
  * `self.current_cohesion`: Nivel actual de cohesión[cite: 12].
  * `self.current_health`: Personal activo actual[cite: 12].
  * `self.objective`: Posición objetivo asignada[cite: 12].
  * `self.attacking`: Booleano de estado de combate activo[cite: 12].
  * `self.moving_steps`: Lista de pasos pendientes generada por el pathfinder[cite: 12].
  * `self.moving_progress`: Contador flotante de progreso de desplazamiento[cite: 12].
* **Métodos principales:**
  * `do_order(self, order)`: Procesa órdenes de tipo `GOTO_ORDER`, `ATTACK_ORDER`, `CANCEL_ORDER` o `REPORT_ORDER`[cite: 12].
  * `check_for_enemies(self, enemy_units)`: Comprueba el radio de detección para avistar unidades enemigas cercanas[cite: 12].
  * `get_distance(self, pos)`: Calcula la distancia euclidiana hasta una posición dada[cite: 12].
  * `mean_range(self)`: Calcula el alcance medio ponderado del inventario de armas[cite: 12].
  * `get_soft_attack(self, pos)` / `get_hard_attack(self, pos)`: Calculan el daño blando y duro considerando el alcance[cite: 12].
  * `get_breakthrough()`: Obtiene la capacidad global de ruptura defensiva del inventario[cite: 12].
  * `get_supression(self, pos)`: Calcula la supresión de fuego sobre una posición[cite: 12].
  * `attack(self, enemy, offensive=True, sim=False)`: Núcleo balístico que computa bajas de personal, pérdida de cohesión y moral. Si `sim=True`, realiza el cálculo analítico sin mutar los estados reales[cite: 12].

---

## 5. Núcleo de Simulación (`engine.py`)

### `class BattleState`[cite: 5]
* **Atributos de instancia:**
  * `self.a_units`: Lista de unidades del bando aliado (`ai1`)[cite: 5].
  * `self.e_units`: Lista de unidades del bando enemigo (`ai2`)[cite: 5].
  * `self.end`: Booleano que marca si la batalla ha finalizado[cite: 5].
  * `self.who_won`: Indicador del ganador (`False` = victoria de `ai1`, `True` = victoria de `ai2`)[cite: 5].
* **Métodos:**
  * `__init__(self, my_units, e_units)`: Inicializa los bandos y banderas de control[cite: 5].

### `class OpenBattleEngine`[cite: 5]
* **Métodos estáticos:**
  * `run_tick(state, player1, player2)`: Ciclo principal de simulación por tick (gestión de órdenes, movimiento con fatiga de terreno, resolución de combate, purga por desbandada y condiciones de victoria)[cite: 5].

---

## 6. Inteligencias Artificiales Base y Genoma (`ai.py`, `iga.py`)

### `class Player`[cite: 2, 3]
* **Métodos:**
  * `__init__(self)`[cite: 2, 3]
  * `do_order(self, state, ally=True)`: Interfaz virtual para procesar las acciones del turno[cite: 2, 3].

### `class ModelGenome`[cite: 2, 3] (iga.py)
* **Atributos de instancia (Pesos y Sesgos Tácticos):**
  * `self.name`[cite: 2, 3]
  * `self.weight_damage`, `self.weight_finish_off`, `self.weight_supression`, `self.weight_flanking`, `self.weight_cover`, `self.weight_optimal_range`[cite: 2, 3]
  * `self.penalty_crowding`, `self.aggression_bias`, `self.positional`[cite: 2, 3]
  * `self.health_bias`, `self.cohesion_bias`, `self.morale_bias`, `self.speed_bias`, `self.range_bias`[cite: 2, 3]
* **Métodos:**
  * `__init__(self, name="Modelo_Táctico")`: Inicializa los pesos con valores por defecto[cite: 2, 3].
  * `create_random(p_name)`: Genera un genoma aleatorio dentro de rangos acotados[cite: 2, 3].
  * `to_dict(self)`: Exporta todos los parámetros del genoma a un diccionario de Python[cite: 2, 3].
  * `save_to_json(self, filepath)`: Guarda el diccionario en un archivo JSON en disco[cite: 2, 3].
  * `load_from_json(filepath)`: Reconstruye un `ModelGenome` cargando un archivo JSON[cite: 2, 3].

### `class IGA` (Algoritmo Genético e IA Táctica Principal)[cite: 2, 3]
* **Atributos de instancia:**
  * `self.genome`: Instancia de `ModelGenome` que rige su comportamiento[cite: 2, 3].
* **Métodos principales:**
  * `__init__(self, genome=None)`[cite: 2, 3]
  * `set_model(self, p_genome)`[cite: 2, 3]
  * `get_unit_force(self, unit)`: Pondera la fuerza militar combinando velocidad, alcance, salud, cohesión y moral[cite: 2, 3].
  * `casilla_force(self, unit_force, distance)`: Atenúa la fuerza según la distancia a la celda[cite: 2, 3].
  * `analyze_position(self, unit, order, ai_units, p_units)`: Genera una matriz de fuerza de tamaño $12 \times 12$ para evaluar el control táctico espacial[cite: 2, 3].
  * `seek_5_best_orders(self, unit, ai_units, p_units)`: Evalúa opciones de ataque y desplazamiento para extraer las 5 mejores alternativas[cite: 2, 3].
  * `get_movement_score(self, unit, ai_units, p_units, objective)`: Puntúa casillas de movimiento valorando cobertura, agresión y distancias relativas[cite: 2, 3].
  * `get_attack_score(self, unit, p_unit)`: Simula intercambios de fuego cruzados para puntuar la rentabilidad ofensiva[cite: 2, 3].
  * `execute_orders(self, state, ally=True)`: Método ejecutor global que evalúa el tablero y asigna las órdenes óptimas a cada unidad activa[cite: 2, 3].

#### EXTRA: hay una función llamada mutate_genome, q toma un genoma padre y devuelve un genoma hijo mutado, esta en iga.py