from engine import OpenBattleEngine, BattleState
from units import *
from ai import *
from aegis import AegisAI
from manticore import ManticoreAI
from idk import DQNCombatAI
from common import GameData
import json
import sys

def ai_duel(a_units, e_units, ai1, ai2):
    print("Batalla Comenzada!")
    state = BattleState(a_units, e_units)
    ticks = 0
    while state.end == False:
        state = OpenBattleEngine.run_tick(state, ai1, ai2)
        render_ascii_map(state, ticks)
        ticks += 1
    if state.who_won:
        print("Ai2 WON")
    else:
        print("Ai1 WON")


def render_ascii_map(state, tick):
    # Inicializamos una matriz vacía de 12x12 con puntos
    grid = [[" . " for _ in range(GameData.map_x_size)] for _ in range(GameData.map_y_size)]
    
    # Pintamos las unidades aliadas con 'A'
    for u in state.a_units:
        if u.current_health > 0:
            grid[u.grid_position.y][u.grid_position.x] = " A "
            
    # Pintamos las unidades enemigas con 'E' (si coinciden, ponemos 'X' de choque)
    for u in state.e_units:
        if u.current_health > 0:
            pos_y = u.grid_position.y
            pos_x = u.grid_position.x
            if grid[pos_y][pos_x] == " A ":
                grid[pos_y][pos_x] = " X "  # ¡Combate cuerpo a cuerpo en la misma casilla!
            else:
                grid[pos_y][pos_x] = " E "
                
    # Imprimimos el estado por consola de forma visual
    print(f"\n--- TICK: {tick} ---")
    print(f" [A] Aliadas vivas: {len(state.a_units)}  |  [E] Enemigas vivas: {len(state.e_units)}")
    print("-" * 39)
    for y in range(GameData.map_y_size):
        row_str = "".join(grid[y])
        print(row_str)
    print("-" * 39)

def std_duel(ai1, ai2):
    units_a = [UnitInstance(UnitData(), f"A_{i}", Position(0, i)) for i in range(3)]
    units_b = [UnitInstance(UnitData(), f"B_{i}", Position(11, i)) for i in range(3)]
    ai_duel(units_a, units_b, ai1, ai2)




if __name__ == "__main__":
    from tournament import run_evolutionary_tournament
    best_genome_path = "campeon_torneo.json"

    best_genome = ModelGenome.load_from_json(best_genome_path)
    iga = IGA(best_genome)


    gemini_ia = GeminiPotentialFieldAI()
    aegis_ia = AegisAI()
    manticore_ia = ManticoreAI()
    dqn_ai_path = "neural_rival.npz"
    dqn_ai = DQNCombatAI(dqn_ai_path)

    fmb = iga
    cmb = dqn_ai
    
    if len(sys.argv) > 1 and sys.argv[1]=="train":
        run_evolutionary_tournament(30, 25, cmb)
    else:
        std_duel(fmb, cmb)

    # vamos a lanzar una mierda generacional de estas
    #run_evolutionary_tournament(15, 50, manticore_ia)
