import random
from ai import ModelGenome, IGA, GeminiTacticalAI
from engine import OpenBattleEngine, BattleState
from units import UnitInstance, UnitData
from common import Position
from game import render_ascii_map


def strange_battle(ai_a, ai_b, max_ticks = 400):
    # Creamos un escenario base de unidades para probar los genomas
    units_a = [UnitInstance(UnitData(), f"A_{i}", Position(0, i)) for i in range(3)]
    units_b = [UnitInstance(UnitData(), f"B_{i}", Position(11, i)) for i in range(3)]
    
    state = BattleState(units_a, units_b)
   
    
    ticks = 0
    init_a_count = len(state.a_units)
    init_e_count = len(state.e_units)


    while not state.end and ticks < max_ticks:
        # Ejecutamos las órdenes de ambos bandos 
        
        # Avanzamos un tick en el motor
        print(f"Tick: {ticks}")
        state = OpenBattleEngine.run_tick(state, ai_a, ai_b)
        ticks += 1
        if ticks % 10 == 0:
            render_ascii_map(state, ticks)
        
    # Sistema de Fitness basado en la especificación del motor
    score = 0
    if state.end:
        if not state.who_won:  # Gana el candidato a_units
            score += 1000 + (max_ticks - ticks) * 2
        else:  # Pierde
            score -= 500
    else:  # Empate por límite de ticks
        score -= 100
        if len(state.a_units) > len(state.e_units):
            score += 300
        elif len(state.e_units) > len(state.a_units):
            score -= 300
            
    my_losses = (init_a_count - len(state.a_units)) / max(1, init_a_count)
    e_losses = (init_e_count - len(state.e_units)) / max(1, init_e_count)
    
    score -= my_losses * 400 - e_losses * 500
    return score
def run_battle(genome_a, genome_b, max_ticks=400):
    # Creamos un escenario base de unidades para probar los genomas
    units_a = [UnitInstance(UnitData(), f"A_{i}", Position(0, i)) for i in range(3)]
    units_b = [UnitInstance(UnitData(), f"B_{i}", Position(11, i)) for i in range(3)]
    
    state = BattleState(units_a, units_b)
    ai_a = IGA(genome_a)
    ai_b = IGA(genome_b)
    
    ticks = 0
    init_a_count = len(state.a_units)
    init_e_count = len(state.e_units)


    while not state.end and ticks < max_ticks:
        # Ejecutamos las órdenes de ambos bandos 
        
        # Avanzamos un tick en el motor
        print(f"Tick: {ticks}")
        state = OpenBattleEngine.run_tick(state, ai_a, ai_b)
        ticks += 1
        if ticks % 10 == 0:
            render_ascii_map(state, ticks)
        
    # Sistema de Fitness basado en la especificación del motor
    score = 0
    if state.end:
        if not state.who_won:  # Gana el candidato a_units
            score += 1000 + (max_ticks - ticks) * 2
        else:  # Pierde
            score -= 500
    else:  # Empate por límite de ticks
        score -= 100
        if len(state.a_units) > len(state.e_units):
            score += 300
        elif len(state.e_units) > len(state.a_units):
            score -= 300
            
    my_losses = (init_a_count - len(state.a_units)) / max(1, init_a_count)
    e_losses = (init_e_count - len(state.e_units)) / max(1, init_e_count)
    
    score -= my_losses * 400 - e_losses * 500
    return score

def mutate_genome(parent_genome):
    # Creamos un hijo heredando el nombre con sufijo
    child = ModelGenome(parent_genome.name + "_mut")
    for key, val in parent_genome.to_dict().items():
        if key != "name":
            setattr(child, key, val)
            
    # Mutaciones basadas estrictamente en los rangos de la especificación
    child.weight_damage += random.uniform(-0.4, 0.4)
    child.weight_finish_off += random.uniform(-0.6, 0.6)
    child.weight_supression += random.uniform(-0.3, 0.3)
    child.weight_flanking += random.uniform(-0.5, 0.5)
    child.weight_cover += random.uniform(-0.4, 0.4)
    child.weight_optimal_range += random.uniform(-0.3, 0.3)
    child.penalty_crowding += random.uniform(-0.2, 0.2)
    child.aggression_bias += random.uniform(-0.2, 0.2)
    
    # Límites obligatorios de la especificación
    child.aggression_bias = max(0.2, min(3.0, child.aggression_bias))
    
    # Mutaciones menores para los sesgos generales
    child.positional += random.uniform(-0.4, 0.4)
    child.health_bias += random.uniform(-0.4, 0.4)
    child.cohesion_bias += random.uniform(-0.4, 0.4)
    child.morale_bias += random.uniform(-0.4, 0.4)
    child.speed_bias += random.uniform(-0.4, 0.4)
    child.range_bias += random.uniform(-0.4, 0.4)
    
    return child


def run_evolutionary_tournament(generations=10, population_size=8, baseline_ai=GeminiTacticalAI):
    print("=== INICIANDO TORNEO EVOLUTIVO DE IAs ===")
    
    # Población inicial de genomas aleatorios
    population = [ModelGenome.create_random(f"Gen0_Bot_{i}") for i in range(population_size)]
    baseline_rival = ModelGenome("Baseline_Rival")
    
    best_overall = None
    max_overall_score = -999999
    
    for gen in range(generations):
        print(f"\n--- Generación {gen + 1} de {generations} ---")
        fitness_scores = {}
        
        for ind in population:
            print("Comienza Batalla")
            # Evaluamos contra el rival base
            score = strange_battle(IGA(ind), baseline_ai)
            fitness_scores[ind] = score
            
        # Ordenamos la población por su puntuación de mayor a menor
        sorted_pop = sorted(population, key=lambda x: fitness_scores[x], reverse=True)
        gen_best = sorted_pop[0]
        print(f"Mejor de la gen: {gen_best.name} | Fitness: {fitness_scores[gen_best]:.2f}")
        
        if fitness_scores[gen_best] > max_overall_score:
            max_overall_score = fitness_scores[gen_best]
            best_overall = gen_best
            
        # ELITISMO: El mejor pasa directamente a la siguiente generación
        next_generation = [gen_best]
        
        # Seleccionamos la mitad superior como padres (elitismo de padres)
        parents = sorted_pop[:len(sorted_pop) // 2]
        
        # Rellenamos el resto de la población mutando a los mejores
        while len(next_generation) < population_size:
            parent = random.choice(parents)
            child = mutate_genome(parent)
            next_generation.append(child)
            
        population = next_generation
        
    # Guardar el campeón absoluto al terminar todas las generaciones
    if best_overall:
        best_overall.save_to_json("campeon_torneo.json")
        print(f"\n¡Torneo finalizado con éxito!")
        print(f"El campeón absoluto es '{best_overall.name}' con un score de {max_overall_score:.2f}.")
        print("Modelo guardado en 'campeon_torneo.json'.")

if __name__ == "__main__":
    run_evolutionary_tournament(generations=100, population_size=50, baseline_ai = GeminiTacticalAI())
