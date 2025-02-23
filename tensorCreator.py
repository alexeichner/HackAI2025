import torch
import json
import os
import numpy as np

# Load data
with open("enhanced_move_data.json", "r") as f:
    move_data = json.load(f)
with open("pokedex_data.json", "r") as f:
    pokemon_data = json.load(f)
    print(f"Sample pokedex keys: {list(pokemon_data.keys())[:5]}")  # Debug keys
    print(f"'iron-valiant' in pokedex: {'iron-valiant' in pokemon_data}")  # Debug specific entry

# Complete type chart
type_chart = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0, "steel": 0.5, "fairy": 2.0},
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0, "steel": 0.5}
}

def compute_type_effectiveness(move_type, defender_types):
    multiplier = 1.0
    for def_type in defender_types:
        multiplier *= type_chart.get(move_type.lower(), {}).get(def_type.lower(), 1.0)
    return multiplier

def compute_stab(move_type, attacker_types):
    return 1.5 if move_type.lower() in [t.lower() for t in attacker_types] else 1.0

def encode_game_state(state):
    features = []
    
    # HP (handle various formats)
    for pid in ["p1a", "p1b", "p2a", "p2b"]:
        hp = state["hp"].get(pid, "100/100")
        if "/" in hp:  # "78/100"
            current, max_hp = map(int, hp.split("/"))
        elif " " in hp:  # "100 brn" or "0 fnt"
            hp_value = hp.split()[0]
            current = int(hp_value)
            max_hp = 100  # Assume max_hp if status present
        else:  # "100" (unlikely, but handle it)
            current = int(hp)
            max_hp = 100
        features.append(current / max_hp)
    
    # Speed (base + modifier)
    for pid in ["p1a", "p1b", "p2a", "p2b"]:
        poke_name = state["active_pokemon"].get(pid, "").lower().replace(" ", "-")
        base_speed = pokemon_data.get(poke_name, {"baseStats": {"spe": 50}})["baseStats"]["spe"]
        modifier = state["stat_changes"].get(pid, {}).get("spe", 0)
        features.append(base_speed * (1 + 0.5 * modifier))
    
    # Status
    for pid in ["p1a", "p1b", "p2a", "p2b"]:
        features.append(1 if pid in state["status_conditions"] else 0)
    
    # Field effects
    field = 1 if "Trick Room" in state["field_effects"] else 0
    features.append(field)
    
    # Ability
    ability_vocab = {"quark-drive": 0, "as-one": 1, "commander": 2, "none": 3}
    for pid in ["p1a", "p1b", "p2a", "p2b"]:
        ability = state["abilities"].get(pid, "none")
        features.append(ability_vocab.get(ability.lower(), 3))
    
    # Move features
    move_vocab = {"Feint": 0, "Spirit Break": 1, "Trick Room": 2, "Heat Wave": 3, "Protect": 4, "Snarl": 5, "Icicle Crash": 6, "Fake Out": 7, "Ally Switch": 8, "Helping Hand": 9, "Play Rough": 10}
    for move_tuple in state["moves"]:
        attacker, move, target = move_tuple
        if attacker.startswith("p1"):
            move_key = move.lower().replace(" ", "")
            move_info = move_data.get(move_key, {
                "priority": 0, "basePower": 0, "accuracy": 100, "type": "normal",
                "category": "Physical", "secondary": {"chance": 0}, "isContact": False, "isSound": False
            })
            print(f"Move: {move}, move_info: {move_info}")
            
            power = move_info["basePower"] if "basePower" in move_info and move_info["basePower"] is not None else 0
            accuracy = move_info["accuracy"] if "accuracy" in move_info and move_info["accuracy"] is not None else 100
            secondary_chance = move_info["secondary"]["chance"] if "secondary" in move_info and move_info["secondary"] else 0
            
            features.append(move_info["priority"])
            features.append(power / 100)
            features.append(accuracy / 100)
            features.append(1 if move_info["category"].lower() == "physical" else 0)
            features.append(1 if move_info["category"].lower() == "special" else 0)
            features.append(1 if move_info["category"].lower() == "status" else 0)
            features.append(secondary_chance / 100 if secondary_chance else 0)
            features.append(1 if move_info.get("isContact", False) else 0)
            features.append(1 if move_info.get("isSound", False) else 0)
            
            attacker_slot = attacker.split(":")[0]
            attacker_name_raw = state["active_pokemon"].get(attacker_slot, "unknown")
            attacker_name = attacker_name_raw.lower().replace(" ", "-")
            attacker_types = pokemon_data.get(attacker_name, {"types": ["Normal"]})["types"]
            if attacker_slot in state["tera_types"]:
                attacker_types = [state["tera_types"][attacker_slot]]
            print(f"Attacker: {attacker_name_raw}, normalized: {attacker_name}, types: {attacker_types}")
            
            target_slot = target.split(":")[0] if ":" in target else target
            target_name_raw = state["active_pokemon"].get(target_slot, "unknown")
            target_name = target_name_raw.lower().replace(" ", "-")
            target_types = pokemon_data.get(target_name, {"types": ["Normal"]})["types"]
            if target_slot in state["tera_types"]:
                target_types = [state["tera_types"][target_slot]]
            print(f"Target: {target_name_raw}, normalized: {target_name}, types: {target_types}")
            
            stab = compute_stab(move_info["type"], attacker_types)
            effectiveness = compute_type_effectiveness(move_info["type"], target_types)
            features.append(stab)
            features.append(effectiveness)
            
            return np.array(features, dtype=np.float32), move_vocab.get(move, -1)
    
    return np.array(features, dtype=np.float32), -1

# Load parsed data and create tensors
PARSED_DIR = "parsed_data"
all_game_states = []
for filename in os.listdir(PARSED_DIR):
    if filename.endswith(".json"):
        with open(os.path.join(PARSED_DIR, filename), 'r') as f:
            game_states = json.load(f)
            all_game_states.extend(game_states)

input_tensors = []
output_tensors = []
for state in all_game_states:
    features, move_idx = encode_game_state(state)
    input_tensors.append(features)
    output_tensors.append(move_idx)

input_tensor = torch.tensor(input_tensors, dtype=torch.float32)
output_tensor = torch.tensor(output_tensors, dtype=torch.long)

print("Input Shape:", input_tensor.shape)
print("Sample Input:", input_tensor[0])
print("Sample Output:", output_tensor[0])