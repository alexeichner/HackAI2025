import torch
import json
import numpy as np

# Load static Pokémon and move data
with open("pokedex_data.json", "r") as f:
    pokedex_data = json.load(f)

with open("enhanced_move_data.json", "r") as f:
    move_data = json.load(f)

type_indices = {"Normal": 0, "Fire": 1, "Water": 2, "Electric": 3, "Grass": 4, "Ice": 5, "Fighting": 6, 
                "Poison": 7, "Ground": 8, "Flying": 9, "Psychic": 10, "Bug": 11, "Rock": 12, "Ghost": 13, 
                "Dragon": 14, "Dark": 15, "Steel": 16, "Fairy": 17}

def encode_pokemon(pokemon_name):
    """Encodes a Pokémon into a feature vector using base stats and type encoding."""
    if pokemon_name not in pokedex_data:
        return torch.zeros(12)  # Default zero vector if Pokémon is unknown
    
    data = pokedex_data[pokemon_name]
    stats = np.array([data["baseStats"][stat] for stat in ["hp", "atk", "def", "spa", "spd", "spe"]])
    
    type_vector = np.zeros(18)
    for t in data["types"]:
        if t in type_indices:
            type_vector[type_indices[t]] = 1
    
    return torch.tensor(np.concatenate((stats, type_vector)), dtype=torch.float32)

def encode_move(move_name):
    """Encodes a move into a feature vector with base power, accuracy, priority, type, and category."""
    move_key = move_name.lower().replace(" ", "")
    if move_key not in move_data:
        return torch.zeros(6)  # Default zero vector if move is unknown
    
    data = move_data[move_key]
    base_power = data.get("basePower", 0)
    accuracy = 100 if data["accuracy"] == True else data.get("accuracy", 0)
    priority = data.get("priority", 0)
    
    category_vector = np.array([1, 0, 0]) if data["category"] == "Physical" else (
                      np.array([0, 1, 0]) if data["category"] == "Special" else np.array([0, 0, 1]))
    
    type_vector = np.zeros(18)
    if data["type"] in type_indices:
        type_vector[type_indices[data["type"]]] = 1
    
    return torch.tensor(np.concatenate(([base_power, accuracy, priority], category_vector, type_vector)), dtype=torch.float32)

def parse_hp(hp_string):
    """Parses HP values, handling status conditions and fainted Pokémon."""
    hp_parts = hp_string.split(" ")[0]  # Extract only the numerical part
    if "fnt" in hp_string:
        return 0.0
    if "/" in hp_parts:
        hp_value = hp_parts.split("/")[0]  # Extract only the current HP
    else:
        hp_value = hp_parts
    return int(hp_value) / 100.0

def create_tensor_from_state(state):
    """Converts a parsed game state into a tensor."""
    active_pokemon_vectors = [encode_pokemon(state["active_pokemon"].get(slot, "")) for slot in ["p1a", "p1b", "p2a", "p2b"]]
    move_vectors = [encode_move(move[1]) for move in state["moves"]] if state["moves"] else [torch.zeros(6)]
    
    hp_values = torch.tensor([parse_hp(state["hp"].get(slot, "0/100")) for slot in ["p1a", "p1b", "p2a", "p2b"]])
    
    field_effects_vector = torch.tensor([
        1 if effect in state["field_effects"]["universal"] else 0 for effect in ["Rain Dance", "Sunny Day", "Sandstorm", "Hail", "Electric Terrain", "Grassy Terrain", "Misty Terrain", "Psychic Terrain", "Trick Room"]
    ])
    
    spikes_vector = torch.tensor([state["field_effects"]["spikes"]["p1"], state["field_effects"]["spikes"]["p2"]], dtype=torch.float32) / 3.0
    
    return torch.cat(active_pokemon_vectors + move_vectors + [hp_values, field_effects_vector, spikes_vector])

def create_tensors_from_parsed_game(parsed_game):
    """Converts all parsed game states into tensors."""
    return [create_tensor_from_state(state) for state in parsed_game]

# Example usage:
with open("parsed_data/parsed_2265587456.json", "r") as f:
    parsed_game = json.load(f)

tensors = create_tensors_from_parsed_game(parsed_game)
print("Generated", len(tensors), "state tensors.")
