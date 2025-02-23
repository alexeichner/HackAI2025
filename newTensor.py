import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# Load move data from JSON
with open("enhanced_move_data.json", "r") as f:
    move_data = json.load(f)

# Create a move-to-index mapping
move_to_index = {move_name: idx for idx, move_name in enumerate(move_data.keys())}
index_to_move = {idx: move_name for move_name, idx in move_to_index.items()}
num_moves = len(move_to_index)  # Total number of move choices

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

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

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
    if not pokemon_name or pokemon_name not in pokedex_data:
        return torch.zeros(12)  # Default zero vector if Pokémon is missing
    
    data = pokedex_data[pokemon_name]
    stats = np.array([data["baseStats"].get(stat, 0) for stat in ["hp", "atk", "def", "spa", "spd", "spe"]])
    
    type_vector = np.zeros(18)
    for t in data.get("types", []):
        if t in type_indices:
            type_vector[type_indices[t]] = 1
    
    return torch.tensor(np.concatenate((stats, type_vector)), dtype=torch.float32)

def encode_move(move_name):
    """Encodes a move into a 24-dimensional vector."""
    move_key = move_name.lower().replace(" ", "") if move_name else ""
    if move_key not in move_data:
        return torch.zeros(24)  # Ensure fixed size
    
    data = move_data[move_key]
    base_power = data.get("basePower", 0)
    accuracy = 100 if data.get("accuracy", 0) == True else data.get("accuracy", 0)
    priority = data.get("priority", 0)
    
    category_vector = np.array([1, 0, 0]) if data.get("category") == "Physical" else (
                      np.array([0, 1, 0]) if data.get("category") == "Special" else np.array([0, 0, 1]))
    
    type_vector = np.zeros(18)
    if data.get("type") in type_indices:
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
    """Converts a parsed game state into a fixed-size tensor (99)."""

    # Ensure exactly 4 Pokémon, pad missing ones with zeros
    pokemon_slots = ["p1a", "p1b", "p2a", "p2b"]
    active_pokemon_vectors = [encode_pokemon(state.get("active_pokemon", {}).get(slot, "")) for slot in pokemon_slots]
    
    # Ensure exactly 4 moves, truncate or pad with zeros
    move_vectors = [encode_move(move[1]) for move in state.get("moves", [])[:4]]
    while len(move_vectors) < 4:
        move_vectors.append(torch.zeros(24))  # Moves should **always** be 24-d

    # Ensure exactly 4 HP values
    hp_values = torch.tensor([parse_hp(state.get("hp", {}).get(slot, "0/100")) for slot in pokemon_slots], dtype=torch.float32)
    
    # Ensure fixed-size field effects vector (9 values)
    field_effect_names = ["Rain Dance", "Sunny Day", "Sandstorm", "Hail", "Electric Terrain", "Grassy Terrain", "Misty Terrain", "Psychic Terrain", "Trick Room"]
    field_effects_vector = torch.tensor([1 if effect in state.get("field_effects", {}).get("universal", []) else 0 for effect in field_effect_names], dtype=torch.float32)
    
    # Ensure fixed-size spikes vector (2 values)
    spikes_vector = torch.tensor([
        state.get("field_effects", {}).get("spikes", {}).get("p1", 0),
        state.get("field_effects", {}).get("spikes", {}).get("p2", 0)
    ], dtype=torch.float32) / 3.0
    
    # Concatenate all components into final tensor
    final_tensor = torch.cat(active_pokemon_vectors + move_vectors + [hp_values, field_effects_vector, spikes_vector])

    # Verify tensor size
    expected_size = 159
    if final_tensor.shape[0] != expected_size:
        print(f"⚠️ Warning: Unexpected tensor size {final_tensor.shape[0]} (Expected: {expected_size})")
        print(f"Tensor breakdown: {final_tensor}")

    return final_tensor

def print_sample_tensor(dataset, index):
    """Prints the tensor values and shape for a given index in the dataset."""
    state_tensor, target_move = dataset[index]
    print(f"Sample {index} tensor shape: {state_tensor.shape}")
    print(f"Tensor values:\n{state_tensor}")
    print(f"Target move tensor:\n{target_move}")

# Define the Pokémon Battle Neural Network
class PokemonBattleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PokemonBattleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output size = number of move choices
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax, since CrossEntropyLoss applies it internally
        return x

# Dataset class for loading parsed Pokémon battles
class PokemonBattleDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        with open(self.file_list[idx], "r") as f:
            parsed_game = json.load(f)
        
        # Select a random game state from the battle
        state = np.random.choice(parsed_game)
        state_tensor = create_tensor_from_state(state)
        
        # Extract the target move (assuming first move in turn)
        target_move_name = state["moves"][0][1] if state["moves"] else "Unknown"
        target_move_index = move_to_index.get(target_move_name, 0)  # Default to 0 if unknown
        
        return state_tensor, torch.tensor(target_move_index, dtype=torch.long)  # Target is now an index

# Initialize model
input_size = 159  # Placeholder, should match tensor size
hidden_size = 128
output_size = 4  # Number of move choices
model = PokemonBattleNN(input_size, hidden_size, output_size)

# Training function
def train_model(model, data_loader, epochs=10):
    loss_function = nn.CrossEntropyLoss()  # Automatically applies softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for state_tensor, target_move in data_loader:
            optimizer.zero_grad()
            output = model(state_tensor)  # Shape: (batch_size, num_moves)
            loss = loss_function(output, target_move)  # target_move is now a class index
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Verify tensor sizes before training
def check_tensor_sizes(dataset):
    for i in range(5):  # Check first 5 samples
        state_tensor, _ = dataset[i]
        print(f"Sample {i} tensor size: {state_tensor.shape}")
        # print_sample_tensor(dataset, i)

# Load dataset and create DataLoader
dataset = PokemonBattleDataset("parsed_data")
check_tensor_sizes(dataset)  # Debug tensor sizes
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
train_model(model, data_loader)
