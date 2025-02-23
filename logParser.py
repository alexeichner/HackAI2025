import json
import os
import requests

REPLAYS_DIR = "replays"
PARSED_DIR = "parsed_data"
os.makedirs(PARSED_DIR, exist_ok=True)

# Fetch Showdown moves data
SHOWDOWN_MOVES_URL = "https://play.pokemonshowdown.com/data/moves.json"
response = requests.get(SHOWDOWN_MOVES_URL)
if response.status_code != 200:
    raise Exception(f"Failed to fetch Showdown moves: Status {response.status_code}")
showdown_moves = json.loads(response.text)

# Fetch Showdown pokedex data
SHOWDOWN_POKEDEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"
response = requests.get(SHOWDOWN_POKEDEX_URL)
if response.status_code != 200:
    raise Exception(f"Failed to fetch Showdown pokedex: Status {response.status_code}")
showdown_pokedex = json.loads(response.text)

# Parser function
def parse_log(log_data):
    game_states = []
    current_state = {
        "turn": 0,
        "active_pokemon": {},
        "moves": [],
        "hp": {},
        "field_effects": [],
        "stat_changes": {},
        "status_conditions": {},
        "item_activations": [],
        "ability_activations": [],
        "move_failures": [],
        "critical_hits": [],
        "move_effectiveness": [],
        "missed_attacks": [],
        "self_inflicted_damage": [],
        "end_of_turn_effects": [],
        "tera_types": {},
        "abilities": {}
    }

    # Initialize field tracking properly
    current_state["field_effects"] = {
        "universal": [],
        "p1-side": [],
        "p2-side": [],
        "spikes": {"p1": 0, "p2": 0}  # Spikes count for each side
    }

    log_lines = log_data["log"].split("\n")
    for line in log_lines:
        if line.startswith("|turn|"):
            if current_state["turn"] > 0:
                game_states.append(current_state.copy())
            current_state = {
                "turn": int(line.split("|")[2]),
                "active_pokemon": current_state["active_pokemon"].copy(),
                "moves": [],
                "hp": current_state["hp"].copy(),
                "field_effects": current_state["field_effects"].copy(),
                "stat_changes": current_state["stat_changes"].copy(),
                "status_conditions": current_state["status_conditions"].copy(),
                "item_activations": [],
                "ability_activations": [],
                "move_failures": [],
                "critical_hits": [],
                "move_effectiveness": [],
                "missed_attacks": [],
                "self_inflicted_damage": [],
                "end_of_turn_effects": [],
                "tera_types": current_state["tera_types"].copy(),
                "abilities": current_state["abilities"].copy()
            }
        
        elif line.startswith("|switch|"):
            parts = line.split("|")
            pokemon_id, pokemon_details = parts[2], parts[3]
            poke_name = pokemon_details.split(",")[0]  # Keep raw name
            slot = pokemon_id.split(":")[0]
            current_state["active_pokemon"][slot] = poke_name  # Store raw name
            if slot not in current_state["hp"]:
                current_state["hp"][slot] = "100/100"
            ability = showdown_pokedex.get(poke_name.lower().replace(" ", "-"), {"abilities": {"0": "none"}})["abilities"].get("0", "none")
            current_state["abilities"][slot] = ability
            print(f"Switch: {pokemon_id}, raw name: {poke_name}")  # Debug raw name
        
        elif line.startswith("|move|"):
            parts = line.split("|")
            attacker, move = parts[2], parts[3]
            
            # Check if the move has multiple targets
            if "[spread]" in parts:
                spread_index = parts.index("[spread]")  # Find where "[spread]" appears
                target_list = parts[spread_index + 1:]  # Capture all targets after "[spread]"
            else:
                target_list = [parts[4]]  # Default to a single target if no "[spread]" is found

            current_state["moves"].append((attacker, move, target_list))

        
        elif line.startswith("|-damage|"):
            parts = line.split("|")
            target, hp_status = parts[2], parts[3]
            current_state["hp"][target.split(":")[0]] = hp_status
        
        elif line.startswith("|-heal|"):
            parts = line.split("|")
            target, hp_status = parts[2], parts[3]
            current_state["hp"][target.split(":")[0]] = hp_status
        
        elif line.startswith("|-crit|"):
            parts = line.split("|")
            target = parts[2]
            current_state["critical_hits"].append(target)
        
        elif line.startswith("|-supereffective|") or line.startswith("|-resisted|"):
            parts = line.split("|")
            target = parts[2]
            effectiveness = "super effective" if "supereffective" in line else "resisted"
            current_state["move_effectiveness"].append((target, effectiveness))
        
        elif line.startswith("|-miss|"):
            parts = line.split("|")
            attacker, move = parts[2], parts[3]
            current_state["missed_attacks"].append((attacker, move))
        
        elif line.startswith("|-recoil|") or "|-confusion|" in line:
            parts = line.split("|")
            target, damage_info = parts[2], parts[3]
            current_state["self_inflicted_damage"].append((target, damage_info))

        elif line.startswith("|-fieldstart|"):
            parts = line.split("|")
            effect = parts[2].split(":")[-1].strip()

            # Universal effects (weather, terrain, Trick Room)
            universal_effects = ["Rain Dance", "Sunny Day", "Sandstorm", "Hail", 
                                "Electric Terrain", "Grassy Terrain", "Misty Terrain", 
                                "Psychic Terrain", "Trick Room"]  # Trick Room added here
            if effect in universal_effects:
                current_state["field_effects"]["universal"].append(effect)

            # One-sided effects (Reflect, Light Screen, Aurora Veil)
            elif effect in ["Reflect", "Light Screen", "Aurora Veil"]:
                side = "p1-side" if "[of] p1" in line else "p2-side"
                current_state["field_effects"][side].append(effect)

            # Spikes (stackable up to 3)
            elif effect == "Spikes":
                side = "p1" if "[of] p1" in line else "p2"
                if current_state["field_effects"]["spikes"][side] < 3:
                    current_state["field_effects"]["spikes"][side] += 1  # Increase spike count

        elif line.startswith("|-fieldend|"):
            parts = line.split("|")
            effect = parts[2].split(":")[-1].strip()

            # Remove from universal effects
            if effect in current_state["field_effects"]["universal"]:
                current_state["field_effects"]["universal"].remove(effect)

            # Remove from one-sided effects
            elif effect in ["Reflect", "Light Screen", "Aurora Veil"]:
                side = "p1-side" if "[of] p1" in line else "p2-side"
                if effect in current_state["field_effects"][side]:
                    current_state["field_effects"][side].remove(effect)

            # Remove Spikes (reduce count)
            elif effect == "Spikes":
                side = "p1" if "[of] p1" in line else "p2"
                if current_state["field_effects"]["spikes"][side] > 0:
                    current_state["field_effects"]["spikes"][side] -= 1


        
        elif line.startswith("|-boost|") or line.startswith("|-unboost|"):
            parts = line.split("|")
            target, stat, amount = parts[2], parts[3], int(parts[4])
            if target not in current_state["stat_changes"]:
                current_state["stat_changes"][target] = {}
            if "-boost" in parts[1]:
                current_state["stat_changes"][target][stat] = current_state["stat_changes"].get(target, {}).get(stat, 0) + amount
            else:
                current_state["stat_changes"][target][stat] = current_state["stat_changes"].get(target, {}).get(stat, 0) - amount
        
        elif line.startswith("|-status|"):
            parts = line.split("|")
            target, status = parts[2], parts[3]
            current_state["status_conditions"][target.split(":")[0]] = status
        
        elif line.startswith("|-endstatus|"):
            parts = line.split("|")
            target = parts[2]
            if target in current_state["status_conditions"]:
                del current_state["status_conditions"][target.split(":")[0]]
        
        elif line.startswith("|-item|"):
            parts = line.split("|")
            target, item = parts[2], parts[3]
            current_state["item_activations"].append((target, item))
        
        elif line.startswith("|-ability|"):
            parts = line.split("|")
            target, ability = parts[2], parts[3]
            current_state["ability_activations"].append((target, ability))
            current_state["abilities"][target.split(":")[0]] = ability
        
        elif line.startswith("|-fail|"):
            parts = line.split("|")
            move = parts[2] if len(parts) > 2 else "Unknown move"
            current_state["move_failures"].append(move)
        
        elif line.startswith("|-terastallize|"):
            parts = line.split("|")
            target, new_type = parts[2], parts[3]
            current_state["tera_types"][target.split(":")[0]] = new_type
        
        elif line.startswith("|upkeep"):
            for next_line in log_lines[log_lines.index(line) + 1:]:
                if next_line.startswith("|turn|") or next_line.startswith("|win|"):
                    break
                elif next_line.startswith("|-heal|"):
                    parts = next_line.split("|")
                    target, hp_status = parts[2], parts[3]
                    source = parts[4] if len(parts) > 4 else "unknown"
                    current_state["end_of_turn_effects"].append(("heal", target, hp_status, source))
                    current_state["hp"][target.split(":")[0]] = hp_status
                elif next_line.startswith("|-damage|") and ("[from] brn" in next_line or "[from] sandstorm" in next_line):
                    parts = next_line.split("|")
                    target, hp_status = parts[2], parts[3]
                    source = parts[4] if len(parts) > 4 else "unknown"
                    current_state["end_of_turn_effects"].append(("damage", target, hp_status, source))
                    current_state["hp"][target.split(":")[0]] = hp_status

    game_states.append(current_state)
    return game_states

# Process replays
for filename in os.listdir(REPLAYS_DIR):
    if filename.endswith(".json"):
        input_path = os.path.join(REPLAYS_DIR, filename)
        output_path = os.path.join(PARSED_DIR, f"parsed_{filename}")
        with open(input_path, 'r') as f:
            log_data = json.load(f)
        parsed_data = parse_log(log_data)
        with open(output_path, 'w') as f:
            json.dump(parsed_data, f, indent=2)
        print(f"Parsed {filename} -> {output_path}")

# Save move and pokedex data
move_data = {move.lower().replace(" ", ""): info for move, info in showdown_moves.items()}
with open("enhanced_move_data.json", "w") as f:
    json.dump(move_data, f, indent=2)

with open("pokedex_data.json", "w") as f:
    json.dump(showdown_pokedex, f, indent=2)