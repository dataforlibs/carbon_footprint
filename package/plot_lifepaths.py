"""
Lifepath Analysis and Visualization - FIXED VERSION

This version collects entities from ALL JSON files, not just the first one.
This fixes the "0 entities found" bug.
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

# ============================================================================
# CONFIGURATION
# ============================================================================

REFERENCE_DATE = datetime(2006, 7, 1)
INPUT_CSV = "pivoted_data_states_final.csv"
INPUT_JSON_DIR = "lifepaths_countries"
OUTPUT_JSON = "saturation_lifepaths_countries.json"
OUTPUT_DIR = "term_behaviours2_countries"
STATES_TO_ANALYZE = None
FIGURE_SIZE = (10, 6)
LINE_WIDTH = 0.5
DPI = 300
ALPHA = 0.7

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_dates_from_csv(csv_path: str) -> List[str]:
    """Load date array from CSV file."""
    dates = pd.read_csv(csv_path)["date"].to_list()
    print(f"Loaded {len(dates)} dates from {csv_path}")
    return dates


def load_lifepath_data(dates: List[str], json_dir: str) -> Tuple[Dict, List[str]]:
    """
    Load lifepath data from JSON files for all dates.
    FIXED: Collects entities from ALL files, not just the first one.
    """
    all_data = {}
    all_entities = set()  # Collect from all files
    files_found = 0
    
    if not Path(json_dir).exists():
        raise FileNotFoundError(
            f"\n\nERROR: Directory '{json_dir}' does not exist!\n"
            f"Current directory: {Path.cwd()}\n"
        )
    
    print(f"Scanning for JSON files...")
    for date_str in dates:
        json_path = Path(json_dir) / f"w_{date_str}_full_params.json"
        
        try:
            with open(json_path, "r") as fp:
                data = json.load(fp)
                all_data[date_str] = data
                files_found += 1
                
                # Collect entities from THIS file
                if data.get("params"):
                    file_entities = set([x.split("_")[0] for x in data["params"].keys()])
                    all_entities.update(file_entities)
                    if files_found <= 5:  # Show first 5 files
                        print(f"  ✓ {date_str}: {len(file_entities)} entities, {len(data['params'])} total entries")
                else:
                    print(f"  ✗ {date_str}: Empty params")
                    
        except FileNotFoundError:
            if files_found == 0:  # Show warnings only if no files found yet
                print(f"  ✗ Not found: {json_path.name}")
    
    if files_found == 0:
        raise FileNotFoundError(
            f"\n\nERROR: No JSON files found in '{json_dir}'!\n"
            f"Expected pattern: w_{{date}}_full_params.json\n"
            f"First expected: w_{dates[0]}_full_params.json\n"
        )
    
    entities = sorted(list(all_entities))
    
    print(f"\n✓ Successfully loaded {files_found}/{len(dates)} JSON files")
    print(f"✓ Found {len(entities)} unique entities across all files")
    
    if len(entities) > 0:
        print(f"\nSample entities: {', '.join(entities[:5])}")
        if len(entities) > 5:
            print(f"  ... and {len(entities) - 5} more")
    
    return all_data, entities


def build_temporal_index(dates: List[str], reference_date: datetime) -> Dict[str, int]:
    """Build temporal index mapping dates to days since reference."""
    temporal_index = {}
    for date_str in dates:
        year, month, day = map(int, date_str.split("-"))
        current_date = datetime(year, month, day)
        days_delta = (current_date - reference_date).days
        temporal_index[date_str] = days_delta
    return temporal_index


def create_interpolators(temporal_index: Dict[str, int]) -> Dict[int, interp1d]:
    """Create interpolation functions for temporal scaling."""
    interpolators = {}
    values = list(temporal_index.values())
    for i, days in enumerate(values):
        interpolators[i] = interp1d([0, 1], [i, days])
    return interpolators


def scale_temporal_value(date_str: str, value: float, dates: List[str], 
                        interpolators: Dict[int, interp1d],
                        temporal_index: Dict[str, int]) -> float:
    """Scale a temporal value using interpolation."""
    date_index = dates.index(date_str)
    if value <= 1:
        return float(interpolators[date_index](value))
    else:
        value = min(value, 1.32)
        return (interpolators[date_index](value - 1) - 
                interpolators[date_index](0) + 
                interpolators[date_index](1))


# ============================================================================
# DATA PROCESSING
# ============================================================================

def organize_data_by_state(all_data: Dict, dates: List[str], 
                           states: List[str]) -> Dict[str, Dict]:
    """Organize raw data into state-indexed structure."""
    years_data = {date: {state: [] for state in states} for date in dates}
    
    for date_str in dates:
        if date_str in all_data:
            for param_key in all_data[date_str]["params"].keys():
                state = param_key.split("_")[0]
                if state in states:
                    years_data[date_str][state] = all_data[date_str]["params"][param_key]
    
    final_structure = {state: {} for state in states}
    for date_str in dates:
        for state in states:
            if years_data[date_str][state]:
                final_structure[state][date_str] = years_data[date_str][state]
    
    return final_structure


def calculate_derivatives(state_data: Dict[str, Dict]) -> Dict[str, List[float]]:
    """Calculate temporal derivatives."""
    derivatives = {}
    for state, date_dict in state_data.items():
        values = [params[2] for params in date_dict.values()]
        df = pd.DataFrame(values, columns=["vals"])
        diffs = df["vals"].diff().fillna(0).to_list()
        derivatives[state] = diffs
        for i, date_str in enumerate(date_dict.keys()):
            date_dict[date_str].append(diffs[i])
    return derivatives


# ============================================================================
# VISUALIZATION
# ============================================================================

def setup_output_directory(output_dir: str):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def plot_individual_lifepath(state: str, state_data: Dict, dates: List[str],
                            reference_date: datetime, interpolators: Dict,
                            temporal_index: Dict, output_dir: str,
                            color: str = "red") -> Tuple[List, List]:
    """Plot and save lifepath for a single state."""
    x_values = []
    y_values = []
    
    for date_str, params in state_data.items():
        scaled_time = scale_temporal_value(date_str, params[0], dates, 
                                          interpolators, temporal_index)
        x_date = reference_date + timedelta(days=round(float(scaled_time)))
        x_values.append(x_date)
        y_value = math.log(params[1] + 0.001)
        y_values.append(y_value)
    
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(x_values, y_values, c=color, linewidth=LINE_WIDTH)
    plt.gca().invert_yaxis()
    plt.title(f"Lifepath: {state}", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Log Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    
    # Sanitize filename
    safe_filename = state.replace("/", "_").replace("\\", "_").replace(" ", "_")
    output_path = Path(output_dir) / f"{safe_filename}.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return x_values, y_values


def plot_all_lifepaths(all_x_values: List[List], all_y_values: List[List],
                      state_names: List[str], output_dir: str):
    """Create combined plot showing all lifepaths together."""
    if len(all_x_values) == 0:
        print("Warning: No data to plot in combined visualization")
        return
        
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(state_names)))
    
    for i, (x_vals, y_vals, state) in enumerate(zip(all_x_values, all_y_values, state_names)):
        plt.plot(x_vals, y_vals, color=colors[i], linewidth=0.3, 
                alpha=ALPHA, label=state if len(state_names) <= 20 else None)
    
    plt.gca().invert_yaxis()
    plt.title("All Lifepaths Combined", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Log Frequency", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    
    if len(state_names) <= 20 and len(state_names) > 0:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                  fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "all_lifepaths_combined.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Combined plot saved to: {output_path}")
    plt.close()


def prepare_export_data(state_data: Dict, dates: List[str], 
                       reference_date: datetime, interpolators: Dict,
                       temporal_index: Dict) -> Dict:
    """Prepare data for JSON export."""
    export_data = {}
    for state, date_dict in state_data.items():
        export_data[state] = []
        for date_str, params in date_dict.items():
            scaled_time = scale_temporal_value(date_str, params[0], dates,
                                              interpolators, temporal_index)
            x_date = reference_date + timedelta(days=round(float(scaled_time)))
            entry = {
                "x": str(x_date).split(" ")[0],
                "y": math.log(params[1] + 0.001),
                "yr": date_str,
                "c": state,
                "o": params[6]
            }
            export_data[state].append(entry)
    return export_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("LIFEPATH ANALYSIS - FIXED VERSION")
    print("=" * 70)
    
    print("\n[1/7] Loading dates from CSV...")
    dates = load_dates_from_csv(INPUT_CSV)
    
    print("\n[2/7] Loading lifepath data from JSON files...")
    all_data, all_entities = load_lifepath_data(dates, INPUT_JSON_DIR)
    
    if len(all_entities) == 0:
        print("\n✗ ERROR: No entities found in JSON files!")
        print("   This could mean:")
        print("   1. JSON files have empty 'params' dictionaries")
        print("   2. JSON file structure is different than expected")
        print("\n   Run 'python3 diagnose_json_contents.py' for more details")
        return
    
    if STATES_TO_ANALYZE:
        entities = [s for s in STATES_TO_ANALYZE if s in all_entities]
        print(f"\n   Analyzing {len(entities)} selected entities: {', '.join(entities[:5])}...")
    else:
        entities = all_entities
        print(f"\n   Analyzing all {len(entities)} entities")
    
    print("\n[3/7] Building temporal indices...")
    temporal_index = build_temporal_index(dates, REFERENCE_DATE)
    interpolators = create_interpolators(temporal_index)
    
    print("\n[4/7] Organizing data by entity...")
    entity_data = organize_data_by_state(all_data, dates, entities)
    
    # Check how many entities have data
    entities_with_data = [e for e in entities if entity_data.get(e)]
    print(f"   Entities with data: {len(entities_with_data)}/{len(entities)}")
    
    print("\n[5/7] Calculating temporal derivatives...")
    derivatives = calculate_derivatives(entity_data)
    
    print("\n[6/7] Creating visualizations...")
    setup_output_directory(OUTPUT_DIR)
    
    all_x_values = []
    all_y_values = []
    entities_plotted = []
    
    for i, entity in enumerate(entities):
        if entity in entity_data and entity_data[entity]:
            x_vals, y_vals = plot_individual_lifepath(
                entity, entity_data[entity], dates, REFERENCE_DATE,
                interpolators, temporal_index, OUTPUT_DIR
            )
            all_x_values.append(x_vals)
            all_y_values.append(y_vals)
            entities_plotted.append(entity)
            
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(entities)} entities")
    
    print(f"   Completed {len(entities_plotted)} individual plots")
    
    print("\n   Creating combined visualization...")
    plot_all_lifepaths(all_x_values, all_y_values, entities_plotted, OUTPUT_DIR)
    
    print("\n[7/7] Exporting data to JSON...")
    export_data = prepare_export_data(entity_data, dates, REFERENCE_DATE,
                                     interpolators, temporal_index)
    
    with open(OUTPUT_JSON, "w") as fp:
        json.dump(export_data, fp, indent=2)
    print(f"   Data exported to: {OUTPUT_JSON}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"- Individual plots: {len(entities_plotted)} files")
    print(f"- Combined plot: all_lifepaths_combined.png")
    print(f"- JSON export: {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
