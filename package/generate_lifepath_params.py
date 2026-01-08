"""
Lifepath Parameter Generation for Countries/States

This script processes temporal publication data and generates parametrized
lifepaths by fitting sigmoid curves to cumulative distributions.

The output JSON files contain curve parameters that can be used to reconstruct
and analyze publication trajectories over time.

Author: Research Team
Date: 2024
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data
INPUT_CSV = "pivoted_data_states_final.csv"  # CSV with date column and entity columns

# Output directory for JSON files
OUTPUT_DIR = "lifepaths_countries"  # Change to "term_lifepaths_last_states" for states

# Reference date for temporal calculations
REFERENCE_DATE = datetime(2007, 1, 1)

# Padding for curve fitting (days before first observation)
PADDING_DAYS = 135  # 100 + 35 in original script

# Fitting parameters
MAX_ITERATIONS = 100000  # Maximum iterations for curve fitting
SHOW_PLOTS = False  # Set to True to visualize fits (for debugging)
PLOT_FINAL_ONLY = False  # Set to True to only show plot for last time period

# ============================================================================
# SIGMOID CURVE FUNCTIONS
# ============================================================================

def sigmoid(x: np.ndarray, x0: float, k: float, A: float) -> np.ndarray:
    """
    Sigmoid function for curve fitting.
    
    Args:
        x: Input values (normalized time, 0-1)
        x0: Midpoint of sigmoid (position of maximum growth)
        k: Growth rate (steepness of curve)
        A: Maximum value (saturation level)
        
    Returns:
        Sigmoid curve values
    """
    return A / (1 + np.exp(-k * (x - x0)))


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize array using z-score normalization.
    
    Args:
        x: Input array
        
    Returns:
        Normalized array
    """
    return (np.array(x) - np.mean(x)) / np.std(x)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_prepare_data(csv_path: str) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Load CSV data and prepare temporal index.
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        Tuple of (dataframe, list_of_dates, temporal_deltas)
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Calculate sum across all entities
    df['all'] = df.sum(axis=1, numeric_only=True)
    
    print(f"Loaded data from {csv_path}")
    print(f"  - Total rows (time periods): {len(df)}")
    print(f"  - Total columns (entities): {len(df.columns) - 2}")  # Exclude 'date' and 'all'
    print(f"  - Total publications: {df['all'].sum():.0f}")
    
    # Extract dates
    dates = df['date'].to_list()
    
    # Calculate temporal deltas (days since reference date)
    temporal_deltas = []
    for date_str in dates:
        current_date = datetime.strptime(str(date_str).split(" ")[0], "%Y-%m-%d")
        delta = (current_date - REFERENCE_DATE).days
        temporal_deltas.append(delta)
    
    return df, dates, temporal_deltas


def fit_sigmoid_to_entity(entity_name: str, 
                          values: List[float], 
                          padding: int) -> Optional[Dict]:
    """
    Fit sigmoid curve to entity's cumulative publication trajectory.
    
    Args:
        entity_name: Name of entity (country/state)
        values: Publication counts over time
        padding: Number of padding points before first observation
        
    Returns:
        Dictionary of parameters or None if fitting fails
    """
    # Calculate cumulative sum
    cumulative = np.cumsum(np.array(values))
    
    # Check if entity has any publications
    if sum(cumulative) == 0:
        return None
    
    # Prepare data with padding
    cumulative_padded = [0] * padding + list(cumulative)
    
    # Normalize to [0, 1] range
    max_value = max(cumulative)
    normalized = np.array(cumulative_padded) / max_value
    
    # Create time array (0 to 1)
    n_points = len(cumulative) + padding
    x = np.linspace(0, 1, n_points)
    y = normalized
    
    try:
        # Fit sigmoid curve
        # Bounds: x0 unbounded, k > 0, A > 0
        params, pcov = curve_fit(
            sigmoid, 
            x, 
            y,
            bounds=((-np.inf, 0, 0), np.inf),
            maxfev=MAX_ITERATIONS
        )
        
        x0, k, A = params
        
        # Calculate fitted curve
        y_fitted = sigmoid(x, x0, k, A)
        
        # Calculate error (normalized RMSE)
        error = np.sqrt(np.sum((normalize(y) - normalize(y_fitted))**2))
        
        # Calculate different saturation estimates
        lk = A  # Original saturation parameter
        
        # Adjusted saturation estimates
        if A > 1:
            r2N = A * max_value  # If oversaturated
        else:
            r2N = max_value  # If undersaturated
        
        r2 = A * max_value  # Direct scaling
        
        # Return parameter dictionary
        # Format: [x0, k, A_scaled, error, r2, r2N, max_value, lk]
        return {
            'params': [x0, k, max_value, error, r2, r2N, max_value, lk],
            'fitted_curve': y_fitted[padding:].tolist(),
            'normalized_data': y[padding:].tolist(),
            'cumulative_data': cumulative.tolist(),
            'relative_time': x[padding:].tolist(),
            'midpoint_coords': [x0, max_value]
        }
        
    except (RuntimeError, ValueError) as e:
        print(f"  Warning: Could not fit {entity_name}: {e}")
        return None


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_entity_fit(entity_name: str, 
                   fit_data: Dict, 
                   date_str: str,
                   padding: int,
                   show: bool = False,
                   save_path: Optional[str] = None):
    """
    Plot fitted sigmoid curve with actual data points.
    
    Args:
        entity_name: Name of entity
        fit_data: Dictionary containing fit results
        date_str: Date string for title
        padding: Padding value used
        show: Whether to display plot
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    x = fit_data['relative_time']
    y_fitted = fit_data['fitted_curve']
    y_actual = fit_data['cumulative_data']
    x0, y0 = fit_data['midpoint_coords']
    
    # Plot fitted curve
    plt.plot(x, y_fitted, 'b-', linewidth=2, label='Fitted sigmoid', alpha=0.7)
    
    # Plot actual data points
    plt.plot(x, y_actual, 'r.', markersize=8, label='Actual data', alpha=0.8)
    
    # Plot midpoint line
    plt.plot([x0, x0], [0, y0], 'gray', linestyle='--', linewidth=1, 
             label=f'Midpoint (x₀={x0:.3f})')
    
    # Labels and title
    plt.xlabel('Relative Time (0-1)', fontsize=12)
    plt.ylabel('Cumulative Publications', fontsize=12)
    plt.title(f'{entity_name} - Lifepath Fit\n{date_str}', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_parameters_for_timepoint(parameters: Dict[str, List], 
                                  date_str: str,
                                  output_dir: str):
    """
    Save fitted parameters to JSON file for a specific timepoint.
    
    Args:
        parameters: Dictionary mapping entity names to parameter lists
        date_str: Date string for filename
        output_dir: Output directory
    """
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data structure
    # Add unique suffix to each entity to match original format
    params_with_suffix = {}
    for entity, params in parameters.items():
        # Add entity name with underscore suffix (matches original pattern)
        key = f"{entity}_0"  # Could be any unique suffix
        params_with_suffix[key] = params
    
    output_data = {
        'params': params_with_suffix
    }
    
    # Save to JSON file (removed _leila from filename)
    output_path = Path(output_dir) / f"w_{date_str}_full_params.json"
    with open(output_path, 'w') as fp:
        json.dump(output_data, fp, indent=2)
    
    return output_path


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_timepoint(df: pd.DataFrame,
                     timepoint_index: int,
                     dates: List[str],
                     temporal_deltas: List[int],
                     output_dir: str,
                     show_plots: bool = False) -> Dict[str, List]:
    """
    Process a single timepoint: fit curves for all entities up to this point.
    
    Args:
        df: Full dataframe
        timepoint_index: Index of current timepoint
        dates: List of all dates
        temporal_deltas: List of temporal deltas
        output_dir: Output directory
        show_plots: Whether to show plots
        
    Returns:
        Dictionary of fitted parameters for all entities
    """
    current_date = dates[timepoint_index]
    
    # Get all entity columns (exclude 'date' and 'all')
    entity_columns = [col for col in df.columns if col not in ['date', 'all']]
    
    # Calculate padding (decreases as we move forward in time)
    padding = PADDING_DAYS - timepoint_index
    
    # Store parameters for all entities
    all_parameters = {}
    successful_fits = 0
    
    print(f"\nProcessing timepoint {timepoint_index + 1}/{len(dates)}: {current_date}")
    print(f"  Entities to process: {len(entity_columns)}")
    
    # Process each entity
    for i, entity in enumerate(entity_columns):
        # Get values up to current timepoint
        values = df[entity].to_list()[:timepoint_index + 1]
        
        # Fit sigmoid
        fit_result = fit_sigmoid_to_entity(entity, values, padding)
        
        if fit_result:
            all_parameters[entity] = fit_result['params']
            successful_fits += 1
            
            # Optionally plot (useful for debugging)
            if show_plots and (timepoint_index == len(dates) - 1 or not PLOT_FINAL_ONLY):
                plot_entity_fit(entity, fit_result, current_date, padding, 
                              show=True)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(entity_columns)} entities")
    
    print(f"  Successfully fitted: {successful_fits}/{len(entity_columns)} entities")
    
    # Save parameters
    output_path = save_parameters_for_timepoint(all_parameters, current_date, output_dir)
    print(f"  Saved to: {output_path}")
    
    return all_parameters


def main():
    """Main execution function."""
    print("=" * 70)
    print("LIFEPATH PARAMETER GENERATION")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    df, dates, temporal_deltas = load_and_prepare_data(INPUT_CSV)
    
    # Process all timepoints
    print(f"\n[2/3] Processing {len(dates)} timepoints...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    all_results = {}
    
    for timepoint_idx in range(len(dates)):
        parameters = process_timepoint(
            df, 
            timepoint_idx, 
            dates, 
            temporal_deltas, 
            OUTPUT_DIR,
            show_plots=SHOW_PLOTS
        )
        all_results[dates[timepoint_idx]] = parameters
    
    # Summary
    print("\n[3/3] Generation complete!")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Timepoints processed: {len(dates)}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Files generated: {len(dates)} JSON files")
    print(f"File pattern: w_{{date}}_full_params.json")
    print("\nYou can now use these files with plot_lifepaths_professional.py")
    print()


if __name__ == "__main__":
    main()
