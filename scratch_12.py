import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# --- 1. Load Data ---
# Using the example data, including Product D which might have no valid Q1-3 responses
data_list = [
    {'respondent id': 1, 'product': 'Product A', 'q1': 3, 'q2': 4, 'q3': 'idk', 'q4': 2, 'q5': 4},
    {'respondent id': 2, 'product': 'Product B', 'q1': 5, 'q2': 6, 'q3': 4, 'q4': 5, 'q5': 6},
    {'respondent id': 3, 'product': 'Product A', 'q1': 1, 'q2': 2, 'q3': 1, 'q4': 'na', 'q5': 3},
    {'respondent id': 4, 'product': 'Product C', 'q1': 4, 'q2': 4, 'q3': 5, 'q4': 3, 'q5': 4},
    {'respondent id': 5, 'product': 'Product B', 'q1': 6, 'q2': 5, 'q3': 6, 'q4': 4, 'q5': 'idk'},
    {'respondent id': 6, 'product': 'Product C', 'q1': 2, 'q2': 3, 'q3': 4, 'q4': 1, 'q5': 3},
    {'respondent id': 7, 'product': 'Product A', 'q1': 5, 'q2': 6, 'q3': 5, 'q4': 4, 'q5': 6},
    {'respondent id': 8, 'product': 'Product B', 'q1': 'na', 'q2': 3, 'q3': 2, 'q4': 4, 'q5': 1},
    {'respondent id': 9, 'product': 'Product C', 'q1': 4, 'q2': 5, 'q3': 5, 'q4': 6, 'q5': 6},
    # Added Example: Product D has no valid responses for Q1-3
    {'respondent id': 10, 'product': 'Product D', 'q1': 'idk', 'q2': 'na', 'q3': 'idk', 'q4': 1, 'q5': 2},
    {'respondent id': 11, 'product': 'Product D', 'q1': 'na', 'q2': 'na', 'q3': 'na', 'q4': 2, 'q5': 1},
]
df_raw = pd.DataFrame(data_list)

# --- Define Valid Responses ---
valid_responses = [1, 2, 3, 4, 5, 6] # All possible numeric responses on the scale

# --- Get ALL unique products from the RAW data ---
# This ensures products with no valid data later still appear
all_products = sorted(df_raw['product'].unique())

# --- 2. Data Cleaning & Transformation ---
question_category = 'Q1-3' # Label for the questions being analyzed
question_cols = ['q1', 'q2', 'q3'] # Select the question columns to process

df_melted = df_raw.melt(
    id_vars=['respondent id', 'product'], value_vars=question_cols,
    var_name='question', value_name='response'
)

# Convert responses to numeric, setting non-numeric to NaN
df_melted['response_num'] = pd.to_numeric(df_melted['response'], errors='coerce')

# Keep only rows with a response number (removes NaNs from conversion)
df_clean = df_melted.dropna(subset=['response_num']).copy()

# Keep only rows where the response number is in our defined valid list
df_clean = df_clean[df_clean['response_num'].isin(valid_responses)].copy()

# Ensure the response numbers are integers
df_clean['response_num'] = df_clean['response_num'].astype(int)

# --- 3. Data Aggregation ---
# Group cleaned data by product and response number to get counts
# This might miss products if they had NO valid responses at all in df_clean
df_counts_partial = df_clean.groupby(['product', 'response_num']).size().unstack(fill_value=0)

# --- Reindex to include ALL products defined earlier ---
# This adds rows for products that were missing, filling counts with 0
df_counts = df_counts_partial.reindex(all_products, fill_value=0)

# --- Ensure all valid response columns exist ---
# Handles cases where a specific response value (e.g., 4) never appeared in the data
for r in valid_responses:
    if r not in df_counts.columns:
        df_counts[r] = 0
# Ensure columns are in the defined order (1, 2, 3, 4, 5, 6)
df_counts = df_counts[valid_responses]

# --- 4. Percentage Calculation ---
# Calculate percentage based on the total responses *for each product*
# Uses the reindexed df_counts, so products with 0 total responses get 0%
df_perc = df_counts.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else 0, axis=1)


# ---------------------------------------------------------------------------
# --- Plotting Section: Second Chart Style (Climate Regs Example) ---
# ---------------------------------------------------------------------------

# --- Style Configuration ---
# 1. Map Response values to Categories and Colors for this chart style
#    (Assumption: Mapping 1, 2 to oppose, 5, 6 to support, ignoring 3, 4)
response_mapping = {
    1: {'name': 'Strongly Oppose (1)', 'color': '#b35806'}, # Dark Brown/Orange
    2: {'name': 'Somewhat Oppose (2)', 'color': '#fdb863'}, # Light Brown/Orange
    5: {'name': 'Somewhat Support (5)', 'color': '#99d594'}, # Light Teal/Green
    6: {'name': 'Strongly Support (6)', 'color': '#008837'}  # Dark Teal/Green
}
# Get the list of response numbers we are actually plotting in this style
responses_to_plot = list(response_mapping.keys())

# 2. Define horizontal spacing/offsets for each category's visual column
category_gap = 10 # Visual gap between category columns
logical_column_width = 45 # Adjust visual width allocated for bars in each column
column_start_positions = {}
current_pos = 0
for i, resp_num in enumerate(responses_to_plot):
    column_start_positions[resp_num] = current_pos
    current_pos += logical_column_width + category_gap

# 3. Text label settings inside bars
TEXT_COLOR_INSIDE_BAR = 'white'
TEXT_SIZE_INSIDE_BAR = 9
TEXT_WEIGHT_INSIDE_BAR = 'bold'
MIN_PERC_FOR_TEXT = 3 # Minimum percentage width for a bar to display text inside

# --- Check if data exists for plotting ---
valid_cols_in_perc = [col for col in responses_to_plot if col in df_perc.columns]
if not valid_cols_in_perc:
     print("Warning: None of the responses selected for plotting (1, 2, 5, 6) exist in the calculated percentages. Cannot generate plot.")
     # Optionally exit or handle differently: import sys; sys.exit()
else:
    # --- Plotting Setup ---
    num_products = len(df_perc.index)
    items = df_perc.index # Get product names (includes all products due to reindex)
    y_pos = np.arange(num_products)

    fig, ax = plt.subplots(figsize=(12, max(5, num_products * 0.6))) # Adjust figsize as needed

    # --- Plot Bars and Text ---
    # Iterate through the products (rows on y-axis)
    for y_idx, product_name in enumerate(items):
        # Iterate through the response categories we want to plot for this style
        for response_num in responses_to_plot:
            # Check if this response column actually exists in df_perc
            if response_num in df_perc.columns:
                # Get data for this specific product and response
                percentage = df_perc.loc[product_name, response_num]
                count = df_counts.loc[product_name, response_num] # Get count for label
                details = response_mapping[response_num]
                column_start = column_start_positions[response_num]

                # Plot the bar segment for this category if percentage > 0
                if percentage > 0:
                    ax.barh(y_pos[y_idx],         # y position for the product
                            percentage,          # Width of the bar is the percentage
                            left=column_start,   # Start position defines the column
                            color=details['color'],
                            edgecolor='white',   # Add edge for clarity
                            linewidth=0.5,
                            height=0.7)          # Adjust bar height (0.0 to 1.0)

                    # Add text label (count) inside the bar if wide enough
                    if percentage >= MIN_PERC_FOR_TEXT and count > 0:
                        x_center = column_start + percentage / 2 # Center text
                        ax.text(x_center,
                                y_pos[y_idx],
                                str(count),      # Display the count
                                ha='center',
                                va='center',
                                color=TEXT_COLOR_INSIDE_BAR,
                                fontsize=TEXT_SIZE_INSIDE_BAR,
                                weight=TEXT_WEIGHT_INSIDE_BAR)

    # --- Formatting ---
    # Set Y-axis ticks and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(items, fontsize=11)
    ax.invert_yaxis() # Product A (or first product) at the top

    # Set X-axis limits
    # Calculate the right edge of the last column
    last_response_num = responses_to_plot[-1]
    max_overall_width = column_start_positions[last_response_num] + logical_column_width
    ax.set_xlim(0, max_overall_width + category_gap) # Set limits based on calculated positions

    # Add category labels at the top
    # Adjust label_y_pos if labels overlap title or plot
    label_y_pos = num_products - 0.5 + (num_products * 0.05) # Position above the top bar slightly
    # Alternative fixed position relative to axis (requires tweaking based on figsize/dpi):
    # label_y_pos = ax.get_ylim()[1] * 1.02 # Example: 2% above the highest y-tick value

    for response_num in responses_to_plot:
        label_x_pos = column_start_positions[response_num] + logical_column_width / 2 # Center above column
        ax.text(label_x_pos, label_y_pos, response_mapping[response_num]['name'],
                ha='center', va='bottom', fontsize=11, weight='bold', clip_on=False)

    # Remove spines and ticks for a cleaner look like the example
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Hide y-axis line itself
    ax.spines['bottom'].set_visible(False) # Hide x-axis line itself
    ax.tick_params(axis='y', length=0) # Remove y-axis tick marks next to labels
    ax.tick_params(axis='x', length=0, labelbottom=False) # Remove x-axis tick marks and numeric labels

    # Add Title
    # Use a dynamic title or customize as needed
    title_text = f"Product Response Distribution ({question_category})"
    # Position title - suptitle often works better with top labels
    fig.suptitle(title_text, fontsize=14, weight='bold', y=0.98) # Adjust y as needed

    # Adjust layout to prevent overlap
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust top value (e.g., 0.93) to make space for suptitle

    plt.show()