import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# --- Configuration ---
# <<< SET YOUR DESIRED CENTER POINT HERE >>>
# Use X.5 to center BETWEEN response X and X+1 (e.g., 3.5)
# Use X.0 to center ON response X (e.g., 3.0)
center_at = 3  # Example: Center between 3 and 4

# <<< COLOR PALETTE >>>
# Adjusted Oranges (Darkest=1, Lightest=3) <-> Blues (Lightest=4, Darkest=6)
colors = {
    1: '#f16913', 2: '#fdae6b', 3: '#fdeada', # Oranges
    4: '#d1e5f0', 5: '#67a9cf', 6: '#2166ac'  # Blues
}

# --- 1. Load Data ---
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
]
df_raw = pd.DataFrame(data_list)

# --- 2. Data Cleaning & Transformation ---
question_category = 'Q1-3'
question_cols = ['q1', 'q2', 'q3']#, 'q4', 'q5']
df_melted = df_raw.melt(
    id_vars=['respondent id', 'product'], value_vars=question_cols,
    var_name='question', value_name='response'
)
valid_responses = [1, 2, 3, 4, 5, 6]
df_melted['response_num'] = pd.to_numeric(df_melted['response'], errors='coerce')
df_clean = df_melted.dropna(subset=['response_num']).copy()
df_clean = df_clean[df_clean['response_num'].isin(valid_responses)].copy()
df_clean['response_num'] = df_clean['response_num'].astype(int)

# --- 3. Data Aggregation ---
df_counts = df_clean.groupby(['product', 'response_num']).size().unstack(fill_value=0)
for r in valid_responses:
    if r not in df_counts.columns: df_counts[r] = 0
df_counts = df_counts[valid_responses]

# --- 4. Percentage Calculation ---
df_perc = df_counts.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else 0, axis=1)

# --- 5. Plotting Logic (Flexible Centering) ---
fig, ax = plt.subplots(figsize=(10, max(5, round(len(df_perc.index) * 0.6))))  # width, height

category_map = {r: f'Response {r}' for r in valid_responses}
items = df_perc.index
y_pos = np.arange(len(items))

# Determine centering type
is_integer_center = (center_at % 1 == 0)
center_col_num = int(center_at) if is_integer_center else None
center_string = f"on {center_col_num}" if is_integer_center else f"between {int(center_at)} & {int(center_at) + 1}"

if is_integer_center and center_col_num in valid_responses:
    # --- Centering ON a specific response value (e.g., 3.0) ---
    neg_cols = [r for r in valid_responses if r < center_col_num]
    pos_cols = [r for r in valid_responses if r > center_col_num]
    center_col = center_col_num

    center_half_width = df_perc[center_col].fillna(0) / 2

    # Plot negative side (responses < center_col)
    current_neg_edge = -center_half_width # Left edge starts here
    for col in sorted(neg_cols, reverse=True): # Plot 2 then 1 if centering on 3
        width = df_perc[col].fillna(0)
        plot_left = current_neg_edge - width # Calculate left edge for this bar
        ax.barh(y_pos, width, left=plot_left, color=colors[col], edgecolor='white', label=category_map[col])
        current_neg_edge = plot_left # Update the edge for the next bar to the left

    # Plot center category (split in half)
    ax.barh(y_pos, center_half_width, left=-center_half_width, color=colors[center_col], edgecolor='white', label=category_map[center_col]) # Left half
    ax.barh(y_pos, center_half_width, left=0, color=colors[center_col], edgecolor='white') # Right half

    # Plot positive side (responses > center_col)
    current_pos_edge = center_half_width # Right edge starts here
    for col in sorted(pos_cols): # Plot 4, 5, 6 if centering on 3
        width = df_perc[col].fillna(0)
        ax.barh(y_pos, width, left=current_pos_edge, color=colors[col], edgecolor='white', label=category_map[col])
        current_pos_edge += width # Update the edge for the next bar to the right

elif not is_integer_center and 0.5 <= center_at <= 5.5:
    # --- Centering BETWEEN response values (e.g., 3.5) ---
    split_point = int(np.floor(center_at))
    neg_cols = [r for r in valid_responses if r <= split_point]
    pos_cols = [r for r in valid_responses if r > split_point]

    # Plot negative side (responses <= split_point)
    current_neg_edge = pd.Series(np.zeros(len(df_perc.index)), index=df_perc.index) # Edge starts at 0
    for col in sorted(neg_cols, reverse=True): # Plot 3, 2, 1 if centering at 3.5
        width = df_perc[col].fillna(0)
        plot_left = -width - current_neg_edge # Calculate left edge relative to 0
        ax.barh(y_pos, width, left=plot_left, color=colors[col], edgecolor='white', label=category_map[col])
        current_neg_edge += width # Accumulate width magnitude

    # Plot positive side (responses > split_point)
    current_pos_edge = pd.Series(np.zeros(len(df_perc.index)), index=df_perc.index) # Edge starts at 0
    for col in sorted(pos_cols): # Plot 4, 5, 6 if centering at 3.5
        width = df_perc[col].fillna(0)
        ax.barh(y_pos, width, left=current_pos_edge, color=colors[col], edgecolor='white', label=category_map[col])
        current_pos_edge += width # Accumulate width magnitude
else:
    print(f"Warning: center_at value {center_at} is outside the typical range or invalid. Plotting may be incorrect.")
    # Basic fallback or error handling could go here if needed


# --- 6. Formatting ---
ax.set_yticks(y_pos)
ax.set_yticklabels(items)
ax.invert_yaxis()

ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel(f"Percentage of Responses (Centered {center_string})") # Updated label
ax.axvline(0, color='grey', linewidth=0.8, linestyle='-')

# Determine x-axis limits based on centering type
min_val, max_val = 0, 0 # Initialize
if is_integer_center and center_col_num in valid_responses:
    neg_total = df_perc[neg_cols].sum(axis=1, skipna=True) + df_perc[center_col].fillna(0) / 2
    pos_total = df_perc[pos_cols].sum(axis=1, skipna=True) + df_perc[center_col].fillna(0) / 2
    min_val = (-neg_total).min()
    max_val = pos_total.max()
elif not is_integer_center and 0.5 <= center_at <= 5.5:
    split_point = int(np.floor(center_at))
    neg_cols = [r for r in valid_responses if r <= split_point]
    pos_cols = [r for r in valid_responses if r > split_point]
    min_val = (-df_perc[neg_cols].sum(axis=1, skipna=True)).min() if neg_cols else 0
    max_val = (df_perc[pos_cols].sum(axis=1, skipna=True)).max() if pos_cols else 0

min_val = min(min_val if pd.notna(min_val) else 0, -5)
max_val = max(max_val if pd.notna(max_val) else 0, 5)
ax.set_xlim(min_val - 5, max_val + 5)

ax.set_title(f"Product Response Distribution (Aggregated {question_category})", pad=30)

# Add Legend - Need to handle labels carefully due to conditional plotting
# Easiest is still to create handles for all possible valid_responses
ordered_responses_for_legend = sorted(valid_responses)
legend_handles = []
legend_labels = []
# Only add handles/labels for categories that actually exist in the data %s
existing_cats_in_plot = set(neg_cols if not is_integer_center else neg_cols + [center_col]) | set(pos_cols)
for i in ordered_responses_for_legend:
     # Check if the category exists in the calculated percentages to avoid adding legend items for missing data categories.
     # Also check if it should be plotted based on the centering logic.
    if i in df_perc.columns and df_perc[i].sum() > 0: # Check if category has data
         legend_handles.append(plt.Rectangle((0,0),1,1, color=colors[i]))
         legend_labels.append(category_map[i])

# Place legend above the plot
ax.legend(legend_handles, legend_labels, ncol=len(legend_handles), bbox_to_anchor=(0.5, 1.10), loc='upper center', frameon=False)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()