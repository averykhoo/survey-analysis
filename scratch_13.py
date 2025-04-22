import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# --- 1. Sample Data (Replace with your actual data) ---
# Create data similar to the chart's structure.
# Values should ideally be percentages summing to 100 for each row.
data = {
    'I dislike them a lot': [18, 15, 5, 6, 4, 3, 4, 3],
    'I dislike them': [15, 10, 8, 8, 6, 5, 6, 5],
    'Neutrals and Don\'t Know': [20, 18, 22, 20, 15, 18, 16, 14],
    'I like them': [30, 35, 40, 42, 45, 40, 44, 48],
    'I like them a lot': [17, 22, 25, 24, 30, 34, 30, 30],
}
index = [
    'Bounty', 'Snickers', 'Milky Way', 'Mars',
    'Galaxy Caramel', 'Twix', 'Galaxy', 'Maltesers Teaser'
]

df = pd.DataFrame(data, index=index)

# Ensure data is in percentages (optional if already percentages)
# df = df.apply(lambda x: x / x.sum() * 100, axis=1)

# --- 2. Prepare data for plotting ---
categories = df.columns
df['Total'] = df.sum(axis=1) # Check if sums are close to 100 if using percentages

# Define the order and colors (adjust colors to match your preference or the original chart)
category_order = [
    'I dislike them a lot', 'I dislike them', 'Neutrals and Don\'t Know',
    'I like them', 'I like them a lot'
]
colors = ['#d6604d', '#f4a582', '#e0e0e0', '#92c5de', '#4393c3'] # Example diverging colors

# Reorder DataFrame columns
df = df[category_order]

# Calculate the midpoint offset for centering (half of the neutral category)
neutral_col = 'Neutrals and Don\'t Know'
df['MidpointOffset'] = df[neutral_col] / 2

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figsize as needed

# Get the list of items (chocolates) and their positions on the y-axis
items = df.index
y_pos = np.arange(len(items))

# Plot each category segment by segment
cumulative_pos = -df['I dislike them a lot'] - df['I dislike them'] - df['MidpointOffset'] # Start plotting from the far left

for i, cat in enumerate(category_order):
    width = df[cat]
    # For the neutral category, plot only half its width on each side
    if cat == neutral_col:
        # Plot first half (negative side)
        ax.barh(y_pos, df['MidpointOffset'], left=cumulative_pos, color=colors[i], edgecolor='white', label=cat if i == 0 else "") # Label only once for legend
        cumulative_pos += df['MidpointOffset']
        # Plot second half (positive side) - Note: width is MidpointOffset again
        ax.barh(y_pos, df['MidpointOffset'], left=cumulative_pos, color=colors[i], edgecolor='white')
        cumulative_pos += df['MidpointOffset']
    else:
        ax.barh(y_pos, width, left=cumulative_pos, color=colors[i], edgecolor='white', label=cat if i == 0 else "") # Label only once for legend
        cumulative_pos += width

# --- 4. Formatting ---
ax.set_yticks(y_pos)
ax.set_yticklabels(items)
ax.invert_yaxis()  # Display items top-to-bottom like the example

# Format x-axis as percentages
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("Percentage of Responses")

# Add a vertical line at 0% for reference
ax.axvline(0, color='grey', linewidth=0.8, linestyle='-')

# Determine x-axis limits based on data range
min_val = (-df['I dislike them a lot'] - df['I dislike them'] - df['MidpointOffset']).min()
max_val = (df['I like them a lot'] + df['I like them'] + df['MidpointOffset']).max()
ax.set_xlim(min_val - 5, max_val + 5) # Add a little padding

# Set Title
ax.set_title("Everyone likes chocolates, but Bounty and Snickers get the most extreme opinions", pad=30) # Added padding

# Add Legend - create custom legend handles to ensure correct labeling
legend_handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(category_order))]
# Place legend above the plot like in the example
ax.legend(legend_handles, category_order, ncol=len(category_order), bbox_to_anchor=(0.5, 1.10), loc='upper center', frameon=False)


# Remove chart borders (spines) for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False) # Keep or remove based on preference
#ax.spines['bottom'].set_position(('data', 0)) # Try to position bottom axis at 0 (might need tweaking)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()