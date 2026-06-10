import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Assumed Pre-defined Variables (based on context from both scripts) ---
# df_raw: The raw DataFrame before melting
# theta_df: A DataFrame containing overall scores/rankings, with a 'year' column
#           and columns named after categories in category_mapping_initial.keys()
#           and an 'I am from' column.
# category_mapping_initial: A dictionary like {'CategoryName': ['q1', 'q2'], ...}
# RESPONSE_OPTIONS: A list of response option strings, e.g., ['Strongly Disagree', ..., 'Strongly Agree']
# engg_only_question_categories: A list of category names considered "engineering only"
# i_am_from_non_engineering: A list of team names (from 'I am from') that are non-engineering
# OUTPUT_DIR: Path to the output directory for saving plots
# category_title: This variable was in the original ax.set_title.
#                 It might be intended to be `category.title()` or a lookup.
#                 Using `category.title()` as a sensible default.
# ID_VAR = 'I am from' # Good practice to define this constant

# --- Main Script Logic ---

# Filter theta_df for the specific year
# This is a logical step you suggested and makes sense.
df_overall_year = theta_df[theta_df['year'] == 2025]

for category, questions_in_category in category_mapping_initial.items():
    print(f"Processing category: {category}")

    # Melt the raw data for the current category and year
    # It's good practice to assign the result of melt to a new variable
    current_category_long_df = df_raw[df_raw['year'] == 2025].melt(
        id_vars=['I am from'], # Using 'I am from' directly as seen; could be ID_VAR
        value_vars=questions_in_category, # Use questions for the current category
        var_name='question',          # Singular 'question' is more standard
        value_name='response'
    )

    if current_category_long_df.empty:
        print(f"No data for category {category} in year 2025. Skipping.")
        continue

    # Calculate counts and percentages for the current category across all teams
    # This ensures percentages are relevant to the plot being generated in this loop iteration
    counts_for_category = current_category_long_df.groupby(['I am from', 'response']).size().unstack(fill_value=0)
    
    if counts_for_category.empty:
        print(f"No response counts for category {category}. Skipping plot.")
        continue
        
    percentages_for_category = counts_for_category.div(counts_for_category.sum(axis=1), axis=0) * 100
    
    if percentages_for_category.empty:
        print(f"Could not calculate percentages for category {category}. Skipping plot.")
        continue

    # Get the order of teams based on their score in the current 'category'
    # Scores are assumed to be in df_overall_year, with column names matching 'category' keys
    # 'ascending=True' means worst (lower score) to best (higher score) -> bottom to top on barh
    if category in df_overall_year.columns:
        # Sort teams by their score in the current 'category'
        ordered_teams_by_score = df_overall_year.sort_values(by=category, ascending=True)['I am from'].tolist()
    else:
        print(f"Warning: Score column '{category}' not found in df_overall_year. Teams will not be sorted by score.")
        # Fallback: use the order as they appear in the percentages data
        ordered_teams_by_score = percentages_for_category.index.tolist()

    # Filter this order to include only teams present in the current percentage data
    # And maintain the score-based order for those present.
    # Add any teams from percentages_for_category not in ordered_teams_by_score (though sort_values should include all)
    teams_in_percentages = set(percentages_for_category.index)
    final_order_of_teams = [team for team in ordered_teams_by_score if team in teams_in_percentages]
    # Add any remaining teams from percentages if they weren't in df_overall_year (e.g., no score)
    final_order_of_teams.extend([team for team in percentages_for_category.index if team not in final_order_of_teams])


    # Remove non-engineering groups if this category is an "engineering only" question category
    if category in engg_only_question_categories:
        # Assuming i_am_from_non_engineering is a list of team names (values from 'I am from')
        final_order_of_teams = [team for team in final_order_of_teams if team not in i_am_from_non_engineering]

    if not final_order_of_teams:
        print(f"No teams left to plot for category {category} after filtering. Skipping.")
        continue

    # Reindex percentages based on the final order of teams
    percentages_to_plot = percentages_for_category.reindex(final_order_of_teams)
    # Reindex columns to match RESPONSE_OPTIONS order (e.g., Likert scale order)
    percentages_to_plot = percentages_to_plot.reindex(columns=RESPONSE_OPTIONS[::-1], fill_value=0) # fill_value=0 for missing response types

    # Plot
    # Adjust figsize height based on the number of teams to plot
    fig_height = max(5, len(percentages_to_plot) * 0.6) # Adjust 0.6 factor as needed
    fig, ax = plt.subplots(figsize=(10, fig_height))

    percentages_to_plot.plot(
        kind='barh',
        stacked=True,
        color=plt.cm.get_cmap('vlag', len(RESPONSE_OPTIONS))(range(len(RESPONSE_OPTIONS))), # Color map based on number of responses
        edgecolor='white',
        ax=ax
    )

    plot_title = category.title() # Use current category for title; .title() for better formatting
    ax.set_title(f'{plot_title} ({2025})\nTeam Response Distribution\n', fontsize=14) # Added a subtitle
    ax.set_xlabel('Percentage of Responses', fontsize=12)
    ax.set_ylabel('Team (Ordered by Score - Bottom to Top)', fontsize=12) # Clarify y-axis
    ax.set_xlim([0, 100])

    # Legend
    ax.legend(
        RESPONSE_OPTIONS[::-1], # Match column order
        title='Response',      # Generic title for legend
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1 if len(percentages_to_plot) > 10 else -0.15), # Adjust anchor based on plot size
        ncol=len(RESPONSE_OPTIONS),
        frameon=False,
        fontsize=10
    )

    # Add text annotations for percentages on bars
    for i, (team_name, row_values) in enumerate(percentages_to_plot.iterrows()):
        current_start_position = 0
        for j, val_percent in enumerate(row_values): # j is the index for RESPONSE_OPTIONS[::-1]
            if val_percent > 3.5: # Only add text if the segment is wide enough
                # Determine text color based on the response option (j maps to RESPONSE_OPTIONS[::-1])
                # This color logic `j in (1,6)` might need adjustment based on actual RESPONSE_OPTIONS
                text_color = 'white' if j in (1, len(RESPONSE_OPTIONS)-2) else 'black' # Example: light text for 2nd and 2nd-to-last options
                
                ax.text(
                    current_start_position + val_percent / 2,
                    i, # y-coordinate (bar index)
                    f'{val_percent:.0f}%',
                    va='center',
                    ha='center',
                    color=text_color,
                    fontsize=9 # Slightly smaller font for annotations
                )
            current_start_position += val_percent

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for legend and title
    
    # Save the figure
    safe_category_name = category.replace(" ", "_").replace("/", "_") # Make category name filename-safe
    output_filename = f'{safe_category_name}_team_responses_likert.svg'
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.show()

print("Processing complete.")