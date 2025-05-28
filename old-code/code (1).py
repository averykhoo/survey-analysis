# -----------------------------------------
# Section 0: Imports & Global Settings 
# -----------------------------------------
# ... (other imports)
import seaborn as sns # Add seaborn import

# ... (rest of Section 0)


# -----------------------------------------
# Section 9: Visualization Functions
# -----------------------------------------
# ... (plot_category_slope_chart function) ...

def plot_omega_clustermap(corr_df, output_dir):
    """
    Visualizes the correlation matrix Omega as a clustered heatmap.

    Args:
        corr_df (pd.DataFrame): DataFrame containing the mean correlation matrix
                                with category names as index and columns.
        output_dir (str): Path to the directory to save the plot.
    """
    if corr_df is None or corr_df.empty:
        print("INFO: Correlation matrix DataFrame is empty. Skipping Omega clustermap.")
        return

    n_cats = corr_df.shape[0]
    if n_cats < 2:
        print("INFO: Need at least 2 categories to create a clustermap. Skipping Omega plot.")
        return
        
    print("\n--- Generating Clustered Heatmap for Omega ---")

    # Choose appropriate figure size
    figsize = (max(8, n_cats * 0.8), max(7, n_cats * 0.7)) # Adjust size based on number of categories
    
    # Determine if annotations are feasible
    annotate = n_cats <= 15 # Only annotate if not too many categories

    try:
        # Create the clustermap
        # 'average' linkage (UPGMA) and 'euclidean' distance on correlations are common defaults
        # 'ward' linkage often produces tighter clusters
        # 'correlation' metric uses 1-|corr| as distance
        cluster_grid = sns.clustermap(
            corr_df,
            method='average',  # Linkage method: 'average', 'ward', 'complete', etc.
            metric='euclidean', # Distance metric: 'euclidean', 'correlation', etc.
            cmap='vlag',       # Diverging colormap (good for correlations around 0)
            vmin=-1, vmax=1,   # Set scale for correlation
            center=0,          # Center colormap at 0
            annot=annotate,    # Show values in cells if not too large
            fmt=".2f",         # Format for annotations
            linewidths=.5,
            linecolor='lightgray',
            figsize=figsize
        )

        # Adjustments for better readability
        plt.setp(cluster_grid.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(cluster_grid.ax_heatmap.get_yticklabels(), rotation=0)
        
        # Add a title 
        cluster_grid.fig.suptitle('Clustered Heatmap of Category Correlations (Mean Omega)', y=1.02) # Adjust y slightly above plot

        # Save the figure
        filepath = os.path.join(output_dir, 'omega_clustermap.png')
        plt.savefig(filepath, bbox_inches='tight') # Use bbox_inches='tight'
        plt.show()
        print(f"Omega clustermap saved to {filepath}")
        
        print("\nInterpretation Notes for Omega Clustermap:")
        print("- Colors indicate correlation strength (e.g., red=positive, blue=negative).")
        print("- Dendrograms (tree diagrams) show the hierarchical clustering.")
        print("- Categories grouped closely together in the dendrogram and heatmap tend to have similar correlation patterns with other categories.")
        print("- Look for distinct blocks along the diagonal, which represent clusters of highly correlated categories.")

    except Exception as e:
        print(f"WARNING: Could not generate Omega clustermap: {e}")


# -----------------------------------------
# Section 10: Main Execution Flow & Reporting
# -----------------------------------------
# ... (after diagnostics and result processing) ...

# --- Generate Visualizations ---
print("\n--- Generating Visualizations ---")
# (Slope chart generation loop) ...

# Generate Omega clustermap
if 'corr_df' in locals() and corr_df is not None:
     plot_omega_clustermap(corr_df, OUTPUT_DIR)
else:
     print("INFO: Skipping Omega clustermap as corr_df was not generated.")
     
# ... (Final Comments & Next Steps) ...