"""
run after scratch 15
"""
import pandas as pd
import matplotlib.pyplot as plt

# 1. load your raw data
# df_raw = pd.DataFrame([...])  # already in your namespace

# 2. melt into “long” format so each row is one (product, response)
response_cols = [c for c in df_raw.columns if c.startswith('qn')]
df_long = df_raw.melt(
    id_vars=['product'],
    value_vars=response_cols,
    var_name='question',
    value_name='response'
)

# 3. count & convert to percentages
counts = (
    df_long
      .groupby(['product','response'])
      .size()
      .unstack(fill_value=0)
)
percentages = counts.div(counts.sum(axis=1), axis=0) * 100

# 4. enforce your category & color order (negative → positive)
cat_order = [
    "I dislike them a lot",
    "I dislike them",
    "Neutrals and Don't Know",
    "I like them",
    "I like them a lot"
]
colors = ['#e6550d','#fdae6b','#c6c6c6','#6baed6','#1f77b4']

percentages = percentages.reindex(columns=cat_order) # order the responses 6 -> 1
# percentages = percentages.reindex(custom_order) # order the teams

# 5. plot
fig, ax = plt.subplots(figsize=(12,6))
percentages.plot(
    kind='barh',
    stacked=True,
    color=colors,
    edgecolor='white',
    linewidth=0.5,
    ax=ax
)

# 6. polish
ax.set_title(
    'Everyone likes chocolates, but Bounty and Snickers get the most extreme opinions',
    fontsize=14, weight='bold'
)
ax.set_xlabel('Percentage', fontsize=12)
ax.set_xlim(0,100)
ax.legend(
    cat_order,
    bbox_to_anchor=(0.5, -0.15),
    loc='upper center',
    ncol=len(cat_order),
    frameon=False
)

# 7. add labels
for i, row in enumerate(percentages.values):
    start = 0
    for j, val in enumerate(row):
        if val >= 5:
            ax.text(
                start + val/2, i, f"{val:.0f}%",
                va='center', ha='center',
                color='white' if j>=3 else 'black',
                fontsize=10
            )
        start += val

plt.tight_layout()
plt.show()
