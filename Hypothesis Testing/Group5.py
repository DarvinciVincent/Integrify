import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%
data = pd.read_csv("C:/Users/darvi/Desktop/GitHub/Python-Assignment1/Group5.csv")
# Solution 1
stats = pd.DataFrame()
stats["central_tendency"] = data.sum() / len(data)
stats["mean"]=data.mean()
stats["Std.Dev"]=data.std()
stats["Var"]=data.var()

# Solution 2
data.describe()
print(data.describe())
data.hist(figsize=(20, 10), grid=False, layout=(1, 3), bins=18, color='#86bf91', zorder=2, rwidth=0.9)
# %%

# Calculate percentiles
percentiles = np.percentile(data["S1"], [5, 25, 50, 75, 95])

# Set plot style and size
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram and KDE
sns.histplot(data=data, x="S1", bins=18, kde=True, stat="density", alpha=0.65, ax=ax)
sns.kdeplot(data=data, x="S1", ax=ax, color = "red")

# Add percentile lines and annotations
for pctl, alpha, ypos in zip(percentiles, [0.6, 0.8, 1, 0.8, 0.6], [0.16, 0.26, 0.36, 0.46, 0.56]):
    ax.axvline(x=pctl, alpha=alpha, ymax=ypos, linestyle=":", color='purple', linewidth=2)
    ax.text(x=pctl-0.1, y=ypos+0.03, s=f"{int(pctl)}th", size=12, alpha=alpha)

# Set axis labels and title
ax.set_xlabel("S1", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_title("Distribution and Percentiles of S1", fontsize=16)

# Show plot
plt.show()


# %%
Colors = ["Red", "Green", "Blue", "Blue", "Green", "Red", "Yellow", "Orange", "Purple", "Blue", "Green", "Red", "Blue", "Yellow", "Orange", "Green", "Blue", "Red", "Yellow", "Green"]

"""
colors_dict = {}
for i in Colors:
    if i not in colors_dict.keys():
        colors_dict[i] = 0
    else:
        colors_dict[i] += 1

for color in colors_dict.keys():
    colors_dict[color] = [colors_dict[color]]   
        
colors_df = pd.DataFrame.from_dict(colors_dict, orient = 'columns')
colors_df.describe()
"""

d = {x: i for i, x in enumerate(set(Colors))}
lst_new = [d[x] for x in Colors]
categorized_color = np.array(lst_new)
print(np.quantile(categorized_color, 0.25))
print(np.quantile(categorized_color, 0.5))
print(np.quantile(categorized_color, 0.75))

data = pd.DataFrame({"Values": range(len(Colors)), "Colors": Colors})

# Calculate the percentage of values for each color
percentages = data["Colors"].value_counts(normalize=True) * 100

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(percentages.index, percentages.values, color=percentages.index)
ax.set_xlabel("Colors")
ax.set_ylabel("Percentage")
ax.set_title("Percentage of Colors")
ax.grid(False)

# Add the percentage values as text on top of the bars
for i, v in enumerate(percentages.values):
    ax.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=8)

plt.show()