import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import textwrap

# Load and parse data with error handling
try:
    with open("../../../data/meta_info/logs/yt_review_decisions/rejected_video_ids.txt", "r") as file:
        lines = file.readlines()
    reasons = []
    for line in lines:
        if '|' in line:
            try:
                reasons.append(line.strip().split('|')[1])
            except IndexError:
                print(f"Skipping malformed line: {line.strip()}")
except FileNotFoundError:
    print("Error: rejected_video_ids.txt not found")
    exit()

reason_counts = Counter(reasons)
labels = list(reason_counts.keys())
counts = list(reason_counts.values())

# Wrap long labels
wrapped_labels = [textwrap.fill(label, 20) for label in labels]

# Set plot parameters for CVPR
plt.rcParams.update({
    "figure.figsize": (7, 4),  # Wider to accommodate annotations
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 600,
    "savefig.dpi": 600,
})

fig, ax = plt.subplots()

# Use colorblind-friendly color
colors = sns.color_palette("colorblind", n_colors=1)
bars = ax.barh(wrapped_labels, counts, color=colors[0])

# Adjust x-axis limit to accommodate annotations
max_count = max(counts)
ax.set_xlim(0, max_count * 1.2)  # 20% padding for annotations

# Add number annotations outside bars
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{int(width)}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),  # 5 points to the right
                textcoords='offset points',
                ha='left', va='center', fontsize=12)

ax.set_xlabel("Number of Videos")
ax.set_title("Distribution of Rejection Reasons (n=765)")
ax.grid(axis='x', alpha=0.5)

plt.tight_layout(pad=1.5)  # Extra padding for annotations
plt.subplots_adjust(left=0.3)  # Room for labels

# Save as high-quality PDF with embedded fonts
plt.savefig("rejection_summary.pdf", format='pdf', bbox_inches='tight', metadata={'Creator': None, 'Producer': None})
print("✅ Saved horizontal bar chart as rejection_summary.pdf")
plt.show()
