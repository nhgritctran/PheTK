# Plot Module

Create publication-ready visualizations of PheWAS results including Manhattan, and forest plots.

## Manhattan Plot

Display -log10(p-values) across phecode categories with effect direction indicators.

### Key Parameters
- `label_values`: What to label - "p_value", "positive_beta", "negative_beta", or specific phecodes (str/list)
- `label_count`: Number of points to label (int, default: 10)
- `label_size`: Font size for labels (int, default: 8)
- `marker_size_by_effect_size`: Scale markers by effect magnitude (bool, default: False)
- `marker_scale_factor`: Scaling factor for marker sizes (float, default: 1)
- `phecode_categories`: Specific categories to plot (list[str], optional)
- `sort_by_significance`: Sort by p-value within categories (bool, default: False)
- `save_plot`: Save plot to file (bool, default: True)
- `output_file_path`: Output path with extension (str, optional)

### Example
```python
from phetk.plot import Plot

# Create plot object
p = Plot("phewas_results.tsv", converged_only=True)

# Generate Manhattan plot
p.manhattan(
    label_values="p_value",
    label_count=10,
    save_plot=True,
    output_file_path="manhattan_plot.png"
)
```

### Advanced Customization
```python
# Custom colors and labeling
p = Plot(
    "phewas_results.tsv",
    color_palette=("blue", "red", "green"),
    bonferroni=2.5e-6
)

# Label specific phecodes with effect-based markers
p.manhattan(
    label_values=["185", "250.2"],
    marker_size_by_effect_size=True,
    marker_scale_factor=1.5,
    phecode_categories=["circulatory system", "endocrine/metabolic"]
)
```

## Forest Plot

Display effect estimates with confidence intervals for selected phecodes.

### Key Parameters
- `phecodes`: List of phecodes to include (list[str], required)
- `show_count`: Show case/control counts (bool, default: True)
- `show_p_value`: Show p-values (bool, default: True)
- `show_sex_restriction`: Show sex-specific phecodes (bool, default: True)
- `highlight_significance`: Bold significant results (bool, default: True)
- `highlight_p_value_threshold`: Threshold for highlighting (float, default: 0.05)
- `sort_by`: Sort by "p_value" or "effect" (str, optional)

### Example
```python
# Top 10 results
top_phecodes = phewas_results.sort("p_value").head(10)["phecode"].to_list()

p.forest(
    phecodes=top_phecodes,
    show_count=True,
    show_p_value=True,
    highlight_significance=True,
    sort_by="p_value",
    save_plot=True
)
```

## Plot Initialization Options

### Color Palettes
```python
# Default palette
p = Plot("results.tsv")

# Colorblind-friendly palette
p = Plot("results.tsv", color_palette="colorblind")

# Rainbow palette
p = Plot("results.tsv", color_palette="rainbow")

# Custom colors
p = Plot("results.tsv", color_palette=("blue", "red", "green"))
```

### Filtering Options
```python
# Only converged results (default)
p = Plot("results.tsv", converged_only=True)

# Include non-converged
p = Plot("results.tsv", converged_only=False)

# Custom Bonferroni threshold
p = Plot("results.tsv", bonferroni=1e-7)
```

## Common Customizations

### Label Specific Results
```python
# Label top p-values
label_values="p_value"

# Label positive effects
label_values="positive_beta"

# Label specific phecodes
label_values=["185", "250.2", "401.1"]
```

### Control Plot Appearance
```python
# High resolution
dpi=300

# Custom title
title="CFTR Variant PheWAS Results"

# Y-axis limit
y_limit=20  # Max -log10(p-value)

# No legend
show_legend=False
```

### Save Options
```python
# Auto-generated filename
save_plot=True

# Custom filename
output_file_path="results/cftr_manhattan.pdf"

# Supported formats: png, pdf, svg, jpg
```

## Volcano Plot

Volcano plot functionality is currently in development and will be available in a future release.
