# Plotting

## Aligned Clustering Results

### Scatter plot of memberships

#### Cluster memberships
::: ace_of_clust.plot
    options:
      show_root_heading: false
      heading_level: 5
      members:
        - scatter_by_cluster
        - plot_single_spatial_membership
        - plot_single_cluster_in_grid
        - separate_scatter_for_cluster_mode
        - overlay_scatter_for_mode
        - plot_compmodels_membership_grid
        - plot_compmodels_membership_selected
        - plot_ref_alt_mapping_grid

#### Differences in cluster memberships / comparisons
::: ace_of_clust.plot
    options:
      show_root_heading: false
      heading_level: 5
      members:
        - plot_compmodels_diff_grid_against_ref
        - plot_compmodels_diff_selected_against_ref
        - plot_cross_model_membership_diff_heatmap

### Structure plot of memberships
::: ace_of_clust.plot
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - plot_membership_clsind_reordered
        - plot_structure_modes_one_level
        - plot_structure_modes_two_level

### Alignment pattern plot
::: ace_of_clust.plot
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - plot_compmodels_alignment_list
        - plot_compmodels_alignment_by_model
        - plot_pair_mapping_alignment

### Joint plot of two or more components
::: ace_of_clust.plot
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - plot_multimodel_major_and_weighted_diff
        - plot_spatial_and_structure_membership_grid

### Other
::: ace_of_clust.plot
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - plot_mode_Q_heatmap
        - plot_all_modes_Q_grid
        - plot_mode_cluster_bars
        - plot_multimodel_avg_membership_barh

## Feature-Level Analysis
::: ace_of_clust.plot
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - plot_feature_scatter
        - plot_feature_kde_with_outliers
        - get_feature_kde_outliers
        - plot_top_features_bar
        - plot_P_sorted
        - plot_mode_P_sorted
        - plot_mode_sepLFC_distribution
        - plot_top_sepLFC_labels
        - plot_mode_metrics_sepCls
        - plot_selected_feature_pvs_across_modes
        - plot_feature_sepLFC_across_modes
        - plot_separated_clusters_for_selected_feature

## Other
::: ace_of_clust.plot
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - plot_discrete_colorbar
        - plot_feature_count
        - plot_mode_annotation_group_diff
