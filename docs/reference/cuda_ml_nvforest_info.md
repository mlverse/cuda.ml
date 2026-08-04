# Inspect an nvForest model

Inspect an nvForest model

## Usage

``` r
cuda_ml_nvforest_info(object)
```

## Arguments

- object:

  An nvForest-backed model.

## Value

A named list with:

- `task_type`:

  One of `"binary_classification"`, `"multiclass_classification"`, or
  `"regression"`.

- `num_classes`, `num_features`, `num_outputs`, and `num_trees`:

  Model dimensions.

- `has_vector_leaves`, `average_tree_output`, and
  `has_probability_output`:

  Logical model properties.

- `device`, `device_id`, `layout`, and `precision`:

  Resolved inference configuration.

- `default_chunk_size` and `align_bytes`:

  Native chunk and memory-alignment settings.

- `treelite_postprocessor`:

  The model's Treelite postprocessor.
