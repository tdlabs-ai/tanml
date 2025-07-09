# TanML Examples

This folder contains example YAML configurations for validating machine learning models using TanML.

## Scenarios Overview

TanML supports **three segmentation scenarios** for model validation:

---

### 🔹 Scenario A: One Model, One Cleaned Dataset (No Segmentation)

* **Path:** `examples/logistic/scenario_a/rules_logistic_scenario_a.yaml`
* **Description:**

  * A single global model trained on a cleaned dataset.
  * No segmentation is used.
* **To run:**

  ```bash
  tanml validate --rules examples/logistic/scenario_a/rules_logistic_scenario_a.yaml
  ```

---

### 🔹 Scenario B: One Model Per Segment, One Dataset Per Segment

* **Path:** `examples/logistic/scenario_b/rules_logistic_scenario_b.yaml`
* **Description:**

  * Each segment has a separate cleaned dataset.
  * One model is trained per dataset (per segment).
* **To run:**

  ```bash
  tanml validate --rules examples/logistic/scenario_b/rules_logistic_scenario_b.yaml
  ```

---

### 🔹 Scenario C: One Model Per Segment, One Dataset with Segment Column

* **Path:** `examples/logistic/scenario_c/rules_logistic_scenario_c.yaml`
* **Description:**

  * A single dataset contains a `segment_column`.
  * Each segment value maps to a corresponding model.
* **To run:**

  ```bash
  tanml validate --rules examples/logistic/scenario_c/rules_logistic_scenario_c.yaml
  ```

---

## Notes

* These examples simulate the output of `tanml init` command for each scenario.
* Ensure paths to model and dataset files are correct before running.
* You can modify thresholds, features, or validation checks in each YAML as needed.
* Similar structure can be followed under other model types (e.g., `random_forest/`).
