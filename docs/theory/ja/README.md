# `bochan` 理論リファレンス

[理論ドキュメント入口](../README.md) · [English](../en/README.md) · **日本語**

このディレクトリは、数理的な問題設定、確率モデル、逐次意思決定、実装契約、Materials Informatics / MLIP / Closed-loop Materials Discoveryを一貫して理解するための理論リファレンスです。

## Part I. 基礎

| 章 | ファイル |
|---:|---|
| 00 | `00_overview.md` |
| 01 | `01_gaussian_process_models.md` |
| 02 | `02_bayesian_optimization.md` |
| 03 | `03_acquisition_functions.md` |
| 03b | `03b_information_theoretic_acquisitions.md` |
| 03c | `03c_knowledge_gradient.md` |
| 03d | `03d_multiobjective_entropy_search.md` |
| 03e | `03e_binary_knowledge_gradient.md` |
| 03f | `03f_ordinal_knowledge_gradient.md` |
| 04 | `04_active_learning.md` |
| 05 | `05_level_set_estimation.md` |
| 06 | `06_classification_and_ordinal_bo.md` |
| 07 | `07_multi_objective_and_constraints.md` |
| 08 | `08_input_perturbation_and_risk.md` |
| 09 | `09_shape_conventions.md` |

## Part II. モデル族

| 章 | ファイル |
|---:|---|
| 10 | `10_regression_models_and_likelihoods.md` |
| 11 | `11_classification_models.md` |
| 12 | `12_ordinal_models.md` |
| 13 | `13_heteroscedastic_and_robust_models.md` |
| 14 | `14_deep_and_high_dimensional_models.md` |
| 15 | `15_heterogeneous_multi_output.md` |
| 16 | `16_level_set_mathematics_and_implementation.md` |

## Part III. Materials基礎

| 章 | ファイル |
|---:|---|
| 17 | `17_materials_informatics_and_representations.md` |
| 18 | `18_machine_learning_interatomic_potentials.md` |
| 19 | `19_mlip_residual_gaussian_process.md` |
| 20 | `20_structure_relaxation_and_bayesian_optimization.md` |
| 21 | `21_composition_models.md` |
| 22 | `22_composition_to_structure.md` |
| 23 | `23_composition_structure_process_optimization.md` |
| 24 | `24_materials_model_selection.md` |

## Part IV. 選択ガイド

| 章 | ファイル |
|---:|---|
| 25 | `25_acquisition_selection_and_implementation.md` |
| 26 | `26_active_learning_selection_and_implementation.md` |
| 27 | `27_gp_model_selection.md` |
| 28 | `28_non_gaussian_and_noise_model_selection.md` |

## Part V. Advanced Bayesian Optimization

| 章 | ファイル |
|---:|---|
| 29 | `29_multifidelity_multitask_transfer_residual.md` |
| 30 | `30_multifidelity_bayesian_optimization.md` |
| 31 | `31_robust_bayesian_optimization.md` |
| 32 | `32_constrained_bayesian_optimization.md` |
| 33 | `33_multiobjective_bo_practice.md` |
| 34 | `34_batch_parallel_bayesian_optimization.md` |
| 35 | `35_lookahead_nonmyopic_bo.md` |
| 36 | `36_information_theoretic_bo.md` |
| 37 | `37_classification_ordinal_bo_practice.md` |
| 38 | `38_level_set_boundary_search.md` |
| 39 | `39_mixed_discrete_combinatorial_bo.md` |
| 40 | `40_high_dimensional_bayesian_optimization.md` |
| 41 | `41_bo_diagnostics_and_failure_modes.md` |

## Part VI. Materials Discovery

| 章 | ファイル |
|---:|---|
| 42 | `42_composition_space_exploration.md` |
| 43 | `43_crystal_structure_exploration.md` |
| 44 | `44_hierarchical_mlip_dft_experiment.md` |
| 45 | `45_materials_active_learning.md` |
| 46 | `46_closed_loop_materials_discovery.md` |

基本の読書順は `00 -> 01 -> 02 -> 03` とし、その後に目的に応じてモデル選択・高度BO・Materials章へ進む構成です。