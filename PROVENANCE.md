# Provenance of every notebook cell
Each code cell starts with a `# [from <notebook>, cell N]` line. Cells are copied verbatim, except for the replacements listed here.

## 02_labelled_set_and_model.ipynb
- `GradientBoosting/GradientBoostingBL_Cluster.ipynb` cell 0 (source md5 3954efaa) — imports
    - `from swapp.catalogues import catalogue_to_edges_df` → `from blxline.catalogues import catalogue_to_edges_df` (×2)
    - `from UTime.GradientBoosting import compute_scores, recall_precision_curve, choose_best_fea` → `from blxline.model import compute_scores, recall_precision_curve, choose_best_features` (×1)
    - `from UTime.GradientBoosting import get_all_feature_combinaisons_multiple_choices, find_bes` → `from blxline.model import get_all_feature_combinaisons_multiple_choices, find_best_thresho` (×1)
    - `from UTime.GradientBoosting import temporal_split, x_y_set, add_scores, compute_scores` → `from blxline.model import temporal_split, x_y_set, add_scores, compute_scores` (×1)
    - `from swapp.engineered_features.tangent_to_MP import V_normal_to_Shue, V_tan1_to_Shue, V_ta` → `` (×1)
- `GradientBoosting/GradientBoostingBL_Cluster.ipynb` cell 2 (source md5 30ee3530) — C1 labels from the catalogues (the pred_BL column read here is not used for training)
- `GradientBoosting/GradientBoostingBL_Cluster.ipynb` cell 11 (source md5 de5c7fa7) — MMS1, THA, THB labelled points
- `GradientBoosting/GradientBoostingBL_Cluster.ipynb` cell 15 (source md5 45251dc3)
- **reconstructed cell** — lost step. Bn_MP, Btan1_MP, Btan2_MP (mandatory features) on all labelled points, with the code of TrainingCommonModelChatGPT cell 10 (lines inside the docstring). Checked: same values as labelled_MMS1_THA_THB_C1_2025-08-05.pkl to 1e-14
- `GradientBoosting/GradientBoostingBL_Cluster.ipynb` cell 18 (source md5 7f5af616) — the training cell reads the undated name: written directly under that name (reconstruction of the lost step)
    - `f'/home/ghisalberti/make_datasets/labelled_MMS1_THA_THB_C1_{str(datetime.now())[:10]}.pkl'` → `f'{OUT}/labelled_MMS1_THA_THB_C1.pkl'` (×1)
- `GradientBoosting/GradientBoostingBL_Cluster.ipynb` cell 131 (source md5 3fa1c368) — trains model_MMS_THEMIS_Cluster on 100% of the labelled set
    - `'/home/ghisalberti/make_datasets/labelled_MMS1_THA_THB_C1.pkl'` → `f'{OUT}/labelled_MMS1_THA_THB_C1.pkl'` (×1)
    - `f'/home/ghisalberti/GradientBoosting/diagnostics/model_MMS_THEMIS_Cluster.pkl'` → `f'{OUT}/model_MMS_THEMIS_Cluster.pkl'` (×1)
    - `from UTime.GradientBoosting import pred_ensemble` → `from blxline.model import pred_ensemble` (×1)
    - `f'/home/ghisalberti/GradientBoosting/all_pred/pred_` → `f'{OUT}/all_pred/pred_` (×1)

## 03_predictions_and_BL_dataset.ipynb
- `GradientBoosting/CommonPred.ipynb` cell 1 (source md5 dc914486) — imports
    - `from UTime.GradientBoosting import compute_shap_values_ensemble_learning, get_all_features` → `from blxline.model import get_all_features_ensemble_model, pred_ensemble` (×1)
    - `from UTime.GradientBoosting import get_all_features_ensemble_model` → `` (×1)
    - `from swapp.dataframes import` → `from blxline.dataframes import` (×1)
    - `from swapp.make_plots.plot_maps import` → `from blxline.maps import` (×1)
- `GradientBoosting/CommonPred.ipynb` cell 2 (source md5 366e1d23) — get_common_features
- `GradientBoosting/CommonPred.ipynb` cell 3 (source md5 cb7121e6) — parameters (suffix = _without_cusps_withoutNp<3_all_sat)
- `GradientBoosting/CommonPred.ipynb` cell 6 (source md5 7489ba38) — predictions (all_sat=True -> model_MMS_THEMIS_Cluster.pkl)
    - `#for sat, mission in zip(['MMS1','THA','THB','THC','THD','THE','C1','C3'], ['MMS']+['THEMI` → `for sat, mission in zip(['MMS1','THA','THB','THC','THD','THE','C1','C3'], ['MMS']+['THEMIS` (×1)
    - `f'/home/ghisalberti/GradientBoosting/all_pred/` → `f'{OUT}/all_pred/` (×2)
- `GradientBoosting/CommonPred.ipynb` cell 20 (source md5 f55086d0) — BL selection + closest MSP/MSH
    - `#for sat, mission in zip(['MMS1','THA','THB','THC','THD','THE','C1','C3'], ['MMS']+['THEMI` → `#for sat, mission in zip(['THD'], ['THEMIS']):
#for sat, mission in zip(['THA','THB','THC'` (×1)
    - `f'/home/ghisalberti/GradientBoosting/all_pred/` → `f'{OUT}/all_pred/` (×5)
- `GradientBoosting/CommonPred.ipynb` cell 25 (source md5 8104601d) — neighbour medians (_MSP, _MSH columns)
    - `#satellites = ['MMS1','THA','THB','THC','THD','THE','C1','C3']
satellites = ['C1']` → `satellites = ['MMS1','THA','THB','THC','THD','THE','C1','C3']` (×1)
    - `f'/home/ghisalberti/GradientBoosting/all_pred/` → `f'{OUT}/all_pred/` (×5)
- `GradientBoosting/CommonPred.ipynb` cell 41 (source md5 418abfd1) — merge
    - `f'/home/ghisalberti/GradientBoosting/all_pred/` → `f'{OUT}/all_pred/` (×1)
    - `f'/DATA/ghisalberti/Datasets/BL_MMS1_THEMIS_Cluster` → `f'{OUT}/BL_MMS1_THEMIS_Cluster` (×2)
- `GradientBoosting/CommonPred.ipynb` cell 50 (source md5 1447bd23) — OMNI (Safrankova propagation)
    - `f'/DATA/ghisalberti/Datasets/BL_MMS1_THEMIS_Cluster` → `f'{OUT}/BL_MMS1_THEMIS_Cluster` (×1)

## 04_map_dataset.ipynb
- `ReorderingTemporalData/DepthComputation.ipynb` cell 0 (source md5 0ebc7a4b) — imports
    - `from UTime.GradientBoosting import choose_best_features` → `from blxline.model import choose_best_features` (×1)
    - `from swapp.make_plots.plot_maps import` → `from blxline.maps import` (×1)
- `ReorderingTemporalData/DepthComputation.ipynb` cell 11 (source md5 d76fd77a) — compute_depth, uniformize_locally (get_features_and_weights is unused)
- `ReorderingTemporalData/DepthComputation.ipynb` cell 14 (source md5 3d9cef6d) — diagnostic plots, called by the next cell
- `ReorderingTemporalData/DepthComputation.ipynb` cell 32 (source md5 dfc2fded) — tilt, depth, uniformization
    - `f'/DATA/ghisalberti/Datasets/BL_MMS1_THEMIS_Cluster_for_depth_without_cusps_withoutNp<3_al` → `f'{OUT}/BL_MMS1_THEMIS_Cluster_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)
- `ReorderingTemporalData/DepthComputation.ipynb` cell 35 (source md5 7a0f9ba5) — save the final map dataset
    - `'/home/ghisalberti/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` → `f'{OUT}/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)

## 05_maps_and_xline.ipynb
- `Maps/FinalMapsPaper.ipynb` cell 0 (source md5 d19853b7) — imports (unused swapp imports removed)
    - `from swapp.make_plots.plot_maps import make_mp_grid, gaussian_filter_nan_datas, plot_color` → `from blxline.maps import make_mp_grid, gaussian_filter_nan_datas, plot_CLA_sector
from blx` (×1)
    - `from swapp.make_plots.plot_maps import reposition, maps_by_sectors, make_slice` → `from blxline.maps import reposition, make_slice` (×1)
    - `from swapp.make_plots.plot_maps import find_stagnation_line` → `from blxline.maps import find_stagnation_line` (×1)
- `Maps/FinalMapsPaper.ipynb` cell 632 (source md5 552e1d1f) — keep_main_part (used by the JGR figure)
- `Maps/FinalMapsPaper.ipynb` cell 628 (source md5 7efea3b4) — DeltaVz_cla{cla}_tilt{tilt}.pkl
    - `from swapp.make_plots.plot_maps import gaussian_filter_nan_1d` → `from blxline.maps import gaussian_filter_nan_1d` (×1)
    - `'/home/ghisalberti/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` → `f'{OUT}/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)
    - `f'/home/ghisalberti/komar_separators/DeltaVz_cla` → `f'{OUT}/xlines/DeltaVz_cla` (×1)
- `Maps/FinalMapsPaper.ipynb` cell 629 (source md5 895bb84c) — DeltaVz_cla{cla}.pkl
    - `from swapp.make_plots.plot_maps import gaussian_filter_nan_1d` → `from blxline.maps import gaussian_filter_nan_1d` (×1)
    - `'/home/ghisalberti/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` → `f'{OUT}/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)
    - `f'/home/ghisalberti/komar_separators/DeltaVz_cla` → `f'{OUT}/xlines/DeltaVz_cla` (×1)
- `Maps/FinalMapsPaper.ipynb` cell 303 (source md5 07410319) — X-lines for the figures (cells 303 -> 288 -> 286 -> 292 ran in this order)
    - `'/home/ghisalberti/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` → `f'{OUT}/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)
- `Maps/FinalMapsPaper.ipynb` cell 288 (source md5 b67523b2)
- `Maps/FinalMapsPaper.ipynb` cell 286 (source md5 e41caa7a)
    - `'/home/ghisalberti/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` → `f'{OUT}/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)
    - `'/home/ghisalberti/Figures/maps_clas_zoom1.png'` → `f'{OUT}/figures/maps_clas_zoom1.png'` (×1)
    - `'/home/ghisalberti/Figures/maps_clas_zoom2.png'` → `f'{OUT}/figures/maps_clas_zoom2.png'` (×1)
- `Maps/FinalMapsPaper.ipynb` cell 292 (source md5 4401b212)
    - `'/home/ghisalberti/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` → `f'{OUT}/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)
    - `'/home/ghisalberti/Figures/maps_tilts_zoom1.png'` → `f'{OUT}/figures/maps_tilts_zoom1.png'` (×1)
    - `'/home/ghisalberti/Figures/maps_tilts_zoom2.png'` → `f'{OUT}/figures/maps_tilts_zoom2.png'` (×1)
- `Maps/FinalMapsPaper.ipynb` cell 630 (source md5 6ee1d1c9) — comparison_models_jgr3.png, first pass: run=1 needs the maps computed by a previous pass, so set to 0 here
    - `from swapp.make_plots.plot_maps import gaussian_filter_nan_1d` → `from blxline.maps import gaussian_filter_nan_1d` (×1)
    - `'/home/ghisalberti/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` → `f'{OUT}/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)
    - `run = 1` → `run = 0` (×1)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×3)
    - `'/home/ghisalberti/Figures/comparison_models_jgr3.png'` → `f'{OUT}/figures/comparison_models_jgr3.png'` (×1)
- `Maps/FinalMapsPaper.ipynb` cell 630 (source md5 6ee1d1c9) — comparison_models_jgr3.png, second pass (run=1, verbatim): reuses the maps of the first pass with a common colour scale, as in the paper figure
    - `from swapp.make_plots.plot_maps import gaussian_filter_nan_1d` → `from blxline.maps import gaussian_filter_nan_1d` (×1)
    - `'/home/ghisalberti/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` → `f'{OUT}/BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl'` (×1)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×3)
    - `'/home/ghisalberti/Figures/comparison_models_jgr3.png'` → `f'{OUT}/figures/comparison_models_jgr3.png'` (×1)

## 06_parametric_study.ipynb
- `Article3.ipynb` cell 0 (markdown)
- `Article3.ipynb` cell 1 (source md5 a370b311)
- `Article3.ipynb` cell 2 (source md5 552e1d1f)
- `Article3.ipynb` cell 3 (source md5 a9717717)
- `Article3.ipynb` cell 4 (markdown)
- `Article3.ipynb` cell 5 (source md5 a0a9b8b3)
    - `f'/home/ghisalberti/komar_separators/DeltaVz_cla` → `f'{OUT}/xlines/DeltaVz_cla` (×1)
- `Article3.ipynb` cell 6 (source md5 98a431c6)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×1)
- `Article3.ipynb` cell 7 (source md5 e14f3aec)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×1)
- `Article3.ipynb` cell 8 (markdown)
- `Article3.ipynb` cell 9 (source md5 25084bb1)
    - `f'/home/ghisalberti/komar_separators/DeltaVz_cla` → `f'{OUT}/xlines/DeltaVz_cla` (×1)
- `Article3.ipynb` cell 11 (markdown)
- `Article3.ipynb` cell 13 (markdown)
- `Article3.ipynb` cell 14 (markdown)
- `Article3.ipynb` cell 15 (markdown)
- `Article3.ipynb` cell 16 (source md5 fc922b7f)
- `Article3.ipynb` cell 17 (source md5 1fec5b9b)
    - `f'/home/ghisalberti/komar_separators/DeltaVz_cla` → `f'{OUT}/xlines/DeltaVz_cla` (×1)
- `Article3.ipynb` cell 18 (source md5 98eef232)
- `Article3.ipynb` cell 19 (source md5 0afc4e19)
    - `f'/home/ghisalberti/komar_separators/DeltaVz_cla` → `f'{OUT}/xlines/DeltaVz_cla` (×1)
- `Article3.ipynb` cell 20 (source md5 3f299a58)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×1)
- `Article3.ipynb` cell 21 (source md5 98eef232)
- `Article3.ipynb` cell 22 (source md5 8c1c3c43)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×1)
- `Article3.ipynb` cell 23 (source md5 98eef232)
- `Article3.ipynb` cell 24 (markdown)
- `Article3.ipynb` cell 25 (markdown)
- `Article3.ipynb` cell 26 (source md5 55c71639)
- `Article3.ipynb` cell 27 (source md5 f454c269)
    - `f'/home/ghisalberti/komar_separators/DeltaVz_cla` → `f'{OUT}/xlines/DeltaVz_cla` (×1)
- `Article3.ipynb` cell 28 (source md5 09ab5856)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×1)
- `Article3.ipynb` cell 29 (source md5 ca71441c)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×1)
- `Article3.ipynb` cell 30 (source md5 52b5b8a1)
- `Article3.ipynb` cell 31 (markdown)
- `Article3.ipynb` cell 32 (source md5 364ca7e2)
    - `f'/home/ghisalberti/komar_separators/DeltaVz_cla` → `f'{OUT}/xlines/DeltaVz_cla` (×1)
    - `f'/home/ghisalberti/komar_separators/` → `f'{MODEL_LINES}/` (×2)
