<!-- markdownlint-disable -->

# API Overview

## Data
- [`data.data_loader`](data.data_loader.md#module-datadata_loader): Data Loader.
- [`data.preprocessing.base_scaler`](data.preprocessing.base_scaler.md#module-datapreprocessingbase_scaler): Base Scaler.
- [`data.preprocessing.sklearn_scaler`](data.preprocessing.sklearn_scaler.md#module-datapreprocessingsklearn_scaler): Sklearn Scaling Function Wrapper.

## Change Detection
- [`change_detection.base_change_detector`](change_detection.base_change_detector.md#module-change_detectionbase_change_detector): Base Change Detector.
- [`change_detection.evaluation.change_detection_evaluator`](change_detection.evaluation.change_detection_evaluator.md#module-change_detectionevaluationchange_detection_evaluator): Change Detection Evaluator.
- [`change_detection.evaluation.measures.detected_change_rate`](change_detection.evaluation.measures.detected_change_rate.md#module-change_detectionevaluationmeasuresdetected_change_rate): Detected Change Rate Measure.
- [`change_detection.evaluation.measures.detection_delay`](change_detection.evaluation.measures.detection_delay.md#module-change_detectionevaluationmeasuresdetection_delay): Detection Delay Measure.
- [`change_detection.evaluation.measures.false_discovery_rate`](change_detection.evaluation.measures.false_discovery_rate.md#module-change_detectionevaluationmeasuresfalse_discovery_rate): False Discovery Rate Measure.
- [`change_detection.evaluation.measures.mean_time_ratio`](change_detection.evaluation.measures.mean_time_ratio.md#module-change_detectionevaluationmeasuresmean_time_ratio): Mean Time Ratio Measure.
- [`change_detection.evaluation.measures.missed_detection_rate`](change_detection.evaluation.measures.missed_detection_rate.md#module-change_detectionevaluationmeasuresmissed_detection_rate): Missed Detection Rate Measure.
- [`change_detection.evaluation.measures.time_between_false_alarms`](change_detection.evaluation.measures.time_between_false_alarms.md#module-change_detectionevaluationmeasurestime_between_false_alarms): Time Between False Alarms Measure.
- [`change_detection.erics`](change_detection.erics.md#module-change_detectionerics): ERICS Change Detection Method.
- [`change_detection.river.river_change_detector`](change_detection.river.river_change_detector.md#module-change_detectionriverriver_change_detector): River Change Detection Model Wrapper.
- [`change_detection.skmultiflow.skmultiflow_change_detector`](change_detection.skmultiflow.skmultiflow_change_detector.md#module-change_detectionskmultiflowskmultiflow_change_detector): Scikit-Multiflow Change Detection Model Wrapper.
- [`change_detection.tornado.adwin`](change_detection.tornado.adwin.md#module-change_detectiontornadoadwin): Adaptive Windowing Drift Detection Method.
- [`change_detection.tornado.cusum`](change_detection.tornado.cusum.md#module-change_detectiontornadocusum): Cumulative Sum Drift Detection Method.
- [`change_detection.tornado.ddm`](change_detection.tornado.ddm.md#module-change_detectiontornadoddm): Drift Detection Method.
- [`change_detection.tornado.eddm`](change_detection.tornado.eddm.md#module-change_detectiontornadoeddm): Early Drift Detection Method.
- [`change_detection.tornado.ewma`](change_detection.tornado.ewma.md#module-change_detectiontornadoewma): Exponentially Weigthed Moving Average Drift Detection Method.
- [`change_detection.tornado.fhddm`](change_detection.tornado.fhddm.md#module-change_detectiontornadofhddm): Fast Hoeffding Drift Detection Method.
- [`change_detection.tornado.fhddms`](change_detection.tornado.fhddms.md#module-change_detectiontornadofhddms): Stacking Fast Hoeffding Drift Detection Method.
- [`change_detection.tornado.fhddms_add`](change_detection.tornado.fhddms_add.md#module-change_detectiontornadofhddms_add): Additive Stacking Fast Hoeffding Drift Detection Method.
- [`change_detection.tornado.hddm_a`](change_detection.tornado.hddm_a.md#module-change_detectiontornadohddm_a): Hoeffding's Bound based Drift Detection Method (A_test Scheme).
- [`change_detection.tornado.hddm_w`](change_detection.tornado.hddm_w.md#module-change_detectiontornadohddm_w): Hoeffding's Bound based Drift Detection Method (W_test Scheme).
- [`change_detection.tornado.mddm_a`](change_detection.tornado.mddm_a.md#module-change_detectiontornadomddm_a): McDiarmid Drift Detection Method (Arithmetic Scheme).
- [`change_detection.tornado.mddm_e`](change_detection.tornado.mddm_e.md#module-change_detectiontornadomddm_e): McDiarmid Drift Detection Method (Euler Scheme).
- [`change_detection.tornado.mddm_g`](change_detection.tornado.mddm_g.md#module-change_detectiontornadomddm_g): McDiarmid Drift Detection Method (Geometric Scheme).
- [`change_detection.tornado.page_hinkley`](change_detection.tornado.page_hinkley.md#module-change_detectiontornadopage_hinkley): Page Hinkley Drift Detection Method.
- [`change_detection.tornado.rddm`](change_detection.tornado.rddm.md#module-change_detectiontornadorddm): Reactive Drift Detection Method.
- [`change_detection.tornado.seqdrift2`](change_detection.tornado.seqdrift2.md#module-change_detectiontornadoseqdrift2): SeqDrift2 Drift Detection Method.

## Online Feature Selection
- [`feature_selection.base_feature_selector`](feature_selection.base_feature_selector.md#module-feature_selectionbase_feature_selector): Base Online Feature Selector.
- [`feature_selection.evaluation.feature_selection_evaluator`](feature_selection.evaluation.feature_selection_evaluator.md#module-feature_selectionevaluationfeature_selection_evaluator): Online Feature Selection Evaluator.
- [`feature_selection.evaluation.measures.nogueira_stability`](feature_selection.evaluation.measures.nogueira_stability.md#module-feature_selectionevaluationmeasuresnogueira_stability): Nogueira Feature Set Stability Measure.
- [`feature_selection.efs`](feature_selection.efs.md#module-feature_selectionefs): Extremal Feature Selection Method.
- [`feature_selection.fires`](feature_selection.fires.md#module-feature_selectionfires): FIRES Feature Selection Method.
- [`feature_selection.fsds`](feature_selection.fsds.md#module-feature_selectionfsds): Fast Feature Selection on Data Streams Method.
- [`feature_selection.ofs`](feature_selection.ofs.md#module-feature_selectionofs): Online Feature Selection Method.
- [`feature_selection.river.river_feature_selector`](feature_selection.river.river_feature_selector.md#module-feature_selectionriverriver_feature_selector): River Feature Selection Model Wrapper.

## Prediction
- [`prediction.base_predictor`](prediction.base_predictor.md#module-predictionbase_predictor): Base Online Predictor.
- [`prediction.evaluation.prediction_evaluator`](prediction.evaluation.prediction_evaluator.md#module-predictionevaluationprediction_evaluator): Predictive Model Evaluator.
- [`prediction.evaluation.measures.mean_drift_performance_deterioration`](prediction.evaluation.measures.mean_drift_performance_deterioration.md#module-predictionevaluationmeasuresmean_drift_performance_deterioration): Drift Performance Deterioration Measure.
- [`prediction.evaluation.measures.mean_drift_restoration_time`](prediction.evaluation.measures.mean_drift_restoration_time.md#module-predictionevaluationmeasuresmean_drift_restoration_time): Drift Restoration Time Measure.
- [`prediction.evaluation.measures.noise_variability`](prediction.evaluation.measures.noise_variability.md#module-predictionevaluationmeasuresnoise_variability): Noise Variability Measure.
- [`prediction.evaluation.measures.river_classification_metric`](prediction.evaluation.measures.river_classification_metric.md#module-predictionevaluationmeasuresriver_classification_metric): River Classification Metric Wrapper.
- [`prediction.river.river_classifier`](prediction.river.river_classifier.md#module-predictionriverriver_classifier): River Predictive Model Wrapper.
- [`prediction.skmultiflow.skmultiflow_classifier`](prediction.skmultiflow.skmultiflow_classifier.md#module-predictionskmultiflowskmultiflow_classifier): Scikit-Multiflow Predictive Model Wrapper.

## Pipeline
- [`pipeline.base_pipeline`](pipeline.base_pipeline.md#module-pipelinebase_pipeline): Base Pipeline.
- [`pipeline.distributed_fold_pipeline`](pipeline.distributed_fold_pipeline.md#module-pipelinedistributed_fold_pipeline): Distributed Fold Pipeline.
- [`pipeline.holdout_pipeline`](pipeline.holdout_pipeline.md#module-pipelineholdout_pipeline): Periodic Holdout Pipeline.
- [`pipeline.prequential_pipeline`](pipeline.prequential_pipeline.md#module-pipelineprequential_pipeline): Prequential Pipeline.
- [`pipeline.utils_pipeline`](pipeline.utils_pipeline.md#module-pipelineutils_pipeline): Pipeline Utils.

## Visualization
- [`visualization.bar`](visualization.bar.md#module-visualizationbar): Standard Bar Plot.
- [`visualization.change_detection_scatter`](visualization.change_detection_scatter.md#module-visualizationchange_detection_scatter): Change Detection Scatter Plot.
- [`visualization.feature_selection_bar`](visualization.feature_selection_bar.md#module-visualizationfeature_selection_bar): Feature Selection Bar Plot.
- [`visualization.feature_selection_scatter`](visualization.feature_selection_scatter.md#module-visualizationfeature_selection_scatter): Feature Selection Scatter Plot.
- [`visualization.feature_weight_box`](visualization.feature_weight_box.md#module-visualizationfeature_weight_box): Feature Weight Box Plot.
- [`visualization.plot`](visualization.plot.md#module-visualizationplot): Standard Line Plot.
- [`visualization.scatter`](visualization.scatter.md#module-visualizationscatter): Standard Scatter Plot.
- [`visualization.spider_chart`](visualization.spider_chart.md#module-visualizationspider_chart): Spider Plot.

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
