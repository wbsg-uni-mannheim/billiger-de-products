# Translation audit summary

offers judged by all 3 models: 2999
models: deepseek/deepseek-v4-flash, google/gemini-3.8-flash, z-ai/glm-5.3-flash

## Majority verdict
- error-free: 2900 (96.70 %)
- minor error: 79 (2.63 %)
- critical error: 4 (0.13 %)
- no majority: 16 (0.53 %)

critical-error rate (majority): 0.13 % (Wilson 95 % CI 0.05 to 0.34 %)
offers where at least one model says critical: 75 (2.50 %)
unanimous verdicts: 2464 (82.16 %)
Fleiss' kappa over the three verdict classes: 0.107

## Per-model verdict distribution
- deepseek/deepseek-v4-flash: error-free 2763, minor error 193, critical error 43
- google/gemini-3.8-flash: error-free 2939, minor error 47, critical error 13
- z-ai/glm-5.3-flash: error-free 2645, minor error 330, critical error 24

flagged for manual review: 75 (flagged_for_review.csv)
calibration sample: 100 unanimous error-free offers (calibration_sample.csv)
