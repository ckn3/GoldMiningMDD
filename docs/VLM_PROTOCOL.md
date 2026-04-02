# ELDOR VLM Evaluation Protocol

## Scope

This document defines a controlled protocol for evaluating vision-language models (VLMs) on ELDOR without any additional model training. The goal is to test pretrained or directly prompted models under the fixed ELDOR label space, not to benchmark open-ended captioning or unconstrained VQA.

The evaluation covers two tasks:

1. `VLM-based multi-label presence prediction`
2. `VLM-based segmentation` for models that explicitly support segmentation or grounded mask prediction

Not every model needs to support every protocol. Each model should be evaluated under all protocols that are technically supported by its native interface. If a protocol is not supported by a model, it should be skipped rather than forced through an unnatural adaptation.

The canonical label space contains the following 14 foreground classes:

1. Building
2. Mining raft
3. Primary Forest
4. Heavy machinery
5. Water bodies
6. Agricultural crop
7. Compact mounds
8. Gravel mounds
9. Grass
10. Type 1 natural regeneration
11. Type 2 natural regeneration
12. Bare ground
13. Sluice
14. Vehicles

Background is not a target class in this protocol.

---

## 1. Evaluation Data Regimes

Since no VLM training is performed here, testing should be reported under two regimes.

### Regime A: Full-dataset evaluation

Evaluate on all available image patches, while also reporting results separately for the predefined `train`, `val`, and `test` splits.

This regime answers:

- how the VLM behaves over the full ELDOR distribution
- whether there is any large split-specific shift in performance
- whether split-specific difficulty patterns are visible without model fitting

### Regime B: Test-only evaluation

Evaluate only on the ELDOR `test` split.

This regime is required to remain directly comparable with the existing supervised benchmark and all previously reported recognition and segmentation results.

Unless otherwise stated:

- all VLM results should be reported for both regimes
- the `test`-only regime should be treated as the primary comparison setting when aligning with prior supervised results

The underlying patch data are the same `512x512` ELDOR cropped patches used in the segmentation benchmark:

- `train`: 65,798 patches
- `val`: 15,988 patches
- `test`: 40,095 patches

For each patch, the ground-truth semantic mask is converted into:

- a `14-dim multi-label presence vector` for recognition
- a `14-class raster target` for segmentation

Ground-truth class presence is defined as:

- class `c` is present if at least one valid pixel in the ground-truth mask belongs to class `c`

No class-presence threshold is used on the ground truth.

---

## 2. VLM-Based Multi-Label Presence Prediction

### 2.1 Core framing

This task should be framed as:

`VLM-based multi-label presence prediction`

Input:

- one UAV patch or sub-image

Output:

- which of the 14 ELDOR classes are present

This should not be evaluated as free-form descriptive VQA. The model is evaluated under a fixed closed label space.

### 2.2 Main protocol design

### Protocol A: Per-class binary presence testing

For each image and each class, test whether the class is present. This produces 14 decisions per image.

This protocol is the primary recognition protocol because it is:

- the most stable
- the easiest to standardize across different model families
- the easiest to parse automatically
- the easiest to score consistently
- the most appropriate for the main comparison table

However, the exact prompt form should follow the model type.

#### Protocol A1: Generative/chat VLMs

For instruction-following or chat-style VLMs, use binary QA.

Template:

`Is there [CLASS] in this image? Answer yes or no.`

Examples:

- `Is there bare ground in this image? Answer yes or no.`
- `Is there mining raft in this image? Answer yes or no.`
- `Is there type 1 natural regeneration in this image? Answer yes or no.`

For these models:

- the binary output is the final decision
- if token probabilities are available, the `yes` probability should be used as the score for `mAP`

#### Protocol A2: CLIP-like contrastive VLMs

For CLIP-like models or retrieval-style vision-language encoders, binary QA is usually not the right interface. These models should instead be evaluated with paired text statements or paired prompts, and the higher-scoring statement determines the decision.

Recommended positive/negative template pair:

- positive: `There is [CLASS] in this image.`
- negative: `There is no [CLASS] in this image.`

Examples:

- `There is bare ground in this image.` vs. `There is no bare ground in this image.`
- `There is mining raft in this image.` vs. `There is no mining raft in this image.`
- `There is type 1 natural regeneration in this image.` vs. `There is no type 1 natural regeneration in this image.`

For these models:

- compute the image-text similarity for both statements
- use the higher-scoring statement as the binary prediction
- use the positive-statement score, or the positive-vs-negative normalized score, as the ranking score for `mAP`

This is the CLIP-compatible version of Protocol A and should be treated as equivalent in intent to binary QA.

### Protocol B: Closed-set category listing

For each image, ask the model once to list all present classes from the fixed class set.

Template:

`From the following 14 categories, list all classes present in this image. Only output class names from the provided list.`

This protocol is useful because it is:

- closer to natural multi-label recognition
- more efficient than 14 separate binary prompts
- a useful supplementary comparison

However, it is less stable than Protocol A because it is more sensitive to output formatting and generation behavior.

### Protocol C: Definition-enhanced binary QA

For each image and class, prepend a short class definition and then ask the binary question.

Template:

`[CLASS DEFINITION] Is there [CLASS] in this image? Answer yes or no.`

Example:

`Type 1 natural regeneration refers to early-stage secondary vegetation dominated by low non-woody vegetation. Is there type 1 natural regeneration in this image? Answer yes or no.`

This protocol measures whether explicit semantic grounding helps VLM recognition on visually ambiguous ELDOR classes.

### 2.3 Prompt sets

### Prompt Set 1: Binary presence prompts

For generative or chat VLMs:

`Is there [CLASS] in this image? Answer yes or no.`

For CLIP-like contrastive VLMs:

- `There is [CLASS] in this image.`
- `There is no [CLASS] in this image.`

### Prompt Set 2: Closed-set listing

`Given the class list [Building, Mining raft, Primary Forest, Heavy machinery, Water bodies, Agricultural crop, Compact mounds, Gravel mounds, Grass, Type 1 natural regeneration, Type 2 natural regeneration, Bare ground, Sluice, Vehicles], output all categories present in the image. Only output class names from the list.`

### Prompt Set 3: Definition-enhanced

`[DEFINITION]. Is there [CLASS] in this image? Answer yes or no.`

### 2.4 Class definitions for Prompt Set 3

These short definitions should be used for definition-enhanced prompting. They are adapted from the ELDOR class cards and should remain fixed across all VLMs.

- `Building`: Human-built infrastructure such as houses, dwellings, and organized built-up areas, often with colored metal roofs and access roads or navigable connections.
- `Mining raft`: Rectangular or square floating mining platforms with dark or blue roofs and metal floats, found in rivers, streams, lakes, or mining ponds.
- `Primary Forest`: Old-growth or minimally disturbed arboreal forest with dense tree cover and little or no direct human intervention.
- `Heavy machinery`: Large mining machinery that cannot be reliably distinguished as a front loader, excavator, or dump truck.
- `Water bodies`: Water-filled depressions, ponds, hollows, or channels associated with mining activity or natural water accumulation.
- `Agricultural crop`: Ordered crop areas used for staple foods or fruit production, often showing regular spacing or cultivation patterns.
- `Compact mounds`: Large truncated-cone mounds of gravel, sand, and stone formed by heavy machinery, typically larger and taller than gravel mounds.
- `Gravel mounds`: Smaller cone-shaped piles of gravel, sand, and loose stones deposited by suction-pump mining operations.
- `Grass`: Pasture or grassy areas used to feed and house cattle.
- `Type 1 natural regeneration`: Early-stage secondary vegetation dominated by low non-woody plants such as grasses and lianas, usually under about 3 meters.
- `Type 2 natural regeneration`: Later-stage secondary vegetation dominated by woody plants such as shrubs, trees, and lianas, typically around 5 to 15 meters tall.
- `Bare ground`: Unvegetated exposed ground, often appearing in gray shades and commonly associated with mining disturbance or abandoned surfaces.
- `Sluice`: Rectangular wooden mining infrastructure near pond edges, typically connected to pipes and used to wash gold-bearing material.
- `Vehicles`: Transportation vehicles such as pick-up trucks or cars, usually near camps or operational zones.

### 2.5 Output normalization

To ensure automatic scoring, VLM outputs should be normalized before evaluation.

#### For Protocol A1

Accepted positive outputs:

- `yes`
- `Yes`
- `YES`

Accepted negative outputs:

- `no`
- `No`
- `NO`

Any other output should be treated as invalid.

#### For Protocol A2

For CLIP-like models:

- compute paired prompt scores for the positive and negative statements
- predict presence if the positive score is larger than the negative score
- optionally convert the paired scores into a normalized confidence score with a softmax over the two prompts

#### For Protocol B

The output should be normalized against the fixed 14-class vocabulary.

Rules:

- only class names from the provided list are valid
- duplicate class names are removed
- spelling variants should be mapped only if they are explicitly defined in a normalization dictionary
- classes outside the 14-class label space are ignored

### 2.6 Confidence scores for mAP

Since `mAP` requires a per-image, per-class score rather than only a binary decision, the scoring rule must be fixed.

Recommended priority:

1. For generative/chat VLMs, if the API exposes token probabilities or answer logits:
   - use the probability of `yes` for Protocol A1
2. For CLIP-like contrastive VLMs:
   - use the positive-prompt similarity score
   - or use the softmax-normalized positive-vs-negative score
3. If a generative model does not expose logits:
   - request a confidence score in a constrained format, for example:
     - `Answer yes or no and give a confidence score from 0 to 100.`
   - convert confidence to `[0,1]`
4. If neither is available:
   - report all non-ranking classification metrics
   - treat `mAP` as not available for that specific model/protocol

For Protocol B, `mAP` is less reliable because list generation typically does not expose class-specific confidence scores. Therefore, `mAP` should mainly be associated with Protocol A whenever possible.

### 2.7 Recognition metrics

For multi-label presence prediction, report all of the following metrics:

- macro precision (`CP`)
- macro recall (`CR`)
- macro F1 (`CF1`)
- micro precision (`OP`)
- micro recall (`OR`)
- micro F1 (`OF1`)
- `Macro-F1`
- `Micro-F1`
- `Sample-F1`
- mean average precision (`mAP`)

Per-class appendix metrics should also be reported:

- per-class precision
- per-class recall
- per-class F1
- per-class AP

No metric family should be omitted at the protocol stage. Everything should be reported, and later selection for the paper can be done from the full result set.

---

## 3. VLM-Based Segmentation

This section applies only to models that natively support grounded segmentation, prompted segmentation, mask decoding, or equivalent spatial output. If a model does not support segmentation, this section should be skipped for that model.

At present, the practically relevant protocol is `Protocol S1`. `Protocol S2` is not expected to be broadly applicable under the current model set.

### 3.1 Goal

The segmentation evaluation should measure whether a VLM can produce spatially meaningful class masks under the ELDOR label space.

The evaluation target remains the same 14 foreground classes. Background is excluded from class reporting and can be treated as unlabeled or residual space, depending on the model interface.

### 3.2 Recommended segmentation protocol

### Protocol S1: Per-class prompted binary segmentation

For each image and each class, prompt the model to segment only that class.

Template:

`Segment all pixels belonging to [CLASS] in this image. Output only the mask for that class.`

This is the preferred VLM segmentation protocol because it is:

- the easiest to standardize across models with grounding or segmentation capability
- the easiest to map into the fixed ELDOR label space
- the least sensitive to open-ended generation instability

If definition-enhanced prompting is used, the class definition from Section 2.4 should be prepended.

This protocol is also the natural entry point for models such as `SAM 3` when used in a prompted per-class setting.

### 3.3 Converting VLM outputs into ELDOR masks

For segmentation, all outputs must be rasterized into a common `512x512` mask on the patch grid.

Rules:

- each class mask is converted to a binary mask in image coordinates
- overlapping masks should be resolved using a fixed rule
- recommended overlap rule:
  - assign each pixel to the class with the highest confidence
  - if no confidence is available, use a fixed class-priority order or larger-mask-first rule
- unlabeled pixels are treated as background or ignored area outside the 14 foreground classes

The overlap rule must be fixed before benchmarking and kept identical across all VLMs.

### 3.4 Segmentation metrics

Use the same segmentation metrics as the main benchmark:

- `mIoU`
- `mIoU_present`
- `macro F1`
- `macro F1_present`
- `foreground overall accuracy`
- per-class IoU
- per-class F1

---

## 4. Prompting and Decoding Controls

Prompting and decoding should stay as close as possible to each model's default stable usage.

The rule is simple:

- use the model's default inference settings whenever possible
- only constrain outputs when needed for evaluation stability and automatic parsing
- avoid aggressive manual tuning unless a model clearly requires it to function under the protocol

In practice:

- binary QA should use short constrained answers
- closed-set listing should use the fixed label list
- segmentation prompting should use the model's native prompting style
- decoding should remain deterministic when that is naturally supported

The goal is not prompt engineering for maximum score. The goal is fair and reproducible testing under a controlled label space.
