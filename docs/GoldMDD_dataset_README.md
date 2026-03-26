# ELDOR Data

- Public dataset name: `ELDOR`
- Local dataset root: `GoldMDD/data`
- Output folders: `train/image`, `train/label`, `val/image`, `val/label`, `test/image`, `test/label` (copied from `GoldMining/Data/Orthomosaic` and `GoldMining/Data/Label`).
- Source folders: `Drone_Orthomosaic_V1/Orthomosaic_org` and `Drone_Orthomosaic_V1/Just_labels`
- Matching rule: label to orthomosaic by filename mapping, then crop/reproject label to orthomosaic grid.
- Canonical labels: 14 semantic classes (`1..14`) plus `0=Background` after class merging in ELDOR.
- Split rule in this dataset: train = sites 1/5/7/10 (with `PlayaMirador1` = bottom half); val = sites 3/8 plus `PlayaMirador2` (top half); test = remaining sites.

## Spatial metadata table (source files + aligned output)

| Site | Ortho file | Label source file | CRS | Ortho size (W x H px) | Label size (W x H px) | Ortho res (m/px x,y) | Label res (m/px x,y) | Ortho lon range | Ortho lat range | Label lon range | Label lat range | Output size (W x H px) | Foreground area (ha) | Acquisition date | Split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| AcumulacionAaron2B | `AcumulacionAaron2B_corrected.tif` | `AcumulacionAaron2B_classified_corrected.tif` | EPSG:32719 | 38122x17643 | 38111x17760 | 0.056308, 0.056308 | 0.056323, 0.055936 | [-70.419597, -70.399788] | [-12.627643, -12.618558] | [-70.419597, -70.399788] | [-12.627643, -12.618558] | 38122x17643 | 212.0501 | 2022-04-24 | train |
| Anel | `Anel_corrected.tif` | `Anel_classified_corrected.tif` | EPSG:32719 | 26909x24725 | 17662x17661 | 0.056620, 0.056620 | 0.058580, 0.059049 | [-69.714044, -69.699976] | [-12.713767, -12.701071] | [-69.711529, -69.701974] | [-12.711964, -12.702509] | 18274x18420 | 102.7175 | 2022-04-08 | test |
| Clavelito | `Clavelito_corrected.tif` | `Clavelito_classified_corrected.tif` | EPSG:32719 | 19865x40307 | 19357x39684 | 0.075601, 0.075601 | 0.076376, 0.076781 | [-70.605945, -70.591929] | [-12.961391, -12.933761] | [-70.605944, -70.592143] | [-12.961387, -12.933761] | 19556x40304 | 344.5330 | 2022-05-09 | val |
| ElEngano | `ElEngano_corrected.tif` | `ElEngano_classified_corrected.tif` | EPSG:32719 | 23616x24742 | 17473x17473 | 0.057233, 0.057233 | 0.057566, 0.058201 | [-69.962478, -69.949966] | [-13.020161, -13.007312] | [-69.960769, -69.951459] | [-13.018407, -13.009178] | 17575x17770 | 101.2404 | 2022-03-03 | test |
| Kotsimba | `Kotsimba_corrected.tif` | `Kotsimba_classified_corrected.tif` | EPSG:32719 | 57331x53961 | 57227x53950 | 0.072760, 0.072760 | 0.072889, 0.072771 | [-70.283655, -70.244998] | [-13.136635, -13.100951] | [-70.283654, -70.244999] | [-13.136634, -13.100952] | 57328x53959 | 693.7270 | 2022-05-26 | train |
| Linda | `Linda2_corrected.tif` | `Linda_classified_corrected.tif` | EPSG:32719 | 23601x23698 | 17352x17352 | 0.057632, 0.057632 | 0.059290, 0.059408 | [-69.953378, -69.940789] | [-13.020082, -13.007688] | [-69.951874, -69.942353] | [-13.018530, -13.009175] | 17852x17888 | 98.8187 | 2022-02-28 | test |
| Los5Rebeldes | `Los5Rebeldes_corrected.tif` | `Los5Rebeldes_classified_corrected.tif` | EPSG:32719 | 43867x76279 | 33334x66668 | 0.030000, 0.030000 | 0.030140, 0.030001 | [-70.672354, -70.660084] | [-13.034336, -13.013573] | [-70.670932, -70.661549] | [-13.032969, -13.014831] | 33490x66672 | 199.9257 | 2022-02-04 | train |
| Nayda | `Nayda_corrected.tif` | `Nayda_classified_corrected.tif` | EPSG:32719 | 22677x23480 | 17665x17665 | 0.056610, 0.056610 | 0.057532, 0.057385 | [-69.721603, -69.709746] | [-12.713363, -12.701313] | [-69.720152, -69.710766] | [-12.711798, -12.702607] | 17954x17908 | 101.9933 | 2022-08-04 | val |
| Paolita | `Paolita1_corrected.tif` | `Paolita1_classified_corrected.tif` | EPSG:32719 | 58663x33523 | 54108x27055 | 0.055446, 0.055446 | 0.055275, 0.055441 | [-69.630457, -69.600462] | [-12.689889, -12.673012] | [-69.629627, -69.602050] | [-12.688983, -12.675355] | 53943x27054 | 445.5226 | 2022-02-21 | test |
| PlayaMirador1 | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.062030, -13.057446] | 17624x8883 | 50.1027 | 2022-03-10 | train |
| PlayaMirador2 | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.057446, -13.052861] | 17624x8883 | 50.1021 | 2022-03-10 | val |
| SantaInesDosMil | `SantaInesDosMil_corrected.tif` | `SantaInesDosMil_classified_corrected.tif` | EPSG:32719 | 22883x24295 | 17185x17184 | 0.058193, 0.058193 | 0.058873, 0.058471 | [-70.386657, -70.374307] | [-13.063436, -13.050589] | [-70.385274, -70.375895] | [-13.061789, -13.052656] | 17387x17267 | 100.9625 | 2022-03-10 | test |

- Note: `PlayaMirador` is additionally split in GoldMDD into `PlayaMirador1` (bottom half, train) and `PlayaMirador2` (top half, val). The two rows above use the same source files and split the label latitude range at the midpoint.

## Unified class mapping (ELDOR merged classes)

| Canonical ID | Class | Merged from original IDs | Color swatch | Color (HEX) | Area (ha) | Percentage (%) | Pixel count | Alias names seen in metadata |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Building | 1,4 | <span style="display:inline-block;width:14px;height:14px;background:#8A6A3D;border:1px solid #222;"></span> | `#8A6A3D` | 2.3745 | 0.09 | 6,367,488 | Area urbana; Campamento minero |
| 2 | Mining raft | 2 | <span style="display:inline-block;width:14px;height:14px;background:#7BEBFB;border:1px solid #222;"></span> | `#7BEBFB` | 0.1467 | 0.01 | 461,925 | Balsa; Balsa de mineria; Raft |
| 3 | Primary Forest | 3 | <span style="display:inline-block;width:14px;height:14px;background:#B04C18;border:1px solid #222;"></span> | `#B04C18` | 961.8239 | 38.45 | 3,308,805,552 | Bosque primario; Primary forest |
| 4 | Heavy machinery | 5,8,9,19 | <span style="display:inline-block;width:14px;height:14px;background:#EE92C6;border:1px solid #222;"></span> | `#EE92C6` | 0.0161 | 0.00 | 43,313 | Cargador frontal; Excavadora; Maquinaria pesada; Volquete; Dump truck |
| 5 | Water bodies | 6 | <span style="display:inline-block;width:14px;height:14px;background:#4F6F6F;border:1px solid #222;"></span> | `#4F6F6F` | 321.2655 | 12.84 | 1,085,789,506 | Cuerpos de agua |
| 6 | Agricultural crop | 7 | <span style="display:inline-block;width:14px;height:14px;background:#84D08C;border:1px solid #222;"></span> | `#84D08C` | 33.5185 | 1.34 | 105,688,237 | Cultivos agricolas; Cultivo agricola; Cultivos; Crops |
| 7 | Compact mounds | 10 | <span style="display:inline-block;width:14px;height:14px;background:#23F3E3;border:1px solid #222;"></span> | `#23F3E3` | 66.2310 | 2.65 | 215,787,431 | Monticulo compacto; Compact mound |
| 8 | Gravel mounds | 11 | <span style="display:inline-block;width:14px;height:14px;background:#585400;border:1px solid #222;"></span> | `#585400` | 13.2680 | 0.53 | 42,049,847 | Monticulos de cascajo |
| 9 | Grass | 12 | <span style="display:inline-block;width:14px;height:14px;background:#8DB51D;border:1px solid #222;"></span> | `#8DB51D` | 28.3472 | 1.13 | 87,888,718 | Pasto |
| 10 | Type 1 natural regeneration | 13 | <span style="display:inline-block;width:14px;height:14px;background:#C2163A;border:1px solid #222;"></span> | `#C2163A` | 228.5563 | 9.14 | 660,993,175 | Regeneracion natural tipo 1; Regeneracion tipo 1; Type 1 disturbed vegetation |
| 11 | Type 2 natural regeneration | 14 | <span style="display:inline-block;width:14px;height:14px;background:#F77757;border:1px solid #222;"></span> | `#F77757` | 523.1002 | 20.91 | 1,710,062,485 | Regeneracion natural tipo 2; Regeneracion tipo 2; Type 2 disturbed vegetation |
| 12 | Bare ground | 15 | <span style="display:inline-block;width:14px;height:14px;background:#2CD874;border:1px solid #222;"></span> | `#2CD874` | 322.9909 | 12.91 | 882,535,708 | Suelo desnudo; Bare soil |
| 13 | Sluice | 16 | <span style="display:inline-block;width:14px;height:14px;background:#613991;border:1px solid #222;"></span> | `#613991` | 0.0403 | 0.00 | 126,368 | Tolva; Tolvas; Hopper |
| 14 | Vehicles | 17,18 | <span style="display:inline-block;width:14px;height:14px;background:#969AAE;border:1px solid #222;"></span> | `#969AAE` | 0.0092 | 0.00 | 85,244 | Vehiculos; Vehiculos pequenos; Vehiculos pequeños; Small Vehicles |

## Per-site classes using ELDOR merged mapping

| Site | Split | Output ortho PNG | Output label PNG | Output size (W x H px) | Output total pixels | Merged class IDs present | Merged class names present |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| AcumulacionAaron2B | train | `AcumulacionAaron2B.png` | `AcumulacionAaron2B.png` | 38122x17643 | 672,586,446 | 1,2,3,4,5,6,8,10,11,12,13,14 | Building; Mining raft; Primary Forest; Heavy machinery; Water bodies; Agricultural crop; Gravel mounds; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground; Sluice; Vehicles |
| Kotsimba | train | `Kotsimba.png` | `Kotsimba.png` | 57328x53959 | 3,093,361,552 | 1,3,5,6,7,10,11,12 | Building; Primary Forest; Water bodies; Agricultural crop; Compact mounds; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground |
| Los5Rebeldes | train | `Los5Rebeldes.png` | `Los5Rebeldes.png` | 33490x66672 | 2,232,845,280 | 1,3,5,6,7,10,11,12,14 | Building; Primary Forest; Water bodies; Agricultural crop; Compact mounds; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground; Vehicles |
| PlayaMirador1 | train | `PlayaMirador1.png` | `PlayaMirador1.png` | 17624x8883 | 156,553,992 | 1,3,5,6,9,10,11,12 | Building; Primary Forest; Water bodies; Agricultural crop; Grass; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground |
| Clavelito | val | `Clavelito.png` | `Clavelito.png` | 19556x40304 | 788,185,024 | 1,3,4,5,6,7,10,11,12 | Building; Primary Forest; Heavy machinery; Water bodies; Agricultural crop; Compact mounds; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground |
| Nayda | val | `Nayda.png` | `Nayda.png` | 17954x17908 | 321,520,232 | 1,2,3,5,6,8,10,11,12,13 | Building; Mining raft; Primary Forest; Water bodies; Agricultural crop; Gravel mounds; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground; Sluice |
| PlayaMirador2 | val | `PlayaMirador2.png` | `PlayaMirador2.png` | 17624x8883 | 156,553,992 | 1,3,5,6,9,10,11,12 | Building; Primary Forest; Water bodies; Agricultural crop; Grass; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground |
| Anel | test | `Anel.png` | `Anel.png` | 18274x18420 | 336,607,080 | 1,2,3,5,6,8,10,11,12,13 | Building; Mining raft; Primary Forest; Water bodies; Agricultural crop; Gravel mounds; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground; Sluice |
| ElEngano | test | `ElEngano.png` | `ElEngano.png` | 17575x17770 | 312,307,750 | 3,5,6,10,11,12 | Primary Forest; Water bodies; Agricultural crop; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground |
| Linda | test | `Linda.png` | `Linda.png` | 17852x17888 | 319,336,576 | 1,2,3,5,10,11,12,13 | Building; Mining raft; Primary Forest; Water bodies; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground; Sluice |
| Paolita | test | `Paolita.png` | `Paolita.png` | 53943x27054 | 1,459,373,922 | 1,2,3,5,6,8,10,11,12,13 | Building; Mining raft; Primary Forest; Water bodies; Agricultural crop; Gravel mounds; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground; Sluice |
| SantaInesDosMil | test | `SantaInesDosMil.png` | `SantaInesDosMil.png` | 17387x17267 | 300,221,329 | 1,3,5,6,10,11,12 | Building; Primary Forest; Water bodies; Agricultural crop; Type 1 natural regeneration; Type 2 natural regeneration; Bare ground |

## Generation workflow

- Source orthomosaic/label alignment and original canonical labels come from `GoldMining/Data`.
- ELDOR labels are remapped from the original 19-class canonical IDs into a 14-class merged scheme (stored only under `GoldMDD/data/*/label`).
- Merge rules:
  - `Heavy machinery` = original IDs 5 (Front loader), 8 (Excavator), 9 (Heavy machinery), 19 (Dump truck)
  - `Vehicles` = original IDs 17 (Vehicles), 18 (Small vehicles)
  - `Building` = original IDs 1 (Urban area), 4 (Mining camp)
- Heatmap output: `GoldMDD/site_class_pixel_counts_heatmap_merged.png`
- Heatmap CSV: `GoldMDD/site_class_pixel_counts_merged.csv`
- Train/val/test distribution plot: `GoldMDD/train_val_test_class_distribution_merged.png`
- Train/val/test distribution CSV: `GoldMDD/train_val_test_class_distribution_merged.csv`

## Cropped patch dataset (`data-cropped`)

- Output root: `GoldMDD/data-cropped`
- Patch size: `512x512`
- Stride: `256`
- Filtering rule: drop a patch if background pixels in the merged label (`label==0`) are `>80%`.
- Windowing rule: full windows only (no padding).
- Patch naming: matching basenames per pair, e.g. image `AcumulacionAaron2B_2_3.jpg` and label `AcumulacionAaron2B_2_3.png` (1-based row/col indices).
- Storage format: image patches = JPEG, label patches = PNG (lossless class IDs).
- Folder structure: `train/image`, `train/label`, `val/image`, `val/label`, `test/image`, `test/label`.

### Crop summary by split

| Split | # Sites | Candidate windows | Kept patches | Dropped (>80% bg) | Kept ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 4 | 91,869 | 65,798 | 26,071 | 0.716 |
| val | 3 | 18,603 | 15,988 | 2,615 | 0.859 |
| test | 5 | 40,172 | 40,095 | 77 | 0.998 |
| **Total** | **12** | **150,644** | **121,881** | **28,763** | **0.809** |

### Crop summary by site

| Split | Site | Source size (W x H) | Candidate windows | Kept patches | Dropped (>80% bg) | Kept ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| train | AcumulacionAaron2B | 38122x17643 | 9,849 | 9,849 | 0 | 1.000 |
| train | Kotsimba | 57328x53959 | 46,398 | 20,327 | 26,071 | 0.438 |
| train | Los5Rebeldes | 33490x66672 | 33,411 | 33,411 | 0 | 1.000 |
| train | PlayaMirador1 | 17624x8883 | 2,211 | 2,211 | 0 | 1.000 |
| val | Clavelito | 19556x40304 | 11,700 | 9,085 | 2,615 | 0.776 |
| val | Nayda | 17954x17908 | 4,692 | 4,692 | 0 | 1.000 |
| val | PlayaMirador2 | 17624x8883 | 2,211 | 2,211 | 0 | 1.000 |
| test | Anel | 18274x18420 | 4,900 | 4,890 | 10 | 0.998 |
| test | ElEngano | 17575x17770 | 4,556 | 4,556 | 0 | 1.000 |
| test | Linda | 17852x17888 | 4,624 | 4,577 | 47 | 0.990 |
| test | Paolita | 53943x27054 | 21,736 | 21,716 | 20 | 0.999 |
| test | SantaInesDosMil | 17387x17267 | 4,356 | 4,356 | 0 | 1.000 |

- Summary CSVs:
  - `GoldMDD/data-cropped/crop_summary_by_split.csv`
  - `GoldMDD/data-cropped/crop_summary_by_site.csv`
