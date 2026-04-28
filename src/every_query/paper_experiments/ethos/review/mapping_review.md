# ETHOS mapping review

Per-EQ-code review of the EQ -> ETHOS code mapping produced by `build_mapping.py`. Each section shows the authoritative EQ-side label resolved through the staged Athena ontology, the verbatim LLM rationale (rendered once per EQ code where one was recorded), the candidate ETHOS tokens with vocab counts and constituent-code expansions, and a status block for the reviewer to mark approve / reject / modify.

## Reviewing principle

Default to the **tightest** mapping that still captures the EQ concept. When a candidate ETHOS token expands to constituent codes that include conditions clearly outside the EQ question (e.g. an ETHOS `ATHEROSCLEROSIS` bucket covering aortic, cerebrovascular, and peripheral vascular disease for an EQ question specifically about coronary artery atherosclerosis), lean toward `reject` or `modify`. The constituent-code lists below show every code (with its English description) the ETHOS token rolls up, so judge by reading those lines, not by the ETHOS token name in isolation.

## Table of contents

- [`DIAGNOSIS//ICD//10//I25118`](#diagnosisicd10i25118)
- [`DIAGNOSIS//ICD//9//3320`](#diagnosisicd93320)
- [`DIAGNOSIS//ICD//9//4271`](#diagnosisicd94271)
- [`DIAGNOSIS//ICD//9//5856`](#diagnosisicd95856)
- [`DIAGNOSIS//ICD//9//7295`](#diagnosisicd97295)
- [`INFUSION_END//229420//value_[10.687433,23.27545)`](#infusion-end229420value-106874332327545)
- [`INFUSION_START//221794//value_[8.004926,10.000001)`](#infusion-start221794value-800492610000001)
- [`LAB//220224//mmHg//value_[89.0,98.0)`](#lab220224mmhgvalue-890980)
- [`LAB//220339//cmH2O//value_[10.0,12.0)`](#lab220339cmh2ovalue-100120)
- [`LAB//224054//UNK//value_[2.0,3.0)`](#lab224054unkvalue-2030)
- [`LAB//224690//insp/min//value_[14.0,16.0)`](#lab224690inspminvalue-140160)
- [`LAB//227073//mEq/L//value_[17.0,19.0)`](#lab227073meqlvalue-170190)
- [`LAB//228724//cm//value_[1.5,2.0)`](#lab228724cmvalue-1520)
- [`LAB//51274//sec//value_[14.1,15.3)`](#lab51274secvalue-141153)
- [`MEDICATION//Carbidopa-Levodopa (25-100)//Administered`](#medicationcarbidopa-levodopa-25-100administered)
- [`MEDICATION//Gabapentin//Delayed Administered`](#medicationgabapentindelayed-administered)
- [`MEDICATION//START//Mupirocin Nasal Ointment 2%`](#medicationstartmupirocin-nasal-ointment-2)
- [`MEDICATION//STOP//Captopril`](#medicationstopcaptopril)
- [`MEDICATION//STOP//Doxycycline Hyclate`](#medicationstopdoxycycline-hyclate)
- [`MEDS_DEATH`](#meds-death)
- [`PROCEDURE//ICD//9//7936`](#procedureicd97936)
- [`TIMELINE//START`](#timelinestart)
- [`INFUSION_END//227536//value_[6.05,8.071587)` _(unmapped)_](#infusion-end227536value-6058071587)
- [`INFUSION_END//227536//value_[8.071587,9.379999)` _(unmapped)_](#infusion-end227536value-80715879379999)
- [`INFUSION_START//220949` _(unmapped)_](#infusion-start220949)
- [`INFUSION_START//225168//value_[284.02966,350.0)` _(unmapped)_](#infusion-start225168value-284029663500)
- [`LAB//220245//ml/min//value_[173.0,188.0)` _(unmapped)_](#lab220245mlminvalue-17301880)
- [`LAB//224665//UNK//value_[-inf,0.12)` _(unmapped)_](#lab224665unkvalue-inf012)
- [`LAB//225640//%//value_[0.3,0.6)` _(unmapped)_](#lab225640value-0306)
- [`LAB//225672//IU/L//value_[30.0,42.0)` _(unmapped)_](#lab225672iulvalue-300420)
- [`LAB//226499//mL//value_[3150.0,inf)` _(unmapped)_](#lab226499mlvalue-31500inf)
- [`LAB//227445//ng/mL//value_[11.0,20.0)` _(unmapped)_](#lab227445ngmlvalue-110200)
- [`LAB//229663//cmH2O//value_[-inf,5.0)` _(unmapped)_](#lab229663cmh2ovalue-inf50)
- [`LAB//229694//UNK//value_[0.0,inf)` _(unmapped)_](#lab229694unkvalue-00inf)
- [`LAB//50957//mmol/L//value_[0.5,0.6)` _(unmapped)_](#lab50957mmollvalue-0506)
- [`LAB//50964//mOsm/kg//value_[288.0,293.0)` _(unmapped)_](#lab50964mosmkgvalue-28802930)
- [`LAB//51066//mg/24hr//value_[177.0,216.0)` _(unmapped)_](#lab51066mg24hrvalue-17702160)
- [`LAB//51769//UNK//value_[1.88,2.29)` _(unmapped)_](#lab51769unkvalue-188229)
- [`SUBJECT_FLUID_OUTPUT//226600//mL//value_[20.0,30.0)` _(unmapped)_](#subject-fluid-output226600mlvalue-200300)
- [`SUBJECT_FLUID_OUTPUT//226600//mL//value_[5.0,10.0)` _(unmapped)_](#subject-fluid-output226600mlvalue-50100)
- [`SUBJECT_FLUID_OUTPUT//227510//mL//value_[25.0,50.0)` _(unmapped)_](#subject-fluid-output227510mlvalue-250500)

---

### <a id="diagnosisicd10i25118"></a>`DIAGNOSIS//ICD//10//I25118`

- **Family:** `DIAGNOSIS`
- **Mapped tiers:** icd_specific

**Authoritative EQ label:** ICD-10-CM `I25118` -- Atherosclerotic heart disease of native coronary artery with other forms of angina pectoris
- 3-char parent: `I25` -- Chronic ischemic heart disease
- SNOMED bridge: Angina co-occurrent and due to coronary arteriosclerosis (concept_id 36712983)

_Found under tier(s): icd_specific_

#### Candidate ETHOS token: `ICD//CM//CHRONIC_ISCHEMIC_HEART_DISEASE|ICD//CM//3-6//118` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `icd10cm:5char_to_label_plus_suffix`
- Inferred source: _no ICD-10-CM 3-char category match for label `CHRONIC_ISCHEMIC_HEART_DISEASE|ICD//CM//3-6//118`._

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="diagnosisicd93320"></a>`DIAGNOSIS//ICD//9//3320`

- **Family:** `DIAGNOSIS`
- **Mapped tiers:** icd_crosswalk

**Authoritative EQ label:** ICD-9-CM `3320` -- Paralysis agitans
- SNOMED bridge: Parkinson's disease (concept_id 381270)
- ICD-10-CM crosswalk (7 codes):
  - `G20` -- Parkinson's disease
  - `G20.A` -- Parkinson's disease without dyskinesia
  - `G20.A1` -- Parkinson's disease without dyskinesia, without mention of fluctuations
  - `G20.A2` -- Parkinson's disease without dyskinesia, with fluctuations
  - `G20.B` -- Parkinson's disease with dyskinesia
  - `G20.B1` -- Parkinson's disease with dyskinesia, without mention of fluctuations
  - `G20.B2` -- Parkinson's disease with dyskinesia, with fluctuations

**LLM rationale (verbatim):**

> ICD-10-CM 3-char parent G20 resolved to ETHOS token via ethos_icd_3char_index.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PARKINSON'S_DISEASE` _(primary)_

- ETHOS vocab count: 5,606
- Match kind: `literal`
- Mapping source: `deterministic:icd_3char_walker`
- Inferred source: ICD-10-CM 3-char category `G20` -- Parkinson's disease
- Constituent ICD-10-CM codes (7):
  - `G20.A` -- Parkinson's disease without dyskinesia
  - `G20.A1` -- Parkinson's disease without dyskinesia, without mention of fluctuations
  - `G20.A2` -- Parkinson's disease without dyskinesia, with fluctuations
  - `G20.B` -- Parkinson's disease with dyskinesia
  - `G20.B1` -- Parkinson's disease with dyskinesia, without mention of fluctuations
  - `G20.B2` -- Parkinson's disease with dyskinesia, with fluctuations
  - `G20.C` -- Parkinsonism, unspecified

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="diagnosisicd94271"></a>`DIAGNOSIS//ICD//9//4271`

- **Family:** `DIAGNOSIS`
- **Mapped tiers:** icd_crosswalk

**Authoritative EQ label:** ICD-9-CM `4271` -- Paroxysmal ventricular tachycardia
- SNOMED bridge: Paroxysmal ventricular tachycardia (concept_id 437579)

**LLM rationale (verbatim):**

> While the EQ code specifically refers to ventricular tachycardia and the ETHOS token represents the broader category of paroxysmal tachycardia (which includes both ventricular and supraventricular types), this is the only available candidate and paroxysmal ventricular tachycardia is a subset of paroxysmal tachycardia. The mapping loses some specificity but remains clinically valid.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PAROXYSMAL_TACHYCARDIA` _(primary)_

- ETHOS vocab count: 9,342
- Match kind: `literal`
- Mapping source: `llm:diagnosis_walker_unresolved`
- Inferred source: ICD-10-CM 3-char category `I47` -- Paroxysmal tachycardia
- Constituent ICD-10-CM codes (10):
  - `I47.0` -- Re-entry ventricular arrhythmia
  - `I47.1` -- Supraventricular tachycardia
  - `I47.10` -- Supraventricular tachycardia, unspecified
  - `I47.11` -- Inappropriate sinus tachycardia, so stated
  - `I47.19` -- Other supraventricular tachycardia
  - `I47.2` -- Ventricular tachycardia
  - `I47.20` -- Ventricular tachycardia, unspecified
  - `I47.21` -- Torsades de pointes
  - `I47.29` -- Other ventricular tachycardia
  - `I47.9` -- Paroxysmal tachycardia, unspecified

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="diagnosisicd95856"></a>`DIAGNOSIS//ICD//9//5856`

- **Family:** `DIAGNOSIS`
- **Mapped tiers:** icd_specific

**Authoritative EQ label:** ICD-9-CM `5856` -- End stage renal disease
- SNOMED bridge: End-stage renal disease (concept_id 193782)
- ICD-10-CM crosswalk (1 code):
  - `N18.6` -- End stage renal disease

_Found under tier(s): icd_specific_

#### Candidate ETHOS token: `ICD//CM//CHRONIC_KIDNEY_DISEASE_(CKD)|ICD//CM//3-6//6` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `icd9_dx:longest_icd10_target`
- Inferred source: _no ICD-10-CM 3-char category match for label `CHRONIC_KIDNEY_DISEASE_(CKD)|ICD//CM//3-6//6`._

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="diagnosisicd97295"></a>`DIAGNOSIS//ICD//9//7295`

- **Family:** `DIAGNOSIS`
- **Mapped tiers:** icd_specific

**Authoritative EQ label:** ICD-9-CM `7295` -- Pain in limb
- SNOMED bridge: Pain in limb (concept_id 138525)
- ICD-10-CM crosswalk (3 codes):
  - `M79.6` -- Pain in limb, hand, foot, fingers and toes
  - `M79.60` -- Pain in limb, unspecified
  - `M79.609` -- Pain in unspecified limb

_Found under tier(s): icd_specific_

#### Candidate ETHOS token: `ICD//CM//OTHER_AND_UNSPECIFIED_SOFT_TISSUE_DISORDERS_NOT_ELSEWHERE_CLASSIFIED|ICD//CM//3-6//609` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `icd9_dx:longest_icd10_target`
- Inferred source: _no ICD-10-CM 3-char category match for label `OTHER_AND_UNSPECIFIED_SOFT_TISSUE_DISORDERS_NOT_ELSEWHERE_CLASSIFIED|ICD//CM//3-6//609`._

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="infusion-end229420value-106874332327545"></a>`INFUSION_END//229420//value_[10.687433,23.27545)`

- **Family:** `INFUSION_END`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `229420` -- Dexmedetomidine (Precedex) (source: `d_items`)
- abbreviation: Dexmedetomidine (Precedex)
- category: Medications
- unitname: mcg
- linksto: inputevents
- EQ-encoded units: `value_[10.687433,23.27545)`

**LLM rationale (verbatim):**

> Dexmedetomidine is a sedative agent used for procedural sedation and ICU sedation. It falls under ATC class N05 (psycholeptics/sedatives), which is the most semantically appropriate match in the ETHOS vocabulary for this alpha-2 agonist sedative medication.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//N05//PSYCHOLEPTICS` _(secondary)_

- ETHOS vocab count: 908,963
- Match kind: `literal`
- Mapping source: `llm:infusion_walker_unresolved`
- Inferred source: ATC level 2 class `N05` -- PSYCHOLEPTICS
- Constituent RxNorm ingredients (143):
  - Valeriana officinalis whole extract
  - acepromazine
  - acetophenazine
  - allobarbital
  - alprazolam
  - amisulpride
  - amobarbital
  - aprobarbital
  - aripiprazole
  - asenapine
  - barbital
  - benperidol
  - brexpiprazole
  - bromazepam
  - bromide ion
  - bromperidol
  - brotizolam
  - buspirone
  - butobarbital
  - butylvinal
  - captodiamine
  - carbromal
  - cariprazine
  - chloral betaine
  - chlordiazepoxide
  - chlormethiazole
  - chlorproethazine
  - chlorpromazine
  - chlorprothixene
  - clobazam
  - ... +113 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="infusion-start221794value-800492610000001"></a>`INFUSION_START//221794//value_[8.004926,10.000001)`

- **Family:** `INFUSION_START`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `221794` -- Furosemide (Lasix) (source: `d_items`)
- abbreviation: Furosemide (Lasix)
- category: Medications
- unitname: mg
- linksto: inputevents
- EQ-encoded units: `value_[8.004926,10.000001)`

**LLM rationale (verbatim):**

> RxNorm ingredient 'furosemide' walks to ATC C03 via OHDSI CONCEPT_ANCESTOR.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//C03//DIURETICS` _(secondary)_

- ETHOS vocab count: 612,787
- Match kind: `literal`
- Mapping source: `deterministic:atc_ancestor_walker`
- Inferred source: ATC level 2 class `C03` -- DIURETICS
- Constituent RxNorm ingredients (35):
  - althiazide
  - amiloride
  - bendroflumethiazide
  - bumetanide
  - buthiazide
  - canrenoate
  - canrenone
  - chlorothiazide
  - chlorthalidone
  - cicletanine
  - clopamide
  - conivaptan
  - cyclopenthiazide
  - cyclothiazide
  - eplerenone
  - ethacrynate
  - finerenone
  - furosemide
  - hydrochlorothiazide
  - hydroflumethiazide
  - indapamide
  - mefruside
  - mersalyl
  - methyclothiazide
  - metolazone
  - piretanide
  - polythiazide
  - quinethazone
  - spironolactone
  - theobromine
  - ... +5 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab220224mmhgvalue-890980"></a>`LAB//220224//mmHg//value_[89.0,98.0)`

- **Family:** `LAB`
- **Mapped tiers:** quantile

**Authoritative EQ label:** MIMIC item-id `220224` -- Arterial O2 pressure (source: `d_items`)
- abbreviation: PO2 (Arterial)
- category: Labs
- unitname: mmHg
- linksto: chartevents
- EQ-encoded units: `mmHg`

_Found under tier(s): quantile_

#### Candidate ETHOS token: `LAB//220224//MMHG|Q4` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `meds-codes.parquet:values_quantiles_or_sibling_bins`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab220339cmh2ovalue-100120"></a>`LAB//220339//cmH2O//value_[10.0,12.0)`

- **Family:** `LAB`
- **Mapped tiers:** quantile_approx

**Authoritative EQ label:** MIMIC item-id `220339` -- PEEP set (source: `d_items`)
- abbreviation: PEEP set
- category: Respiratory
- unitname: cmH2O
- linksto: chartevents
- EQ-encoded units: `cmH2O`

_Found under tier(s): quantile_approx_

#### Candidate ETHOS token: `LAB//220339//CMH2O|Q8` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `meds-codes.parquet:sibling_bins_positional`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab224054unkvalue-2030"></a>`LAB//224054//UNK//value_[2.0,3.0)`

- **Family:** `LAB`
- **Mapped tiers:** quantile_approx

**Authoritative EQ label:** MIMIC item-id `224054` -- Braden Sensory Perception (source: `d_items`)
- abbreviation: Braden Sensory Perception
- category: Skin - Assessment
- linksto: chartevents
- EQ-encoded units: `UNK`

_Found under tier(s): quantile_approx_

#### Candidate ETHOS token: `LAB//224054//UNK|Q5` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `meds-codes.parquet:sibling_bins_positional`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab224690inspminvalue-140160"></a>`LAB//224690//insp/min//value_[14.0,16.0)`

- **Family:** `LAB`
- **Mapped tiers:** quantile

**Authoritative EQ label:** MIMIC item-id `224690` -- Respiratory Rate (Total) (source: `d_items`)
- abbreviation: Respiratory Rate (Total)
- category: Respiratory
- unitname: insp/min
- linksto: chartevents
- EQ-encoded units: `insp/min`

_Found under tier(s): quantile_

#### Candidate ETHOS token: `LAB//224690//INSP/MIN|Q2` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `meds-codes.parquet:values_quantiles_or_sibling_bins`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab227073meqlvalue-170190"></a>`LAB//227073//mEq/L//value_[17.0,19.0)`

- **Family:** `LAB`
- **Mapped tiers:** quantile_approx

**Authoritative EQ label:** MIMIC item-id `227073` -- Anion gap (source: `d_items`)
- abbreviation: Anion gap
- category: Labs
- unitname: None
- linksto: chartevents
- EQ-encoded units: `mEq/L`

_Found under tier(s): quantile_approx_

#### Candidate ETHOS token: `LAB//227073//MEQ/L|Q9` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `meds-codes.parquet:sibling_bins_positional`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab228724cmvalue-1520"></a>`LAB//228724//cm//value_[1.5,2.0)`

- **Family:** `LAB`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `228724` -- Pressure ulcer #2- Length (source: `d_items`)
- abbreviation: Pressure ulcer #2- Length
- category: Skin - Impairment
- unitname: cm
- linksto: chartevents
- EQ-encoded units: `cm`

**LLM rationale (verbatim):**

> The EQ code explicitly measures the length of pressure ulcer #2, making it a direct measurement related to pressure ulcers. The ICD-10-CM code L89 for Pressure Ulcer is the most semantically specific and faithful match among the candidates.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PRESSURE_ULCER` _(secondary)_

- ETHOS vocab count: 11,617
- Match kind: `literal`
- Mapping source: `llm:orphan_lab`
- Inferred source: ICD-10-CM 3-char category `L89` -- Pressure ulcer
- Constituent ICD-10-CM codes (207):
  - `L89.0` -- Pressure ulcer of elbow
  - `L89.00` -- Pressure ulcer of unspecified elbow
  - `L89.000` -- Pressure ulcer of unspecified elbow, unstageable
  - `L89.001` -- Pressure ulcer of unspecified elbow, stage 1
  - `L89.002` -- Pressure ulcer of unspecified elbow, stage 2
  - `L89.003` -- Pressure ulcer of unspecified elbow, stage 3
  - `L89.004` -- Pressure ulcer of unspecified elbow, stage 4
  - `L89.006` -- Pressure-induced deep tissue damage of unspecified elbow
  - `L89.009` -- Pressure ulcer of unspecified elbow, unspecified stage
  - `L89.01` -- Pressure ulcer of right elbow
  - `L89.010` -- Pressure ulcer of right elbow, unstageable
  - `L89.011` -- Pressure ulcer of right elbow, stage 1
  - `L89.012` -- Pressure ulcer of right elbow, stage 2
  - `L89.013` -- Pressure ulcer of right elbow, stage 3
  - `L89.014` -- Pressure ulcer of right elbow, stage 4
  - `L89.016` -- Pressure-induced deep tissue damage of right elbow
  - `L89.019` -- Pressure ulcer of right elbow, unspecified stage
  - `L89.02` -- Pressure ulcer of left elbow
  - `L89.020` -- Pressure ulcer of left elbow, unstageable
  - `L89.021` -- Pressure ulcer of left elbow, stage 1
  - `L89.022` -- Pressure ulcer of left elbow, stage 2
  - `L89.023` -- Pressure ulcer of left elbow, stage 3
  - `L89.024` -- Pressure ulcer of left elbow, stage 4
  - `L89.026` -- Pressure-induced deep tissue damage of left elbow
  - `L89.029` -- Pressure ulcer of left elbow, unspecified stage
  - `L89.1` -- Pressure ulcer of back
  - `L89.10` -- Pressure ulcer of unspecified part of back
  - `L89.100` -- Pressure ulcer of unspecified part of back, unstageable
  - `L89.101` -- Pressure ulcer of unspecified part of back, stage 1
  - `L89.102` -- Pressure ulcer of unspecified part of back, stage 2
  - ... +177 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab51274secvalue-141153"></a>`LAB//51274//sec//value_[14.1,15.3)`

- **Family:** `LAB`
- **Mapped tiers:** quantile

**Authoritative EQ label:** MIMIC item-id `51274` -- PT (source: `d_labitems`)
- category: Hematology
- fluid: Blood
- EQ-encoded units: `sec`

_Found under tier(s): quantile_

#### Candidate ETHOS token: `LAB//51274//SEC|Q6` _(primary)_

- ETHOS vocab count: 0
- Match kind: `code+next_token`
- Mapping source: `meds-codes.parquet:values_quantiles_or_sibling_bins`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="medicationcarbidopa-levodopa-25-100administered"></a>`MEDICATION//Carbidopa-Levodopa (25-100)//Administered`

- **Family:** `MEDICATION`
- **Mapped tiers:** atc_crosswalk

**Authoritative EQ label:** MIMIC medication `Carbidopa-Levodopa (25-100)` (admin modes: Administered)
- RxNorm match: levodopa (concept_class Ingredient)
- Ingredient: levodopa (concept_id 789578)
- ATC level 3: `N04B` -- DOPAMINERGIC AGENTS
- ATC level 4: `N04BA` -- Dopa and dopa derivatives

**LLM rationale (verbatim):**

> RxNorm ingredient 'levodopa' walks to ATC N04 via OHDSI CONCEPT_ANCESTOR.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//N04//ANTI-PARKINSON_DRUGS` _(secondary)_

- ETHOS vocab count: 162,103
- Match kind: `literal`
- Mapping source: `deterministic:atc_ancestor_walker`
- Inferred source: ATC level 2 class `N04` -- ANTI-PARKINSON DRUGS
- Constituent RxNorm ingredients (32):
  - amantadine
  - apomorphine
  - benztropine
  - biperiden
  - bornaprine
  - bromocriptine
  - budipine
  - cabergoline
  - dexetimide
  - dihydroergocryptine
  - diphenhydramine
  - entacapone
  - ethybenztropine
  - etilevodopa
  - istradefylline
  - levodopa
  - methixene
  - opicapone
  - orphenadrine
  - pergolide
  - piribedil
  - pramipexole
  - procyclidine
  - profenamine
  - rasagiline
  - ropinirole
  - rotigotine
  - safinamide
  - selegiline
  - tolcapone
  - ... +2 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="medicationgabapentindelayed-administered"></a>`MEDICATION//Gabapentin//Delayed Administered`

- **Family:** `MEDICATION`
- **Mapped tiers:** atc_crosswalk

**Authoritative EQ label:** MIMIC medication `Gabapentin` (admin modes: Delayed Administered)
- RxNorm match: gabapentin (concept_class Ingredient)
- Ingredient: gabapentin (concept_id 797399)
- ATC level 3: `N02B` -- OTHER ANALGESICS AND ANTIPYRETICS
- ATC level 4: `N03AX` -- Other antiepileptics

**LLM rationale (verbatim):**

> RxNorm ingredient 'gabapentin' (concept_id=797399) sits in two distinct ATC chains; both are committed since they reflect the same source concept.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//N02//ANALGESICS` _(secondary)_

- ETHOS vocab count: 5,128,410
- Match kind: `literal`
- Mapping source: `deterministic:atc_same_source_concept_tie`
- Inferred source: ATC level 2 class `N02` -- ANALGESICS
- Constituent RxNorm ingredients (81):
  - acetaminophen
  - almotriptan
  - aloxiprin
  - aminopyrine
  - antipyrine
  - aspirin
  - atogepant
  - benorilate
  - buprenorphine
  - butorphanol
  - cannabidiol
  - cannabigerol
  - cannabigerolate
  - cannabinol
  - choline
  - clonidine
  - codeine
  - dextromoramide
  - dezocine
  - diflunisal
  - dihydrocodeine
  - dihydroergotamine
  - dipyrone
  - dronabinol
  - eletriptan
  - eptinezumab
  - erenumab
  - ergotamine
  - ethenzamide
  - fentanyl
  - ... +51 more

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//N03//ANTIEPILEPTICS` _(secondary)_

- ETHOS vocab count: 1,322,698
- Match kind: `literal`
- Mapping source: `deterministic:atc_same_source_concept_tie`
- Inferred source: ATC level 2 class `N03` -- ANTIEPILEPTICS
- Constituent RxNorm ingredients (45):
  - aminobutyrate
  - barbexaclone
  - beclamide
  - brivaracetam
  - cannabidiol
  - carbamazepine
  - cenobamate
  - clonazepam
  - dipropylacetamide
  - eslicarbazepine
  - ethosuximide
  - ethotoin
  - ezogabine
  - felbamate
  - fenfluramine
  - fosphenytoin
  - gabapentin
  - ganaxolone
  - lacosamide
  - lamotrigine
  - levetiracetam
  - mephenytoin
  - mephobarbital
  - metharbital
  - methsuximide
  - oxcarbazepine
  - paramethadione
  - perampanel
  - phenacemide
  - phenobarbital
  - ... +15 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="medicationstartmupirocin-nasal-ointment-2"></a>`MEDICATION//START//Mupirocin Nasal Ointment 2%`

- **Family:** `MEDICATION`
- **Mapped tiers:** atc_crosswalk

**Authoritative EQ label:** MIMIC medication `Mupirocin Nasal Ointment 2%` (admin modes: START)
- RxNorm match: mupirocin (concept_class Ingredient)
- Ingredient: mupirocin (concept_id 951511)
- ATC level 3: `B05C` -- IRRIGATING SOLUTIONS
- ATC level 4: `B05CA` -- Antiinfectives

**LLM rationale (verbatim):**

> Mupirocin Nasal Ointment 2% is specifically formulated for nasal application, making R01 (Nasal Preparations) the most semantically specific match. While mupirocin is an antibiotic that could fall under D06 for dermatological use, the nasal formulation and route of administration makes R01 the most faithful representation of this specific product.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//R01//NASAL_PREPARATIONS` _(secondary)_

- ETHOS vocab count: 125,895
- Match kind: `literal`
- Mapping source: `llm:atc_multi_chain`
- Inferred source: ATC level 2 class `R01` -- NASAL PREPARATIONS
- Constituent RxNorm ingredients (43):
  - all-trans-retinol
  - antazoline
  - azelastine
  - beclomethasone
  - betamethasone
  - budesonide
  - calcium
  - ciclesonide
  - cromoglycate
  - cromolyn
  - cyclopentamine
  - dexamethasone
  - ephedrine
  - epinephrine
  - fenoxazoline
  - flunisolide
  - fluticasone
  - framycetin
  - hexamidine
  - hyaluronate
  - hydrocortisone
  - indanazoline
  - ipratropium
  - isospaglumic acid
  - levocabastine
  - mometasone
  - mupirocin
  - naphazoline
  - nedocromil
  - olopatadine
  - ... +13 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="medicationstopcaptopril"></a>`MEDICATION//STOP//Captopril`

- **Family:** `MEDICATION`
- **Mapped tiers:** atc_crosswalk

**Authoritative EQ label:** MIMIC medication `Captopril` (admin modes: STOP)
- RxNorm match: captopril (concept_class Ingredient)
- Ingredient: captopril (concept_id 1340128)
- ATC level 3: `C09A` -- ACE INHIBITORS, PLAIN
- ATC level 4: `C03AX` -- Thiazides, combinations with other drugs

**LLM rationale (verbatim):**

> RxNorm ingredient 'captopril' walks to ATC C09 via OHDSI CONCEPT_ANCESTOR.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//C09//AGENTS_ACTING_ON_THE_RENIN-ANGIOTENSIN_SYSTEM` _(secondary)_

- ETHOS vocab count: 285,699
- Match kind: `literal`
- Mapping source: `deterministic:atc_ancestor_walker`
- Inferred source: ATC level 2 class `C09` -- AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM
- Constituent RxNorm ingredients (25):
  - aliskiren
  - azilsartan
  - benazepril
  - candesartan
  - captopril
  - cilazapril
  - enalapril
  - enalaprilat
  - eprosartan
  - fosinopril
  - imidapril
  - irbesartan
  - lisinopril
  - losartan
  - moexipril
  - olmesartan
  - perindopril
  - quinapril
  - ramipril
  - sparsentan
  - spirapril
  - telmisartan
  - trandolapril
  - valsartan
  - zofenopril

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="medicationstopdoxycycline-hyclate"></a>`MEDICATION//STOP//Doxycycline Hyclate`

- **Family:** `MEDICATION`
- **Mapped tiers:** atc_crosswalk

**Authoritative EQ label:** MIMIC medication `Doxycycline Hyclate` (admin modes: STOP)
- RxNorm match: doxycycline hyclate (concept_class Precise Ingredient)

**LLM rationale (verbatim):**

> LLM proposed 'ATC//J01//ANTIBACTERIALS_FOR_SYSTEMIC_USE'; salvaged to canonical ETHOS token 'ATC//J01//ANTIBIOTICS_AND_ANTIBACTERIALS_FOR_SYSTEMIC_USE' via unique namespace-prefix match. Original rationale: Doxycycline hyclate is a tetracycline antibiotic used systemically for bacterial infections. The ATC level-3 class J01 (Antibacterials for Systemic Use) is the most semantically appropriate match, as ETHOS tokens do not appear to include specific drug ingredients or brand formulations, only therapeutic drug classes.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//J01//ANTIBIOTICS_AND_ANTIBACTERIALS_FOR_SYSTEMIC_USE` _(secondary)_

- ETHOS vocab count: 2,103,569
- Match kind: `literal`
- Mapping source: `llm:medication_walker_unresolved`
- Inferred source: ATC level 2 class `J01` -- ANTIBACTERIALS FOR SYSTEMIC USE
- Constituent RxNorm ingredients (187):
  - amdinocillin
  - amdinocillin pivoxil
  - amikacin
  - amoxicillin
  - ampicillin
  - arbekacin
  - avibactam
  - azidocillin
  - azithromycin
  - azlocillin
  - aztreonam
  - bacampicillin
  - bacitracin
  - bacitracin methylene disalicylate
  - brodimoprim
  - carbenicillin
  - cefaclor
  - cefadroxil
  - cefamandole
  - cefatrizine
  - cefazolin
  - cefdinir
  - cefditoren
  - cefepime
  - cefetamet
  - cefiderocol
  - cefixime
  - cefmetazole
  - cefodizime
  - cefonicid
  - ... +157 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="meds-death"></a>`MEDS_DEATH`

- **Family:** `MEDS_DEATH`
- **Mapped tiers:** exact

**Authoritative EQ label:** Patient death (MEDS-format mortality token)

_Found under tier(s): exact_

#### Candidate ETHOS token: `MEDS_DEATH` _(primary)_

- ETHOS vocab count: 34,014
- Match kind: `literal`
- Mapping source: `code:string_equality`
- Inferred source: ETHOS mortality token (passthrough, no ontology lookup).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="procedureicd97936"></a>`PROCEDURE//ICD//9//7936`

- **Family:** `PROCEDURE`
- **Mapped tiers:** icd_crosswalk

**Authoritative EQ label:** ICD-9-Proc `7936` -- Open reduction of fracture with internal fixation, tibia and fibula

**LLM rationale (verbatim):**

> The EQ code describes a procedure for open reduction with internal fixation of tibia and fibula fractures. The tibia and fibula are the two bones of the lower leg, which directly corresponds to the ETHOS token for fractures of the lower leg including ankle (S82).

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//FRACTURE_OF_LOWER_LEG_INCLUDING_ANKLE` _(secondary)_

- ETHOS vocab count: 4,413
- Match kind: `literal`
- Mapping source: `llm:no_pcs_tokens_in_ethos`
- Inferred source: ICD-10-CM 3-char category `S82` -- Fracture of lower leg, including ankle
- Constituent ICD-10-CM codes (3345):
  - `S82.0` -- Fracture of patella
  - `S82.00` -- Unspecified fracture of patella
  - `S82.001` -- Unspecified fracture of right patella
  - `S82.001A` -- Unspecified fracture of right patella, initial encounter for closed fracture
  - `S82.001B` -- Unspecified fracture of right patella, initial encounter for open fracture type I or II
  - `S82.001C` -- Unspecified fracture of right patella, initial encounter for open fracture type IIIA, IIIB, or IIIC
  - `S82.001D` -- Unspecified fracture of right patella, subsequent encounter for closed fracture with routine healing
  - `S82.001E` -- Unspecified fracture of right patella, subsequent encounter for open fracture type I or II with routine healing
  - `S82.001F` -- Unspecified fracture of right patella, subsequent encounter for open fracture type IIIA, IIIB, or IIIC with routine healing
  - `S82.001G` -- Unspecified fracture of right patella, subsequent encounter for closed fracture with delayed healing
  - `S82.001H` -- Unspecified fracture of right patella, subsequent encounter for open fracture type I or II with delayed healing
  - `S82.001J` -- Unspecified fracture of right patella, subsequent encounter for open fracture type IIIA, IIIB, or IIIC with delayed healing
  - `S82.001K` -- Unspecified fracture of right patella, subsequent encounter for closed fracture with nonunion
  - `S82.001M` -- Unspecified fracture of right patella, subsequent encounter for open fracture type I or II with nonunion
  - `S82.001N` -- Unspecified fracture of right patella, subsequent encounter for open fracture type IIIA, IIIB, or IIIC with nonunion
  - `S82.001P` -- Unspecified fracture of right patella, subsequent encounter for closed fracture with malunion
  - `S82.001Q` -- Unspecified fracture of right patella, subsequent encounter for open fracture type I or II with malunion
  - `S82.001R` -- Unspecified fracture of right patella, subsequent encounter for open fracture type IIIA, IIIB, or IIIC with malunion
  - `S82.001S` -- Unspecified fracture of right patella, sequela
  - `S82.002` -- Unspecified fracture of left patella
  - `S82.002A` -- Unspecified fracture of left patella, initial encounter for closed fracture
  - `S82.002B` -- Unspecified fracture of left patella, initial encounter for open fracture type I or II
  - `S82.002C` -- Unspecified fracture of left patella, initial encounter for open fracture type IIIA, IIIB, or IIIC
  - `S82.002D` -- Unspecified fracture of left patella, subsequent encounter for closed fracture with routine healing
  - `S82.002E` -- Unspecified fracture of left patella, subsequent encounter for open fracture type I or II with routine healing
  - `S82.002F` -- Unspecified fracture of left patella, subsequent encounter for open fracture type IIIA, IIIB, or IIIC with routine healing
  - `S82.002G` -- Unspecified fracture of left patella, subsequent encounter for closed fracture with delayed healing
  - `S82.002H` -- Unspecified fracture of left patella, subsequent encounter for open fracture type I or II with delayed healing
  - `S82.002J` -- Unspecified fracture of left patella, subsequent encounter for open fracture type IIIA, IIIB, or IIIC with delayed healing
  - `S82.002K` -- Unspecified fracture of left patella, subsequent encounter for closed fracture with nonunion
  - ... +3315 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="timelinestart"></a>`TIMELINE//START`

- **Family:** `TIMELINE`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** Start of patient record (first event marker)

**LLM rationale (verbatim):**

> EQ's TIMELINE//START fires at the first event of a patient's record; ETHOS uses HOSPITAL_ADMISSION as its admission marker.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `HOSPITAL_ADMISSION` _(secondary)_

- ETHOS vocab count: 297,949
- Match kind: `literal`
- Mapping source: `direct:event_alignment`
- Inferred source: ETHOS hospital-admission marker (synthetic event token).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="infusion-end227536value-6058071587"></a>`INFUSION_END//227536//value_[6.05,8.071587)` _(unmapped)_

- **Family:** `INFUSION_END`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `227536` -- KCl (CRRT) (source: `d_items`)
- abbreviation: KCl (CRRT)
- category: Medications
- unitname: mEq.
- linksto: inputevents
- EQ-encoded units: `value_[6.05,8.071587)`

**Unmapped reason (from mapping_coverage):** `ethos_token_missing/no_infusion_prefix_in_vocab`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: KCl (CRRT) (MIMIC item 227536)
- reason: LLM proposed 'LAB//227536//mEq/L' which is not in the ETHOS vocab; original rationale: This represents a potassium chloride (KCl) measurement during continuous renal replacement therapy (CRRT), which is a laboratory value. The MIMIC item 227536 corresponds to a serum potassium lab result, typically measured in mEq/L, making it appropriately mapped to the LAB namespace with the same identifier.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="infusion-end227536value-80715879379999"></a>`INFUSION_END//227536//value_[8.071587,9.379999)` _(unmapped)_

- **Family:** `INFUSION_END`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `227536` -- KCl (CRRT) (source: `d_items`)
- abbreviation: KCl (CRRT)
- category: Medications
- unitname: mEq.
- linksto: inputevents
- EQ-encoded units: `value_[8.071587,9.379999)`

**Unmapped reason (from mapping_coverage):** `ethos_token_missing/no_infusion_prefix_in_vocab`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: KCl (CRRT) (MIMIC item 227536)
- reason: LLM proposed 'LAB//227536//mEq/L' which is not in the ETHOS vocab; original rationale: This represents a potassium chloride (KCl) infusion measurement during continuous renal replacement therapy (CRRT), which is a laboratory value tracking serum potassium levels. The value range [8.071587,9.379999) appears to represent potassium concentration in mEq/L, making this appropriately mapped to a LAB token with the MIMIC item ID and standard potassium units.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="infusion-start220949"></a>`INFUSION_START//220949` _(unmapped)_

- **Family:** `INFUSION_START`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `220949` -- Dextrose 5% (source: `d_items`)
- abbreviation: Dextrose 5%
- category: Fluids/Intake
- unitname: mL
- linksto: inputevents

**Unmapped reason (from mapping_coverage):** `ethos_token_missing/no_infusion_prefix_in_vocab`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Dextrose 5% (MIMIC item 220949)
- reason: Dextrose 5% is an intravenous fluid used for hydration and as a vehicle for medication delivery, not a therapeutic drug class. The ETHOS vocabulary includes ATC drug classes for pharmacologically active medications, but D5W is a basic crystalloid solution without a specific therapeutic indication that would map to ICD diagnosis categories, ATC drug classes, or procedure codes.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="infusion-start225168value-284029663500"></a>`INFUSION_START//225168//value_[284.02966,350.0)` _(unmapped)_

- **Family:** `INFUSION_START`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `225168` -- Packed Red Blood Cells (source: `d_items`)
- abbreviation: PRBC's
- category: Blood Products/Colloids
- unitname: mL
- linksto: inputevents
- EQ-encoded units: `value_[284.02966,350.0)`

**Unmapped reason (from mapping_coverage):** `ethos_token_missing/no_infusion_prefix_in_vocab`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Packed Red Blood Cells (MIMIC item 225168)
- reason: Packed Red Blood Cells are actual blood products (whole blood components), not blood substitutes or perfusion solutions. The ATC B05 category explicitly covers substitutes and solutions used in place of blood, whereas PRBCs are the real biological product itself, making this mapping semantically incorrect.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab220245mlminvalue-17301880"></a>`LAB//220245//ml/min//value_[173.0,188.0)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `220245` -- CO2 production (source: `d_items`)
- abbreviation: CO2 production
- category: Respiratory
- unitname: ml/min
- linksto: chartevents
- EQ-encoded units: `ml/min`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: CO2 production (MIMIC LAB 220245, units=ml/min)
- reason: LLM proposed 'LAB//220245//ml/min' which is not in the ETHOS vocab; original rationale: CO2 production is a specific laboratory measurement from MIMIC (lab ID 220245) measured in ml/min, typically obtained during metabolic or respiratory monitoring. This maps directly to the LAB namespace using the MIMIC identifier and units structure.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab224665unkvalue-inf012"></a>`LAB//224665//UNK//value_[-inf,0.12)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `224665` -- PCA total dose (source: `d_items`)
- abbreviation: PCA total dose
- category: Pain/Sedation
- unitname: None
- linksto: chartevents
- EQ-encoded units: `UNK`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: PCA total dose (MIMIC LAB 224665, units=UNK)
- reason: PCA (Patient-Controlled Analgesia) total dose is a cumulative medication administration metric, not a laboratory measurement. This represents a nursing/pharmacy documentation value tracking opioid delivery rather than a biological specimen analysis, so it does not fit the LAB namespace which is reserved for actual laboratory test results.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab225640value-0306"></a>`LAB//225640//%//value_[0.3,0.6)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `225640` -- Differential-Eos (source: `d_items`)
- abbreviation: Differential-Eos
- category: Labs
- unitname: None
- linksto: chartevents
- EQ-encoded units: `%`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Differential-Eos (MIMIC LAB 225640, units=%)
- reason: LLM proposed 'LAB//225640//%' which is not in the ETHOS vocab; original rationale: This represents eosinophil percentage from a differential white blood cell count, a standard laboratory test. The ETHOS LAB namespace includes MIMIC lab IDs with units, and this matches the structure LAB//<MIMIC-id>//<UNITS> where 225640 is the MIMIC identifier and % is the unit of measurement.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab225672iulvalue-300420"></a>`LAB//225672//IU/L//value_[30.0,42.0)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `225672` -- Lipase (source: `d_items`)
- abbreviation: Lipase
- category: Labs
- unitname: None
- linksto: chartevents
- EQ-encoded units: `IU/L`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Lipase (MIMIC LAB 225672, units=IU/L)
- reason: LLM proposed 'LAB//225672//IU/L' which is not in the ETHOS vocab; original rationale: This is a lipase laboratory test result from MIMIC (code 225672) measured in IU/L. The ETHOS LAB namespace supports MIMIC lab identifiers with units, making this a direct structural match for the serum lipase measurement.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab226499mlvalue-31500inf"></a>`LAB//226499//mL//value_[3150.0,inf)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `226499` -- Hemodialysis Output (source: `d_items`)
- abbreviation: Hemodialysis Output
- category: Dialysis
- unitname: mL
- linksto: chartevents
- EQ-encoded units: `mL`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Hemodialysis Output (MIMIC LAB 226499, units=mL)
- reason: This EQ code represents a specific numeric range (≥3150 mL) of hemodialysis output volume, which is a binned measurement value rather than the measurement itself. ETHOS LAB tokens represent laboratory or clinical measurements by their LOINC/MIMIC identifiers and units, but do not encode specific numeric thresholds or ranges within the token structure.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab227445ngmlvalue-110200"></a>`LAB//227445//ng/mL//value_[11.0,20.0)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `227445` -- CK-MB (source: `d_items`)
- abbreviation: CK-MB
- category: Labs
- unitname: None
- linksto: chartevents
- EQ-encoded units: `ng/mL`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: CK-MB (MIMIC LAB 227445, units=ng/mL)
- reason: LLM proposed 'LAB//227445//ng/mL' which is not in the ETHOS vocab; original rationale: CK-MB (creatine kinase-MB) is a cardiac biomarker measured in ng/mL, commonly used to diagnose myocardial infarction. The ETHOS LAB namespace supports MIMIC lab identifiers with units, making LAB//227445//ng/mL the direct faithful match for this laboratory test.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab229663cmh2ovalue-inf50"></a>`LAB//229663//cmH2O//value_[-inf,5.0)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `229663` -- Pinsp (Hamilton) (source: `d_items`)
- abbreviation: Pinsp (Hamilton)
- category: Respiratory
- unitname: cmH2O
- linksto: chartevents
- EQ-encoded units: `cmH2O`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Pinsp (Hamilton) (MIMIC LAB 229663, units=cmH2O)
- reason: Pinsp (inspiratory pressure) from Hamilton ventilators is a ventilator setting/parameter rather than a traditional laboratory test result. ETHOS LAB tokens are designed for laboratory measurements (chemistry, hematology, etc.), not mechanical ventilation parameters or device settings.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab229694unkvalue-00inf"></a>`LAB//229694//UNK//value_[0.0,inf)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `229694` -- CPOT-Vocalization (CPOTa) (source: `d_items`)
- abbreviation: Vocalization-Post
- category: Pain/Sedation
- linksto: chartevents
- EQ-encoded units: `UNK`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: CPOT-Vocalization (CPOTa) (MIMIC LAB 229694, units=UNK)
- reason: CPOT (Critical-Care Pain Observation Tool) Vocalization is a behavioral pain assessment subscale score used in ICU settings, not a laboratory measurement. This is a clinical assessment score rather than a lab value, procedure, diagnosis, or medication, and therefore falls outside the ETHOS vocabulary's defined namespaces.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab50957mmollvalue-0506"></a>`LAB//50957//mmol/L//value_[0.5,0.6)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `50957` -- Lithium (source: `d_labitems`)
- category: Chemistry
- fluid: Blood
- EQ-encoded units: `mmol/L`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Lithium (MIMIC LAB 50957, units=mmol/L)
- reason: LLM proposed 'LAB//50957//mmol/L' which is not in the ETHOS vocab; original rationale: This is a lithium level measurement from MIMIC lab code 50957 in mmol/L units. The ETHOS LAB namespace supports structured lab tokens with MIMIC identifiers and units, making this a direct match for the laboratory test regardless of the specific value range.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab50964mosmkgvalue-28802930"></a>`LAB//50964//mOsm/kg//value_[288.0,293.0)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `50964` -- Osmolality, Measured (source: `d_labitems`)
- category: Chemistry
- fluid: Blood
- EQ-encoded units: `mOsm/kg`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Osmolality, Measured (MIMIC LAB 50964, units=mOsm/kg)
- reason: LLM proposed 'LAB//50964//mOsm/kg' which is not in the ETHOS vocab; original rationale: This is a direct match for serum/plasma osmolality measured in the laboratory (MIMIC LAB 50964) with units of mOsm/kg. The ETHOS LAB namespace supports MIMIC lab identifiers with their corresponding units, making this a faithful semantic match for the measured osmolality concept.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab51066mg24hrvalue-17702160"></a>`LAB//51066//mg/24hr//value_[177.0,216.0)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `51066` -- 24 hr Calcium (source: `d_labitems`)
- category: Chemistry
- fluid: Urine
- EQ-encoded units: `mg/24hr`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: 24 hr Calcium (MIMIC LAB 51066, units=mg/24hr)
- reason: The EQ code represents a laboratory measurement of 24-hour urinary calcium with values in the range 177.0-216.0 mg/24hr, which is typically within or near normal range. The candidate token refers to dietary calcium deficiency (a diagnosis), which is semantically distinct from a calcium measurement and does not faithfully represent the laboratory test result.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="lab51769unkvalue-188229"></a>`LAB//51769//UNK//value_[1.88,2.29)` _(unmapped)_

- **Family:** `LAB`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `51769` -- RUBIgGV (source: `d_labitems`)
- category: Chemistry
- fluid: Blood
- EQ-encoded units: `UNK`

**Unmapped reason (from mapping_coverage):** `value_bin_or_units_unmatched`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: RUBIgGV (MIMIC LAB 51769, units=UNK)
- reason: LLM proposed 'LAB//51769//UNK' which is not in the ETHOS vocab; original rationale: RUBIgGV refers to Rubella IgG antibody levels, a specific laboratory test (MIMIC LAB code 51769). The ETHOS LAB namespace can represent this test with the same MIMIC identifier and unknown units designation, making it a direct semantic match at the test level regardless of the specific value range.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="subject-fluid-output226600mlvalue-200300"></a>`SUBJECT_FLUID_OUTPUT//226600//mL//value_[20.0,30.0)` _(unmapped)_

- **Family:** `SUBJECT_FLUID_OUTPUT`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `226600` -- Jackson Pratt #2 (source: `d_items`)
- abbreviation: Jackson Pratt #2
- category: Drains
- unitname: mL
- linksto: outputevents
- EQ-encoded units: `mL`

**Unmapped reason (from mapping_coverage):** `ethos_token_missing/no_fluid_output_prefix_in_vocab`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Jackson Pratt #2 (MIMIC SUBJECT_FLUID_OUTPUT 226600, units=mL)
- reason: The EQ code represents a specific measurement of surgical drain output (Jackson Pratt drain #2, 20-30 mL), which is a procedural fluid output measurement. None of the candidate ETHOS tokens appropriately capture this concept - they refer to disorders of amniotic fluid, cerebrospinal fluid findings, electrolyte imbalances, or intake symptoms, rather than post-surgical drain output measurements.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="subject-fluid-output226600mlvalue-50100"></a>`SUBJECT_FLUID_OUTPUT//226600//mL//value_[5.0,10.0)` _(unmapped)_

- **Family:** `SUBJECT_FLUID_OUTPUT`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `226600` -- Jackson Pratt #2 (source: `d_items`)
- abbreviation: Jackson Pratt #2
- category: Drains
- unitname: mL
- linksto: outputevents
- EQ-encoded units: `mL`

**Unmapped reason (from mapping_coverage):** `ethos_token_missing/no_fluid_output_prefix_in_vocab`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Jackson Pratt #2 (MIMIC SUBJECT_FLUID_OUTPUT 226600, units=mL)
- reason: The EQ code represents a specific measurement of surgical drain output (Jackson Pratt drain #2), which is a procedural fluid output measurement. None of the candidate ETHOS tokens appropriately capture this concept - they refer to disorders of amniotic fluid, cerebrospinal fluid findings, electrolyte imbalances, or food/fluid intake symptoms, none of which semantically match a post-surgical drain output measurement.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```


---

### <a id="subject-fluid-output227510mlvalue-250500"></a>`SUBJECT_FLUID_OUTPUT//227510//mL//value_[25.0,50.0)` _(unmapped)_

- **Family:** `SUBJECT_FLUID_OUTPUT`
- **Mapped tiers:** (none)

**Authoritative EQ label:** MIMIC item-id `227510` -- TF Residual (source: `d_items`)
- abbreviation: TF Residual
- category: Output
- unitname: mL
- linksto: outputevents
- EQ-encoded units: `mL`

**Unmapped reason (from mapping_coverage):** `ethos_token_missing/no_fluid_output_prefix_in_vocab`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: TF Residual (MIMIC SUBJECT_FLUID_OUTPUT 227510, units=mL)
- reason: TF Residual refers to tube feeding residual volume, which is a measured output value from enteral feeding tubes used to assess gastric emptying. None of the candidates address enteral/tube feeding measurements; they cover amniotic fluid, cerebrospinal fluid, electrolyte disorders, and general food/fluid intake symptoms, none of which faithfully represent the specific clinical concept of gastric residual volume measurement.

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```

