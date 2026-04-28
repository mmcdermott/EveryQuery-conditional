# ETHOS mapping review

Per-EQ-code review of the EQ -> ETHOS code mapping produced by `build_mapping.py`. Each section shows the authoritative EQ-side label resolved through the staged Athena ontology, the verbatim LLM rationale (rendered once per EQ code where one was recorded), the candidate ETHOS tokens with vocab counts and constituent-code expansions, and a status block for the reviewer to mark approve / reject / modify.

## Reviewing principle

Default to the **tightest** mapping that still captures the EQ concept. When a candidate ETHOS token expands to constituent codes that include conditions clearly outside the EQ question (e.g. an ETHOS `ATHEROSCLEROSIS` bucket covering aortic, cerebrovascular, and peripheral vascular disease for an EQ question specifically about coronary artery atherosclerosis), lean toward `reject` or `modify`. The constituent-code lists below show every code (with its English description) the ETHOS token rolls up, so judge by reading those lines, not by the ETHOS token name in isolation.

## Table of contents

- [`DIAGNOSIS//ICD//10//I25118`](#diagnosisicd10i25118)
- [`DIAGNOSIS//ICD//9//3320`](#diagnosisicd93320)
- [`DIAGNOSIS//ICD//9//4271`](#diagnosisicd94271)
- [`DIAGNOSIS//ICD//9//5856`](#diagnosisicd95856)
- [`INFUSION_END//227536//value_[6.05,8.071587)`](#infusion-end227536value-6058071587)
- [`INFUSION_END//227536//value_[8.071587,9.379999)`](#infusion-end227536value-80715879379999)
- [`INFUSION_END//229420//value_[10.687433,23.27545)`](#infusion-end229420value-106874332327545)
- [`INFUSION_START//220949`](#infusion-start220949)
- [`INFUSION_START//221794//value_[8.004926,10.000001)`](#infusion-start221794value-800492610000001)
- [`INFUSION_START//225168//value_[284.02966,350.0)`](#infusion-start225168value-284029663500)
- [`LAB//220224//mmHg//value_[89.0,98.0)`](#lab220224mmhgvalue-890980)
- [`LAB//220339//cmH2O//value_[10.0,12.0)`](#lab220339cmh2ovalue-100120)
- [`LAB//224054//UNK//value_[2.0,3.0)`](#lab224054unkvalue-2030)
- [`LAB//224690//insp/min//value_[14.0,16.0)`](#lab224690inspminvalue-140160)
- [`LAB//226499//mL//value_[3150.0,inf)`](#lab226499mlvalue-31500inf)
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
- [`DIAGNOSIS//ICD//9//7295` _(unmapped)_](#diagnosisicd97295)
- [`LAB//220245//ml/min//value_[173.0,188.0)` _(unmapped)_](#lab220245mlminvalue-17301880)
- [`LAB//224665//UNK//value_[-inf,0.12)` _(unmapped)_](#lab224665unkvalue-inf012)
- [`LAB//225640//%//value_[0.3,0.6)` _(unmapped)_](#lab225640value-0306)
- [`LAB//225672//IU/L//value_[30.0,42.0)` _(unmapped)_](#lab225672iulvalue-300420)
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
- **Mapped tiers:** icd_crosswalk

**Authoritative EQ label:** ICD-10-CM `I25118` -- Atherosclerotic heart disease of native coronary artery with other forms of angina pectoris
- 3-char parent: `I25` -- Chronic ischemic heart disease
- SNOMED bridge: Angina co-occurrent and due to coronary arteriosclerosis (concept_id 36712983)

**LLM rationale (verbatim):**

> I25.118 is a 5-digit child of I25 (chronic ischemic heart disease). ICD//CM//CHRONIC_ISCHEMIC_HEART_DISEASE is the I25 3-char category and is the tightest ETHOS bridge that still covers the EQ concept. Earlier candidates ANGINA_PECTORIS (I20, angina without specified atherosclerosis) and ATHEROSCLEROSIS (I70, mostly peripheral and leg atherosclerosis, not coronary) were dropped during precision review since they overshoot the EQ concept.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//CHRONIC_ISCHEMIC_HEART_DISEASE`

- ETHOS vocab count: 130,280
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `I25` -- Chronic ischemic heart disease
- Constituent ICD-10-CM codes (69):
  - `I25.1` -- Atherosclerotic heart disease of native coronary artery
  - `I25.10` -- Atherosclerotic heart disease of native coronary artery without angina pectoris
  - `I25.11` -- Atherosclerotic heart disease of native coronary artery with angina pectoris
  - `I25.110` -- Atherosclerotic heart disease of native coronary artery with unstable angina pectoris
  - `I25.111` -- Atherosclerotic heart disease of native coronary artery with angina pectoris with documented spasm
  - `I25.112` -- Atherosclerotic heart disease of native coronary artery with refractory angina pectoris
  - `I25.118` -- Atherosclerotic heart disease of native coronary artery with other forms of angina pectoris
  - `I25.119` -- Atherosclerotic heart disease of native coronary artery with unspecified angina pectoris
  - `I25.2` -- Old myocardial infarction
  - `I25.3` -- Aneurysm of heart
  - `I25.4` -- Coronary artery aneurysm and dissection
  - `I25.41` -- Coronary artery aneurysm
  - `I25.42` -- Coronary artery dissection
  - `I25.5` -- Ischemic cardiomyopathy
  - `I25.6` -- Silent myocardial ischemia
  - `I25.7` -- Atherosclerosis of coronary artery bypass graft(s) and coronary artery of transplanted heart with angina pectoris
  - `I25.70` -- Atherosclerosis of coronary artery bypass graft(s), unspecified, with angina pectoris
  - `I25.700` -- Atherosclerosis of coronary artery bypass graft(s), unspecified, with unstable angina pectoris
  - `I25.701` -- Atherosclerosis of coronary artery bypass graft(s), unspecified, with angina pectoris with documented spasm
  - `I25.702` -- Atherosclerosis of coronary artery bypass graft(s), unspecified, with refractory angina pectoris
  - `I25.708` -- Atherosclerosis of coronary artery bypass graft(s), unspecified, with other forms of angina pectoris
  - `I25.709` -- Atherosclerosis of coronary artery bypass graft(s), unspecified, with unspecified angina pectoris
  - `I25.71` -- Atherosclerosis of autologous vein coronary artery bypass graft(s) with angina pectoris
  - `I25.710` -- Atherosclerosis of autologous vein coronary artery bypass graft(s) with unstable angina pectoris
  - `I25.711` -- Atherosclerosis of autologous vein coronary artery bypass graft(s) with angina pectoris with documented spasm
  - `I25.712` -- Atherosclerosis of autologous vein coronary artery bypass graft(s) with refractory angina pectoris
  - `I25.718` -- Atherosclerosis of autologous vein coronary artery bypass graft(s) with other forms of angina pectoris
  - `I25.719` -- Atherosclerosis of autologous vein coronary artery bypass graft(s) with unspecified angina pectoris
  - `I25.72` -- Atherosclerosis of autologous artery coronary artery bypass graft(s) with angina pectoris
  - `I25.720` -- Atherosclerosis of autologous artery coronary artery bypass graft(s) with unstable angina pectoris
  - ... +39 more

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

> Direct semantic match to ICD-10 G20.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PARKINSON'S_DISEASE`

- ETHOS vocab count: 5,606
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
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

> Direct match to ICD-10 I47.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PAROXYSMAL_TACHYCARDIA`

- ETHOS vocab count: 9,342
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
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
- **Mapped tiers:** icd_crosswalk

**Authoritative EQ label:** ICD-9-CM `5856` -- End stage renal disease
- SNOMED bridge: End-stage renal disease (concept_id 193782)
- ICD-10-CM crosswalk (1 code):
  - `N18.6` -- End stage renal disease

**LLM rationale (verbatim):**

> ICD-9 585.6 is ESRD; ICD-10 N18 is CKD with N18.6 specifically being ESRD. ETHOS's CKD label captures the N18 family. Alternative tokens (HCPCS UNSCHED_DIALYSIS_ESRD_PT_HOS, ENCOUNTER_FOR_CARE_INVOLVING_RENAL_DIALYSIS) are too rare to be useful as standalone matches.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//CHRONIC_KIDNEY_DISEASE_(CKD)`

- ETHOS vocab count: 84,796
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `N18` -- Chronic kidney disease (CKD)
- Constituent ICD-10-CM codes (10):
  - `N18.1` -- Chronic kidney disease, stage 1
  - `N18.2` -- Chronic kidney disease, stage 2 (mild)
  - `N18.3` -- Chronic kidney disease, stage 3 (moderate)
  - `N18.30` -- Chronic kidney disease, stage 3 unspecified
  - `N18.31` -- Chronic kidney disease, stage 3a
  - `N18.32` -- Chronic kidney disease, stage 3b
  - `N18.4` -- Chronic kidney disease, stage 4 (severe)
  - `N18.5` -- Chronic kidney disease, stage 5
  - `N18.6` -- End stage renal disease
  - `N18.9` -- Chronic kidney disease, unspecified

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="infusion-end227536value-6058071587"></a>`INFUSION_END//227536//value_[6.05,8.071587)`

- **Family:** `INFUSION_END`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `227536` -- KCl (CRRT) (source: `d_items`)
- abbreviation: KCl (CRRT)
- category: Medications
- unitname: mEq.
- linksto: inputevents
- EQ-encoded units: `value_[6.05,8.071587)`

**LLM rationale (verbatim):**

> Potassium chloride is ATC A12BA01 (mineral supplement). When administered via CRRT it is an electrolyte solution (B05X). We OR both classes since the underlying drug is K+ supplement but the route is CRRT.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//A12//MINERAL_SUPPLEMENTS`

- ETHOS vocab count: 890,010
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ATC level 2 class `A12` -- MINERAL SUPPLEMENTS
- Constituent RxNorm ingredients (127):
  - 1,2-docosahexanoyl-sn-glycero-3-phosphoserine calcium
  - 1,2-icosapentoyl-sn-glycero-3-phosphoserine calcium
  - allantoin calcium pantothenate
  - aluminum magnesium hydroxide carbonate
  - aluminum magnesium silicate
  - calcium
  - calcium acetate
  - calcium alginate
  - calcium aluminosilicate
  - calcium aluminum borosilicate
  - calcium amino acid chelate
  - calcium arsenate
  - calcium ascorbate
  - calcium aspartate
  - calcium bicarbonate
  - calcium bromide
  - calcium carbimide
  - calcium carbonate
  - calcium carbonate, precipitated
  - calcium chlorate dihydrate
  - calcium chloride
  - calcium citrate
  - calcium citrate malate
  - calcium creosotate
  - calcium fluoride
  - calcium galactogluconate bromide
  - calcium glubionate
  - calcium glucarate
  - calcium gluceptate
  - calcium gluconate
  - ... +97 more

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//B05//BLOOD_SUBSTITUTES_AND_PERFUSION_SOLUTIONS`

- ETHOS vocab count: 540,854
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ATC level 2 class `B05` -- BLOOD SUBSTITUTES AND PERFUSION SOLUTIONS
- Constituent RxNorm ingredients (375):
  - 4-aminomethylbenzoic acid
  - 6-aminocaproic acid
  - acetate
  - acetic acid
  - acyclovir
  - alanylglutamine
  - alatrofloxacin
  - albendazole
  - albumin human, USP
  - alpha tocopherol
  - aluminum acetotartrate
  - amcinonide
  - amdinocillin
  - amdinocillin pivoxil
  - amikacin
  - aminocaproate
  - ammonium chloride
  - amoxicillin
  - amphotericin B
  - ampicillin
  - anidulafungin
  - arbekacin
  - arginine
  - artemether
  - ascorbic acid
  - atovaquone
  - azidocillin
  - azithromycin
  - azlocillin
  - aztreonam
  - ... +345 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="infusion-end227536value-80715879379999"></a>`INFUSION_END//227536//value_[8.071587,9.379999)`

- **Family:** `INFUSION_END`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `227536` -- KCl (CRRT) (source: `d_items`)
- abbreviation: KCl (CRRT)
- category: Medications
- unitname: mEq.
- linksto: inputevents
- EQ-encoded units: `value_[8.071587,9.379999)`

**LLM rationale (verbatim):**

> Potassium chloride is ATC A12BA01 (mineral supplement). When administered via CRRT it is an electrolyte solution (B05X). We OR both classes since the underlying drug is K+ supplement but the route is CRRT.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//A12//MINERAL_SUPPLEMENTS`

- ETHOS vocab count: 890,010
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ATC level 2 class `A12` -- MINERAL SUPPLEMENTS
- Constituent RxNorm ingredients (127):
  - 1,2-docosahexanoyl-sn-glycero-3-phosphoserine calcium
  - 1,2-icosapentoyl-sn-glycero-3-phosphoserine calcium
  - allantoin calcium pantothenate
  - aluminum magnesium hydroxide carbonate
  - aluminum magnesium silicate
  - calcium
  - calcium acetate
  - calcium alginate
  - calcium aluminosilicate
  - calcium aluminum borosilicate
  - calcium amino acid chelate
  - calcium arsenate
  - calcium ascorbate
  - calcium aspartate
  - calcium bicarbonate
  - calcium bromide
  - calcium carbimide
  - calcium carbonate
  - calcium carbonate, precipitated
  - calcium chlorate dihydrate
  - calcium chloride
  - calcium citrate
  - calcium citrate malate
  - calcium creosotate
  - calcium fluoride
  - calcium galactogluconate bromide
  - calcium glubionate
  - calcium glucarate
  - calcium gluceptate
  - calcium gluconate
  - ... +97 more

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//B05//BLOOD_SUBSTITUTES_AND_PERFUSION_SOLUTIONS`

- ETHOS vocab count: 540,854
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ATC level 2 class `B05` -- BLOOD SUBSTITUTES AND PERFUSION SOLUTIONS
- Constituent RxNorm ingredients (375):
  - 4-aminomethylbenzoic acid
  - 6-aminocaproic acid
  - acetate
  - acetic acid
  - acyclovir
  - alanylglutamine
  - alatrofloxacin
  - albendazole
  - albumin human, USP
  - alpha tocopherol
  - aluminum acetotartrate
  - amcinonide
  - amdinocillin
  - amdinocillin pivoxil
  - amikacin
  - aminocaproate
  - ammonium chloride
  - amoxicillin
  - amphotericin B
  - ampicillin
  - anidulafungin
  - arbekacin
  - arginine
  - artemether
  - ascorbic acid
  - atovaquone
  - azidocillin
  - azithromycin
  - azlocillin
  - aztreonam
  - ... +345 more

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

> Dexmedetomidine is ATC N05CM18 (other hypnotics/sedatives); rolled up to N05 (psycholeptics) which is the broadest sedative/anxiolytic class.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//N05//PSYCHOLEPTICS`

- ETHOS vocab count: 908,963
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
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

### <a id="infusion-start220949"></a>`INFUSION_START//220949`

- **Family:** `INFUSION_START`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `220949` -- Dextrose 5% (source: `d_items`)
- abbreviation: Dextrose 5%
- category: Fluids/Intake
- unitname: mL
- linksto: inputevents

**LLM rationale (verbatim):**

> Dextrose 5% is ATC B05BA03 -- IV solution; rolled up to B05.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//B05//BLOOD_SUBSTITUTES_AND_PERFUSION_SOLUTIONS`

- ETHOS vocab count: 540,854
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ATC level 2 class `B05` -- BLOOD SUBSTITUTES AND PERFUSION SOLUTIONS
- Constituent RxNorm ingredients (375):
  - 4-aminomethylbenzoic acid
  - 6-aminocaproic acid
  - acetate
  - acetic acid
  - acyclovir
  - alanylglutamine
  - alatrofloxacin
  - albendazole
  - albumin human, USP
  - alpha tocopherol
  - aluminum acetotartrate
  - amcinonide
  - amdinocillin
  - amdinocillin pivoxil
  - amikacin
  - aminocaproate
  - ammonium chloride
  - amoxicillin
  - amphotericin B
  - ampicillin
  - anidulafungin
  - arbekacin
  - arginine
  - artemether
  - ascorbic acid
  - atovaquone
  - azidocillin
  - azithromycin
  - azlocillin
  - aztreonam
  - ... +345 more

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

> Furosemide is ATC C03CA01 (diuretic); rolled up to C03.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//C03//DIURETICS`

- ETHOS vocab count: 612,787
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
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

### <a id="infusion-start225168value-284029663500"></a>`INFUSION_START//225168//value_[284.02966,350.0)`

- **Family:** `INFUSION_START`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `225168` -- Packed Red Blood Cells (source: `d_items`)
- abbreviation: PRBC's
- category: Blood Products/Colloids
- unitname: mL
- linksto: inputevents
- EQ-encoded units: `value_[284.02966,350.0)`

**LLM rationale (verbatim):**

> PRBC transfusion belongs to ATC B05A (blood substitutes); B05 class.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//B05//BLOOD_SUBSTITUTES_AND_PERFUSION_SOLUTIONS`

- ETHOS vocab count: 540,854
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ATC level 2 class `B05` -- BLOOD SUBSTITUTES AND PERFUSION SOLUTIONS
- Constituent RxNorm ingredients (375):
  - 4-aminomethylbenzoic acid
  - 6-aminocaproic acid
  - acetate
  - acetic acid
  - acyclovir
  - alanylglutamine
  - alatrofloxacin
  - albendazole
  - albumin human, USP
  - alpha tocopherol
  - aluminum acetotartrate
  - amcinonide
  - amdinocillin
  - amdinocillin pivoxil
  - amikacin
  - aminocaproate
  - ammonium chloride
  - amoxicillin
  - amphotericin B
  - ampicillin
  - anidulafungin
  - arbekacin
  - arginine
  - artemether
  - ascorbic acid
  - atovaquone
  - azidocillin
  - azithromycin
  - azlocillin
  - aztreonam
  - ... +345 more

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab220224mmhgvalue-890980"></a>`LAB//220224//mmHg//value_[89.0,98.0)`

- **Family:** `LAB`
- **Mapped tiers:** drop_bin, quantile

**Authoritative EQ label:** MIMIC item-id `220224` -- Arterial O2 pressure (source: `d_items`)
- abbreviation: PO2 (Arterial)
- category: Labs
- unitname: mmHg
- linksto: chartevents
- EQ-encoded units: `mmHg`

_Found under tier(s): drop_bin_

#### Candidate ETHOS token: `LAB//220224//MMHG`

- ETHOS vocab count: 224,599
- Match kind: `literal`
- Mapping source: `code:strip_value_bin+upper_units`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

_Found under tier(s): quantile_

#### Candidate ETHOS token: `LAB//220224//MMHG|Q4`

- ETHOS vocab count: 0
- Match kind: `lab+next_qk`
- Mapping source: `meds-codes.parquet:values_quantiles_or_sibling_bins`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab220339cmh2ovalue-100120"></a>`LAB//220339//cmH2O//value_[10.0,12.0)`

- **Family:** `LAB`
- **Mapped tiers:** drop_bin

**Authoritative EQ label:** MIMIC item-id `220339` -- PEEP set (source: `d_items`)
- abbreviation: PEEP set
- category: Respiratory
- unitname: cmH2O
- linksto: chartevents
- EQ-encoded units: `cmH2O`

_Found under tier(s): drop_bin_

#### Candidate ETHOS token: `LAB//220339//CMH2O`

- ETHOS vocab count: 768,487
- Match kind: `literal`
- Mapping source: `code:strip_value_bin+upper_units`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab224054unkvalue-2030"></a>`LAB//224054//UNK//value_[2.0,3.0)`

- **Family:** `LAB`
- **Mapped tiers:** drop_bin

**Authoritative EQ label:** MIMIC item-id `224054` -- Braden Sensory Perception (source: `d_items`)
- abbreviation: Braden Sensory Perception
- category: Skin - Assessment
- linksto: chartevents
- EQ-encoded units: `UNK`

_Found under tier(s): drop_bin_

#### Candidate ETHOS token: `LAB//224054//UNK`

- ETHOS vocab count: 808,972
- Match kind: `literal`
- Mapping source: `code:strip_value_bin+upper_units`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab224690inspminvalue-140160"></a>`LAB//224690//insp/min//value_[14.0,16.0)`

- **Family:** `LAB`
- **Mapped tiers:** drop_bin, quantile

**Authoritative EQ label:** MIMIC item-id `224690` -- Respiratory Rate (Total) (source: `d_items`)
- abbreviation: Respiratory Rate (Total)
- category: Respiratory
- unitname: insp/min
- linksto: chartevents
- EQ-encoded units: `insp/min`

_Found under tier(s): drop_bin_

#### Candidate ETHOS token: `LAB//224690//INSP/MIN`

- ETHOS vocab count: 710,225
- Match kind: `literal`
- Mapping source: `code:strip_value_bin+upper_units`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

_Found under tier(s): quantile_

#### Candidate ETHOS token: `LAB//224690//INSP/MIN|Q2`

- ETHOS vocab count: 0
- Match kind: `lab+next_qk`
- Mapping source: `meds-codes.parquet:values_quantiles_or_sibling_bins`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab226499mlvalue-31500inf"></a>`LAB//226499//mL//value_[3150.0,inf)`

- **Family:** `LAB`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `226499` -- Hemodialysis Output (source: `d_items`)
- abbreviation: Hemodialysis Output
- category: Dialysis
- unitname: mL
- linksto: chartevents
- EQ-encoded units: `mL`

**LLM rationale (verbatim):**

> Hemodialysis-output measurement happens during dialysis encounters; we proxy via the dialysis encounter (ICD-10-CM Z49). The previously-OR'd CHRONIC_KIDNEY_DISEASE_(CKD) was dropped during precision review -- it includes all CKD stages (1-5) so any patient with mild CKD triggers it, but the EQ measurement only happens during active dialysis.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ICD//CM//ENCOUNTER_FOR_CARE_INVOLVING_RENAL_DIALYSIS`

- ETHOS vocab count: 70
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ICD-10-CM 3-char category `Z49` -- Encounter for care involving renal dialysis
- Constituent ICD-10-CM codes (6):
  - `Z49.0` -- Preparatory care for renal dialysis
  - `Z49.01` -- Encounter for fitting and adjustment of extracorporeal dialysis catheter
  - `Z49.02` -- Encounter for fitting and adjustment of peritoneal dialysis catheter
  - `Z49.3` -- Encounter for adequacy testing for dialysis
  - `Z49.31` -- Encounter for adequacy testing for hemodialysis
  - `Z49.32` -- Encounter for adequacy testing for peritoneal dialysis

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="lab227073meqlvalue-170190"></a>`LAB//227073//mEq/L//value_[17.0,19.0)`

- **Family:** `LAB`
- **Mapped tiers:** drop_bin

**Authoritative EQ label:** MIMIC item-id `227073` -- Anion gap (source: `d_items`)
- abbreviation: Anion gap
- category: Labs
- unitname: None
- linksto: chartevents
- EQ-encoded units: `mEq/L`

_Found under tier(s): drop_bin_

#### Candidate ETHOS token: `LAB//227073//MEQ/L`

- ETHOS vocab count: 405,801
- Match kind: `literal`
- Mapping source: `code:strip_value_bin+upper_units`
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

> Pressure-ulcer length measurement is recorded for patients who already have a pressure ulcer; this is the underlying ICD diagnosis.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PRESSURE_ULCER`

- ETHOS vocab count: 11,617
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
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
- **Mapped tiers:** drop_bin, quantile

**Authoritative EQ label:** MIMIC item-id `51274` -- PT (source: `d_labitems`)
- category: Hematology
- fluid: Blood
- EQ-encoded units: `sec`

_Found under tier(s): drop_bin_

#### Candidate ETHOS token: `LAB//51274//SEC`

- ETHOS vocab count: 1,316,289
- Match kind: `literal`
- Mapping source: `code:strip_value_bin+upper_units`
- Inferred source: ETHOS-internal token (no Athena ontology bridge).

_Found under tier(s): quantile_

#### Candidate ETHOS token: `LAB//51274//SEC|Q6`

- ETHOS vocab count: 0
- Match kind: `lab+next_qk`
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

> Carbidopa-Levodopa is ATC N04BA02; rolling up to N04 class.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//N04//ANTI-PARKINSON_DRUGS`

- ETHOS vocab count: 162,103
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
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

> Gabapentin is ATC N03AX12; rolling up to N03 class.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//N03//ANTIEPILEPTICS`

- ETHOS vocab count: 1,322,698
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
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

> Mupirocin nasal is ATC R01AX06; the same active ingredient as topical D06AX09. We OR both since either nasal or dermatological tokens could reasonably reflect the medication being administered.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//D06//ANTIBIOTICS_AND_CHEMOTHERAPEUTICS_FOR_DERMATOLOGICAL_USE`

- ETHOS vocab count: 294,768
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ATC level 2 class `D06` -- ANTIBIOTICS AND CHEMOTHERAPEUTICS FOR DERMATOLOGICAL USE
- Constituent RxNorm ingredients (40):
  - acyclovir
  - amikacin
  - bacitracin
  - bacitracin methylene disalicylate
  - berdazimer
  - chloramphenicol
  - chlortetracycline
  - demeclocycline
  - docosanol
  - edoxudine
  - fusidate
  - gentamicin
  - idoxuridine
  - imiquimod
  - ingenol mebutate
  - inosine
  - lysozyme
  - mafenide
  - metronidazole
  - mupirocin
  - neomycin
  - oxytetracycline
  - ozenoxacin
  - penciclovir
  - podofilox
  - retapamulin
  - rifamycin SV
  - rifamycins
  - rifaximin
  - silver sulfadiazine
  - ... +10 more

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//R01//NASAL_PREPARATIONS`

- ETHOS vocab count: 125,895
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
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

> Captopril is ATC C09AA01; rolling up to C09 class.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//C09//AGENTS_ACTING_ON_THE_RENIN-ANGIOTENSIN_SYSTEM`

- ETHOS vocab count: 285,699
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
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

> Doxycycline is ATC J01AA02; rolling up to J01 class.

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//J01//ANTIBIOTICS_AND_ANTIBACTERIALS_FOR_SYSTEMIC_USE`

- ETHOS vocab count: 2,103,569
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
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

#### Candidate ETHOS token: `MEDS_DEATH`

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

> ETHOS uses chunked ICD-PCS tokens that don't carry procedure semantics directly. We map this procedure to the underlying fracture diagnosis as a proxy: predicting the fracture diagnosis is a reasonable proxy for the reduction procedure. The previously-OR'd FRACTURE_OF_LOWER_LEG_INCLUDING_ANKLE was dropped during precision review since it covers different anatomy than the foot-bone procedure. Mapping is still loose -- AUC interpretation should reflect that we're predicting the indication, not the procedure itself.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//FRACTURE_OF_FOOT_AND_TOE_EXCEPT_ANKLE`

- ETHOS vocab count: 1,544
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `S92` -- Fracture of foot and toe, except ankle
- Constituent ICD-10-CM codes (1630):
  - `S92.0` -- Fracture of calcaneus
  - `S92.00` -- Unspecified fracture of calcaneus
  - `S92.001` -- Unspecified fracture of right calcaneus
  - `S92.001A` -- Unspecified fracture of right calcaneus, initial encounter for closed fracture
  - `S92.001B` -- Unspecified fracture of right calcaneus, initial encounter for open fracture
  - `S92.001D` -- Unspecified fracture of right calcaneus, subsequent encounter for fracture with routine healing
  - `S92.001G` -- Unspecified fracture of right calcaneus, subsequent encounter for fracture with delayed healing
  - `S92.001K` -- Unspecified fracture of right calcaneus, subsequent encounter for fracture with nonunion
  - `S92.001P` -- Unspecified fracture of right calcaneus, subsequent encounter for fracture with malunion
  - `S92.001S` -- Unspecified fracture of right calcaneus, sequela
  - `S92.002` -- Unspecified fracture of left calcaneus
  - `S92.002A` -- Unspecified fracture of left calcaneus, initial encounter for closed fracture
  - `S92.002B` -- Unspecified fracture of left calcaneus, initial encounter for open fracture
  - `S92.002D` -- Unspecified fracture of left calcaneus, subsequent encounter for fracture with routine healing
  - `S92.002G` -- Unspecified fracture of left calcaneus, subsequent encounter for fracture with delayed healing
  - `S92.002K` -- Unspecified fracture of left calcaneus, subsequent encounter for fracture with nonunion
  - `S92.002P` -- Unspecified fracture of left calcaneus, subsequent encounter for fracture with malunion
  - `S92.002S` -- Unspecified fracture of left calcaneus, sequela
  - `S92.009` -- Unspecified fracture of unspecified calcaneus
  - `S92.009A` -- Unspecified fracture of unspecified calcaneus, initial encounter for closed fracture
  - `S92.009B` -- Unspecified fracture of unspecified calcaneus, initial encounter for open fracture
  - `S92.009D` -- Unspecified fracture of unspecified calcaneus, subsequent encounter for fracture with routine healing
  - `S92.009G` -- Unspecified fracture of unspecified calcaneus, subsequent encounter for fracture with delayed healing
  - `S92.009K` -- Unspecified fracture of unspecified calcaneus, subsequent encounter for fracture with nonunion
  - `S92.009P` -- Unspecified fracture of unspecified calcaneus, subsequent encounter for fracture with malunion
  - `S92.009S` -- Unspecified fracture of unspecified calcaneus, sequela
  - `S92.01` -- Fracture of body of calcaneus
  - `S92.011` -- Displaced fracture of body of right calcaneus
  - `S92.011A` -- Displaced fracture of body of right calcaneus, initial encounter for closed fracture
  - `S92.011B` -- Displaced fracture of body of right calcaneus, initial encounter for open fracture
  - ... +1600 more

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

> EQ's TIMELINE//START fires at the first event of a patient's record. Token frequencies match closely (EQ n=200,773 / ETHOS n=297,949), and ETHOS uses HOSPITAL_ADMISSION as its admission marker.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `HOSPITAL_ADMISSION`

- ETHOS vocab count: 297,949
- Match kind: `literal`
- Mapping source: `direct:event_alignment`
- Inferred source: ETHOS hospital-admission marker (synthetic event token).

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="diagnosisicd97295"></a>`DIAGNOSIS//ICD//9//7295` _(unmapped)_

- **Family:** `DIAGNOSIS`
- **Mapped tiers:** (none)

**Authoritative EQ label:** ICD-9-CM `7295` -- Pain in limb
- SNOMED bridge: Pain in limb (concept_id 138525)
- ICD-10-CM crosswalk (3 codes):
  - `M79.6` -- Pain in limb, hand, foot, fingers and toes
  - `M79.60` -- Pain in limb, unspecified
  - `M79.609` -- Pain in unspecified limb

**Unmapped reason (from mapping_coverage):** `ethos_token_missing/diagnosis_uses_descriptive_label_not_icd_code`

**YAML rationale (`crosswalks/mimic_items.yaml`):**

- description: Pain in soft tissues of limb (ICD-9 729.5)
- reason: ETHOS has no specific limb-pain or musculoskeletal-pain ICD-10-CM 3-char category. The previously-proposed proxies OTHER_DISORDERS_OF_MUSCLE (M62, generic muscle disorder grab-bag) and PAIN_NOT_ELSEWHERE_CLASSIFIED (R52, unspecified pain) were dropped during precision review -- both overshoot to unrelated conditions and neither captures the limb-specific clinical concept.

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

- description: PCA total dose (patient-controlled analgesia), low-dose bin
- reason: The specific bin [-inf, 0.12) represents very low / no PCA dose -- which is the OPPOSITE of an analgesic-administration event. Mapping to ATC//N02//ANALGESICS would invert the AUROC. No reliable ETHOS counterpart.

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

- description: CK-MB elevated bin (cardiac biomarker, MIMIC chartevents 227445)
- reason: Removed during precision review. The previously-proposed bridge ACUTE_MYOCARDIAL_INFARCTION is a behavioral proxy (CK-MB elevation correlates with AMI but also with post-cardiac-surgery state, myocarditis, severe renal failure, rhabdomyolysis), not a vocabulary equivalence. Predicting "AMI Dx appears" when "elevated CK-MB measurement appears" yields asymmetric false positives and false negatives depending on cohort composition. No tighter ETHOS token is available; safer to mark unmappable than to score the loose proxy.

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

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

_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._

**STATUS:** [ ] confirm no ETHOS counterpart  [ ] propose a candidate (notes below)

**NOTES:**

```

```

