# ETHOS mapping review

Per-EQ-code review of the EQ -> ETHOS code mapping produced by `build_mapping.py`. Each section shows the authoritative EQ-side label resolved through the staged Athena ontology, the verbatim LLM rationale (rendered once per EQ code where one was recorded), the candidate ETHOS tokens with vocab counts and constituent-code expansions, and a status block for the reviewer to mark approve / reject / modify.

## Table of contents

- [`DIAGNOSIS//ICD//10//I25118`](#diagnosisicd10i25118)
- [`DIAGNOSIS//ICD//9//3320`](#diagnosisicd93320)
- [`DIAGNOSIS//ICD//9//4271`](#diagnosisicd94271)
- [`DIAGNOSIS//ICD//9//5856`](#diagnosisicd95856)
- [`DIAGNOSIS//ICD//9//7295`](#diagnosisicd97295)
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
- [`LAB//227445//ng/mL//value_[11.0,20.0)`](#lab227445ngmlvalue-110200)
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
- [`LAB//220245//ml/min//value_[173.0,188.0)` _(unmapped)_](#lab220245mlminvalue-17301880)
- [`LAB//224665//UNK//value_[-inf,0.12)` _(unmapped)_](#lab224665unkvalue-inf012)
- [`LAB//225640//%//value_[0.3,0.6)` _(unmapped)_](#lab225640value-0306)
- [`LAB//225672//IU/L//value_[30.0,42.0)` _(unmapped)_](#lab225672iulvalue-300420)
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

> I25.118 is a 5-digit child of I25 (chronic ischemic heart disease) with angina pectoris. The three ETHOS tokens together capture the I25 family semantics; we OR them so any of the three counts as a hit.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//ANGINA_PECTORIS`

- ETHOS vocab count: 4,650
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `I20` -- Angina pectoris
- Constituent ICD-10-CM codes (7): I20.0, I20.1, I20.2, I20.8, I20.81, I20.89, I20.9

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//ATHEROSCLEROSIS`

- ETHOS vocab count: 12,922
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `I70` -- Atherosclerosis
- Constituent ICD-10-CM codes (295): I70.0, I70.1, I70.2, I70.20, I70.201, I70.202, I70.203, I70.208, I70.209, I70.21, I70.211, I70.212, I70.213, I70.218, I70.219, I70.22, I70.221, I70.222, I70.223, I70.228, I70.229, I70.23, I70.231, I70.232, I70.233, I70.234, I70.235, I70.238, I70.239, I70.24, ... +265 more

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//CHRONIC_ISCHEMIC_HEART_DISEASE`

- ETHOS vocab count: 130,280
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `I25` -- Chronic ischemic heart disease
- Constituent ICD-10-CM codes (69): I25.1, I25.10, I25.11, I25.110, I25.111, I25.112, I25.118, I25.119, I25.2, I25.3, I25.4, I25.41, I25.42, I25.5, I25.6, I25.7, I25.70, I25.700, I25.701, I25.702, I25.708, I25.709, I25.71, I25.710, I25.711, I25.712, I25.718, I25.719, I25.72, I25.720, ... +39 more

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
- ICD-10-CM crosswalk (7 codes): G20, G20.A, G20.A1, G20.A2, G20.B, G20.B1, G20.B2

**LLM rationale (verbatim):**

> Direct semantic match to ICD-10 G20.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PARKINSON'S_DISEASE`

- ETHOS vocab count: 5,606
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `G20` -- Parkinson's disease
- Constituent ICD-10-CM codes (7): G20.A, G20.A1, G20.A2, G20.B, G20.B1, G20.B2, G20.C

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
- ICD-10-CM crosswalk (0 codes): (none)

**LLM rationale (verbatim):**

> Direct match to ICD-10 I47.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PAROXYSMAL_TACHYCARDIA`

- ETHOS vocab count: 9,342
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `I47` -- Paroxysmal tachycardia
- Constituent ICD-10-CM codes (10): I47.0, I47.1, I47.10, I47.11, I47.19, I47.2, I47.20, I47.21, I47.29, I47.9

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
- ICD-10-CM crosswalk (1 code): N18.6

**LLM rationale (verbatim):**

> ICD-9 585.6 is ESRD; ICD-10 N18 is CKD with N18.6 specifically being ESRD. ETHOS's CKD label captures the N18 family. Alternative tokens (HCPCS UNSCHED_DIALYSIS_ESRD_PT_HOS, ENCOUNTER_FOR_CARE_INVOLVING_RENAL_DIALYSIS) are too rare to be useful as standalone matches.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//CHRONIC_KIDNEY_DISEASE_(CKD)`

- ETHOS vocab count: 84,796
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `N18` -- Chronic kidney disease (CKD)
- Constituent ICD-10-CM codes (10): N18.1, N18.2, N18.3, N18.30, N18.31, N18.32, N18.4, N18.5, N18.6, N18.9

**STATUS:** [ ] approve  [ ] reject  [ ] modify

**NOTES:**

```

```


---

### <a id="diagnosisicd97295"></a>`DIAGNOSIS//ICD//9//7295`

- **Family:** `DIAGNOSIS`
- **Mapped tiers:** icd_crosswalk

**Authoritative EQ label:** ICD-9-CM `7295` -- Pain in limb
- SNOMED bridge: Pain in limb (concept_id 138525)
- ICD-10-CM crosswalk (3 codes): M79.6, M79.60, M79.609

**LLM rationale (verbatim):**

> ETHOS has no exact "pain in limb" token. The two listed are the closest semantic proxies (other muscle disorders + general unclassified pain); mapping is loose so the AUC for this code should be interpreted carefully.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//OTHER_DISORDERS_OF_MUSCLE`

- ETHOS vocab count: 4,479
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `M62` -- Other disorders of muscle
- Constituent ICD-10-CM codes (174): M62.0, M62.00, M62.01, M62.011, M62.012, M62.019, M62.02, M62.021, M62.022, M62.029, M62.03, M62.031, M62.032, M62.039, M62.04, M62.041, M62.042, M62.049, M62.05, M62.051, M62.052, M62.059, M62.06, M62.061, M62.062, M62.069, M62.07, M62.071, M62.072, M62.079, ... +144 more

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//PAIN_NOT_ELSEWHERE_CLASSIFIED`

- ETHOS vocab count: 35,199
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `G89` -- Pain, not elsewhere classified
- Constituent ICD-10-CM codes (12): G89.0, G89.1, G89.11, G89.12, G89.18, G89.2, G89.21, G89.22, G89.28, G89.29, G89.3, G89.4

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
- Constituent RxNorm ingredients (127): 1,2-docosahexanoyl-sn-glycero-3-phosphoserine calcium, 1,2-icosapentoyl-sn-glycero-3-phosphoserine calcium, allantoin calcium pantothenate, aluminum magnesium hydroxide carbonate, aluminum magnesium silicate, calcium, calcium acetate, calcium alginate, calcium aluminosilicate, calcium aluminum borosilicate, calcium amino acid chelate, calcium arsenate, calcium ascorbate, calcium aspartate, calcium bicarbonate, calcium bromide, calcium carbimide, calcium carbonate, calcium carbonate, precipitated, calcium chlorate dihydrate, calcium chloride, calcium citrate, calcium citrate malate, calcium creosotate, calcium fluoride, calcium galactogluconate bromide, calcium glubionate, calcium glucarate, calcium gluceptate, calcium gluconate, ... +97 more

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//B05//BLOOD_SUBSTITUTES_AND_PERFUSION_SOLUTIONS`

- ETHOS vocab count: 540,854
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ATC level 2 class `B05` -- BLOOD SUBSTITUTES AND PERFUSION SOLUTIONS
- Constituent RxNorm ingredients (375): 4-aminomethylbenzoic acid, 6-aminocaproic acid, acetate, acetic acid, acyclovir, alanylglutamine, alatrofloxacin, albendazole, albumin human, USP, alpha tocopherol, aluminum acetotartrate, amcinonide, amdinocillin, amdinocillin pivoxil, amikacin, aminocaproate, ammonium chloride, amoxicillin, amphotericin B, ampicillin, anidulafungin, arbekacin, arginine, artemether, ascorbic acid, atovaquone, azidocillin, azithromycin, azlocillin, aztreonam, ... +345 more

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
- Constituent RxNorm ingredients (127): 1,2-docosahexanoyl-sn-glycero-3-phosphoserine calcium, 1,2-icosapentoyl-sn-glycero-3-phosphoserine calcium, allantoin calcium pantothenate, aluminum magnesium hydroxide carbonate, aluminum magnesium silicate, calcium, calcium acetate, calcium alginate, calcium aluminosilicate, calcium aluminum borosilicate, calcium amino acid chelate, calcium arsenate, calcium ascorbate, calcium aspartate, calcium bicarbonate, calcium bromide, calcium carbimide, calcium carbonate, calcium carbonate, precipitated, calcium chlorate dihydrate, calcium chloride, calcium citrate, calcium citrate malate, calcium creosotate, calcium fluoride, calcium galactogluconate bromide, calcium glubionate, calcium glucarate, calcium gluceptate, calcium gluconate, ... +97 more

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ATC//B05//BLOOD_SUBSTITUTES_AND_PERFUSION_SOLUTIONS`

- ETHOS vocab count: 540,854
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ATC level 2 class `B05` -- BLOOD SUBSTITUTES AND PERFUSION SOLUTIONS
- Constituent RxNorm ingredients (375): 4-aminomethylbenzoic acid, 6-aminocaproic acid, acetate, acetic acid, acyclovir, alanylglutamine, alatrofloxacin, albendazole, albumin human, USP, alpha tocopherol, aluminum acetotartrate, amcinonide, amdinocillin, amdinocillin pivoxil, amikacin, aminocaproate, ammonium chloride, amoxicillin, amphotericin B, ampicillin, anidulafungin, arbekacin, arginine, artemether, ascorbic acid, atovaquone, azidocillin, azithromycin, azlocillin, aztreonam, ... +345 more

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
- Constituent RxNorm ingredients (143): Valeriana officinalis whole extract, acepromazine, acetophenazine, allobarbital, alprazolam, amisulpride, amobarbital, aprobarbital, aripiprazole, asenapine, barbital, benperidol, brexpiprazole, bromazepam, bromide ion, bromperidol, brotizolam, buspirone, butobarbital, butylvinal, captodiamine, carbromal, cariprazine, chloral betaine, chlordiazepoxide, chlormethiazole, chlorproethazine, chlorpromazine, chlorprothixene, clobazam, ... +113 more

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
- Constituent RxNorm ingredients (375): 4-aminomethylbenzoic acid, 6-aminocaproic acid, acetate, acetic acid, acyclovir, alanylglutamine, alatrofloxacin, albendazole, albumin human, USP, alpha tocopherol, aluminum acetotartrate, amcinonide, amdinocillin, amdinocillin pivoxil, amikacin, aminocaproate, ammonium chloride, amoxicillin, amphotericin B, ampicillin, anidulafungin, arbekacin, arginine, artemether, ascorbic acid, atovaquone, azidocillin, azithromycin, azlocillin, aztreonam, ... +345 more

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
- Constituent RxNorm ingredients (35): althiazide, amiloride, bendroflumethiazide, bumetanide, buthiazide, canrenoate, canrenone, chlorothiazide, chlorthalidone, cicletanine, clopamide, conivaptan, cyclopenthiazide, cyclothiazide, eplerenone, ethacrynate, finerenone, furosemide, hydrochlorothiazide, hydroflumethiazide, indapamide, mefruside, mersalyl, methyclothiazide, metolazone, piretanide, polythiazide, quinethazone, spironolactone, theobromine, ... +5 more

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
- Constituent RxNorm ingredients (375): 4-aminomethylbenzoic acid, 6-aminocaproic acid, acetate, acetic acid, acyclovir, alanylglutamine, alatrofloxacin, albendazole, albumin human, USP, alpha tocopherol, aluminum acetotartrate, amcinonide, amdinocillin, amdinocillin pivoxil, amikacin, aminocaproate, ammonium chloride, amoxicillin, amphotericin B, ampicillin, anidulafungin, arbekacin, arginine, artemether, ascorbic acid, atovaquone, azidocillin, azithromycin, azlocillin, aztreonam, ... +345 more

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

> Hemodialysis-output measurement happens during dialysis encounters; we proxy via the dialysis encounter and underlying CKD diagnosis tokens. Loose mapping (predicting indication/encounter, not the measurement).

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ICD//CM//CHRONIC_KIDNEY_DISEASE_(CKD)`

- ETHOS vocab count: 84,796
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ICD-10-CM 3-char category `N18` -- Chronic kidney disease (CKD)
- Constituent ICD-10-CM codes (10): N18.1, N18.2, N18.3, N18.30, N18.31, N18.32, N18.4, N18.5, N18.6, N18.9

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ICD//CM//ENCOUNTER_FOR_CARE_INVOLVING_RENAL_DIALYSIS`

- ETHOS vocab count: 70
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ICD-10-CM 3-char category `Z49` -- Encounter for care involving renal dialysis
- Constituent ICD-10-CM codes (6): Z49.0, Z49.01, Z49.02, Z49.3, Z49.31, Z49.32

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

### <a id="lab227445ngmlvalue-110200"></a>`LAB//227445//ng/mL//value_[11.0,20.0)`

- **Family:** `LAB`
- **Mapped tiers:** mimic_item_crosswalk

**Authoritative EQ label:** MIMIC item-id `227445` -- CK-MB (source: `d_items`)
- abbreviation: CK-MB
- category: Labs
- unitname: None
- linksto: chartevents
- EQ-encoded units: `ng/mL`

**LLM rationale (verbatim):**

> CK-MB is ordered when myocardial infarction is suspected; elevated values (the [11, 20) ng/mL bin is clearly elevated -- normal CK-MB is <5 ng/mL) strongly correlate with AMI. Proxy via AMI ICD label.

_Found under tier(s): mimic_item_crosswalk_

#### Candidate ETHOS token: `ICD//CM//ACUTE_MYOCARDIAL_INFARCTION`

- ETHOS vocab count: 13,427
- Match kind: `literal`
- Mapping source: `physionet/mimic-iv-demo:icu/d_items.csv+llm`
- Inferred source: ICD-10-CM 3-char category `I21` -- Acute myocardial infarction
- Constituent ICD-10-CM codes (17): I21.0, I21.01, I21.02, I21.09, I21.1, I21.11, I21.19, I21.2, I21.21, I21.29, I21.3, I21.4, I21.9, I21.A, I21.A1, I21.A9, I21.B

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
- Constituent ICD-10-CM codes (207): L89.0, L89.00, L89.000, L89.001, L89.002, L89.003, L89.004, L89.006, L89.009, L89.01, L89.010, L89.011, L89.012, L89.013, L89.014, L89.016, L89.019, L89.02, L89.020, L89.021, L89.022, L89.023, L89.024, L89.026, L89.029, L89.1, L89.10, L89.100, L89.101, L89.102, ... +177 more

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
- Constituent RxNorm ingredients (32): amantadine, apomorphine, benztropine, biperiden, bornaprine, bromocriptine, budipine, cabergoline, dexetimide, dihydroergocryptine, diphenhydramine, entacapone, ethybenztropine, etilevodopa, istradefylline, levodopa, methixene, opicapone, orphenadrine, pergolide, piribedil, pramipexole, procyclidine, profenamine, rasagiline, ropinirole, rotigotine, safinamide, selegiline, tolcapone, ... +2 more

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
- Constituent RxNorm ingredients (45): aminobutyrate, barbexaclone, beclamide, brivaracetam, cannabidiol, carbamazepine, cenobamate, clonazepam, dipropylacetamide, eslicarbazepine, ethosuximide, ethotoin, ezogabine, felbamate, fenfluramine, fosphenytoin, gabapentin, ganaxolone, lacosamide, lamotrigine, levetiracetam, mephenytoin, mephobarbital, metharbital, methsuximide, oxcarbazepine, paramethadione, perampanel, phenacemide, phenobarbital, ... +15 more

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
- Constituent RxNorm ingredients (40): acyclovir, amikacin, bacitracin, bacitracin methylene disalicylate, berdazimer, chloramphenicol, chlortetracycline, demeclocycline, docosanol, edoxudine, fusidate, gentamicin, idoxuridine, imiquimod, ingenol mebutate, inosine, lysozyme, mafenide, metronidazole, mupirocin, neomycin, oxytetracycline, ozenoxacin, penciclovir, podofilox, retapamulin, rifamycin SV, rifamycins, rifaximin, silver sulfadiazine, ... +10 more

_Found under tier(s): atc_crosswalk_

#### Candidate ETHOS token: `ATC//R01//NASAL_PREPARATIONS`

- ETHOS vocab count: 125,895
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ATC level 2 class `R01` -- NASAL PREPARATIONS
- Constituent RxNorm ingredients (43): all-trans-retinol, antazoline, azelastine, beclomethasone, betamethasone, budesonide, calcium, ciclesonide, cromoglycate, cromolyn, cyclopentamine, dexamethasone, ephedrine, epinephrine, fenoxazoline, flunisolide, fluticasone, framycetin, hexamidine, hyaluronate, hydrocortisone, indanazoline, ipratropium, isospaglumic acid, levocabastine, mometasone, mupirocin, naphazoline, nedocromil, olopatadine, ... +13 more

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
- Constituent RxNorm ingredients (25): aliskiren, azilsartan, benazepril, candesartan, captopril, cilazapril, enalapril, enalaprilat, eprosartan, fosinopril, imidapril, irbesartan, lisinopril, losartan, moexipril, olmesartan, perindopril, quinapril, ramipril, sparsentan, spirapril, telmisartan, trandolapril, valsartan, zofenopril

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
- Constituent RxNorm ingredients (187): amdinocillin, amdinocillin pivoxil, amikacin, amoxicillin, ampicillin, arbekacin, avibactam, azidocillin, azithromycin, azlocillin, aztreonam, bacampicillin, bacitracin, bacitracin methylene disalicylate, brodimoprim, carbenicillin, cefaclor, cefadroxil, cefamandole, cefatrizine, cefazolin, cefdinir, cefditoren, cefepime, cefetamet, cefiderocol, cefixime, cefmetazole, cefodizime, cefonicid, ... +157 more

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
- ICD-10-PCS crosswalk (0 codes): (none)

**LLM rationale (verbatim):**

> ETHOS uses chunked ICD-PCS tokens that don't carry procedure semantics directly. We map this procedure to the underlying fracture diagnosis as a proxy: predicting the fracture diagnosis is a reasonable proxy for the reduction procedure. Mapping is loose; AUC interpretation should reflect that we're predicting the indication, not the procedure itself.

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//FRACTURE_OF_FOOT_AND_TOE_EXCEPT_ANKLE`

- ETHOS vocab count: 1,544
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `S92` -- Fracture of foot and toe, except ankle
- Constituent ICD-10-CM codes (1630): S92.0, S92.00, S92.001, S92.001A, S92.001B, S92.001D, S92.001G, S92.001K, S92.001P, S92.001S, S92.002, S92.002A, S92.002B, S92.002D, S92.002G, S92.002K, S92.002P, S92.002S, S92.009, S92.009A, S92.009B, S92.009D, S92.009G, S92.009K, S92.009P, S92.009S, S92.01, S92.011, S92.011A, S92.011B, ... +1600 more

_Found under tier(s): icd_crosswalk_

#### Candidate ETHOS token: `ICD//CM//FRACTURE_OF_LOWER_LEG_INCLUDING_ANKLE`

- ETHOS vocab count: 4,413
- Match kind: `literal`
- Mapping source: `llm:claude_clinical_knowledge`
- Inferred source: ICD-10-CM 3-char category `S82` -- Fracture of lower leg, including ankle
- Constituent ICD-10-CM codes (3345): S82.0, S82.00, S82.001, S82.001A, S82.001B, S82.001C, S82.001D, S82.001E, S82.001F, S82.001G, S82.001H, S82.001J, S82.001K, S82.001M, S82.001N, S82.001P, S82.001Q, S82.001R, S82.001S, S82.002, S82.002A, S82.002B, S82.002C, S82.002D, S82.002E, S82.002F, S82.002G, S82.002H, S82.002J, S82.002K, ... +3315 more

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

