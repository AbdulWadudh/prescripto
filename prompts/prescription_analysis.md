# Prescription Analysis — Field Reference & Documentation

## Overview

When a prescription image or PDF is uploaded, it is passed to Qwen2.5-VL (the vision model) along with the system prompt from `default_prompt.txt`. The model reads all visible text, translates non-English content to English, interprets Indian medical shorthand, and returns a structured JSON object.

That JSON is then:
1. Rendered into a formatted Markdown report displayed in the chat
2. Saved to disk as `prescription.json` and `prescription.md`
3. Stored in the local SQLite database (`prescriptions.db`)

---

## Language Handling

All structured fields are output in **English only**. The model translates or transliterates any non-English text before placing it in a field.

- Doctor/patient names in regional scripts are transliterated to Roman script
- Diagnosis, complaints, instructions written in regional languages are translated to English
- Abbreviations are expanded to full English (e.g. `BD` → `Twice Daily`, `F` → `Female`)

The only exception is `raw_transcription`, which stores verbatim visible text in the original script(s) as they appear on the prescription.

**Supported languages:** Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, English, and any mix.

---

## JSON Output Schema

```json
{
  "languages_detected": ["Malayalam", "English"],
  "doctor": {
    "name": "Dr. T. M. Sreedharan",
    "qualifications": "MBBS, M.D. Paediatrics",
    "designation": "Registrar",
    "specialty": "Paediatrics",
    "clinic_hospital": "CHC, Nemmara",
    "address": null,
    "contact": "8886993168",
    "registration_number": "52547",
    "timing": "7:00 to 8:45, 15:30 to 19:30"
  },
  "patient": {
    "name": "Ashvika",
    "age": "4 years",
    "gender": "Female",
    "date": "20-09-2022",
    "patient_id": null,
    "contact": null,
    "address": null
  },
  "clinical": {
    "chief_complaints": ["Cough", "Fever"],
    "history": [],
    "diagnosis": ["Upper Respiratory Tract Infection"],
    "vitals": {
      "Weight": "13.25 kg",
      "RR": "22/min"
    }
  },
  "medications": [
    {
      "name": "SYP CALPOL (250/5)",
      "type": "Syrup",
      "dose": "4 mL",
      "frequency": "Every 6 Hours",
      "duration": "3 days",
      "instructions": null
    }
  ],
  "investigations": [],
  "advice": ["Rest", "Adequate fluids"],
  "follow_up": "Review after 5 days if no improvement",
  "summary": "4-year-old female patient Ashvika presenting with URTI...",
  "confidence_notes": "Doctor name partially legible — transliterated from Malayalam script",
  "raw_transcription": "ഡോ. ടി. എം. ശ്രീധരൻ MBBS CHC നെമ്മാറ..."
}
```

---

## Field Reference

### `languages_detected`
Array of all languages identified anywhere on the prescription.
Example: `["Malayalam", "English"]`

---

### `doctor`

| Field | Description | Example |
|---|---|---|
| `name` | Full name in English | `"Dr. T. M. Sreedharan"` |
| `qualifications` | Degrees, separated by commas | `"MBBS, M.D. Paediatrics (JIPMER)"` |
| `designation` | Role or title | `"Registrar"`, `"Senior Consultant"` |
| `specialty` | Medical specialty | `"Paediatrics"`, `"General Physician"` |
| `clinic_hospital` | Clinic or hospital name | `"CHC, Nemmara"` |
| `address` | Full address | `"123 MG Road, Thrissur, Kerala"` |
| `contact` | Phone, mobile, email | `"8886993168"` |
| `registration_number` | Medical council reg. no. | `"52547"` |
| `timing` | Consultation hours | `"Mon–Sat 10:00–14:00"` |

Set to `null` if not present or not clearly readable.

---

### `patient`

| Field | Description | Example |
|---|---|---|
| `name` | Full name (transliterated to English if needed) | `"Ashvika"` |
| `age` | Age as written or decoded | `"4 years"`, `"32 years"` |
| `gender` | Expanded: Male or Female | `"Female"` (decoded from `F`) |
| `date` | Date of prescription | `"20-09-2022"` |
| `patient_id` | Hospital/clinic patient ID | `"PT-00123"` |
| `contact` | Phone number | `"9876543210"` |
| `address` | Patient address | `"Palakkad, Kerala"` |

---

### `clinical`

| Field | Description |
|---|---|
| `chief_complaints` | List of symptoms/complaints (decoded from `c/o` shorthand) |
| `history` | Past medical history (decoded from `h/o`, `k/c/o`) |
| `diagnosis` | Working or confirmed diagnoses |
| `vitals` | Key-value map of measured values |

Vitals example:
```json
{
  "BP": "120/80 mmHg",
  "Weight": "65 kg",
  "Temperature": "99.2°F",
  "SPO2": "98%"
}
```

---

### `medications`

Array of all prescribed medicines. Each object:

| Field | Description | Example |
|---|---|---|
| `name` | Drug name — generic or brand | `"SYP CALPOL (250/5)"` |
| `type` | Dosage form | `"Syrup"`, `"Tablet"`, `"Injection"` |
| `dose` | Strength or volume | `"4 mL"`, `"500 mg"` |
| `frequency` | Expanded timing | `"Twice Daily"`, `"Three Times Daily"` |
| `duration` | Course length | `"3 days"`, `"2 weeks"` |
| `instructions` | Special notes | `"After meals"`, `"With warm water"` |

---

### `investigations`
Array of tests ordered (blood tests, X-rays, ECG, urine culture, etc.).
Example: `["CBC", "Chest X-ray", "Urine routine"]`

---

### `advice`
Array of lifestyle, dietary, or care advice given.
Example: `["Rest for 3 days", "Avoid cold food", "Steam inhalation twice daily"]`

---

### `follow_up`
Next appointment or review instruction.
Example: `"Review after 5 days if no improvement"`

---

### `summary`
4–5 sentence clinical narrative in plain English covering:
- Who the patient is (name, age, gender)
- What condition is being treated
- Each medication prescribed with dosage and purpose
- Key advice and follow-up instructions

---

### `confidence_notes`
Anything that was unclear, partially legible, or ambiguous. Tells the user what to verify against the original prescription.
Example: `"Doctor name partially legible — transliterated from Malayalam script. Medication #3 dose unclear."`

---

### `raw_transcription`
Verbatim copy of visible text exactly as printed/written in the original script(s). No translation, no inference, no additions. Only characters that are physically visible in the image.

---

## Abbreviation Reference

### Patient Patterns
| Written | Meaning |
|---|---|
| `M` / `F` | Male / Female |
| `18/M`, `18Y/M`, `18Yr/M` | 18-year-old Male |
| `Y/O` | Years Old |
| `Pt` | Patient |

### Frequency
| Written | Meaning |
|---|---|
| `OD` | Once Daily |
| `BD` / `BID` | Twice Daily |
| `TDS` / `TID` | Three Times Daily |
| `QID` | Four Times Daily |
| `HS` | At Bedtime |
| `AC` | Before Meals |
| `PC` | After Meals |
| `SOS` / `PRN` | As Needed |
| `Stat` | Immediately |
| `ON` | Every Night |
| `Q6H` | Every 6 Hours |
| `Q8H` | Every 8 Hours |

### Duration
| Written | Meaning |
|---|---|
| `×5D` / `x 5 days` | For 5 Days |
| `×2W` | For 2 Weeks |
| `×1M` | For 1 Month |
| `D` / `W` / `M` | Days / Weeks / Months |

### Dosage Forms
| Written | Meaning |
|---|---|
| `Tab` | Tablet |
| `Cap` | Capsule |
| `Syp` / `Syr` | Syrup |
| `Inj` | Injection |
| `Oint` / `Ung` | Ointment |
| `Susp` | Suspension |
| `Gt` / `Gtt` | Drops |
| `Sachet` | Sachet |

### Clinical Shorthands
| Written | Meaning |
|---|---|
| `Rx` | Prescription |
| `c/o` | Complains of |
| `k/c/o` | Known case of |
| `h/o` | History of |
| `O/E` | On Examination |
| `D/D` | Differential Diagnosis |
| `B/P` | Blood Pressure |
| `PR` | Pulse Rate |
| `SPO2` | Oxygen Saturation |
| `Wt` | Weight |
| `Ht` | Height |
| `T` | Temperature |
| `URTI` | Upper Respiratory Tract Infection |
| `RS` | Respiratory System |
| `AEE` | Air Entry Equal |

### Qualifications
`MBBS` · `MD` · `MS` · `DNB` · `DM` · `MCh` · `DGO` · `DCH` · `FRCS` · `BDS` · `MDS` · `BAMS` · `BHMS` · `BPT` · `MPT`

---

## Saved Output Structure

```
output/
├── general/                              ← General mode
│   └── YYYY-MM-DD_HHMMSS_Doctor_Patient/
│       ├── prescription.(jpg|png|pdf)
│       ├── prescription.json
│       └── prescription.md
├── patients/                             ← Patient Wise mode
│   └── Patient_Name/
│       └── YYYY-MM-DD_HHMMSS/
│           ├── prescription.(jpg|png|pdf)
│           ├── prescription.json
│           └── prescription.md
└── doctors/                              ← Doctor Wise mode
    └── Doctor_Name/
        └── YYYY-MM-DD_HHMMSS/
            ├── prescription.(jpg|png|pdf)
            ├── prescription.json
            └── prescription.md
```

The folder name is automatically built from the date, time, and extracted doctor/patient names. Spaces and special characters are replaced with underscores.

---

## Customizing the Prompt

The system prompt is loaded from `prompts/default_prompt.txt` at startup — not hardcoded in Python. To change extraction behavior:

1. Edit `prompts/default_prompt.txt`
2. Restart the app (`python index.py`)

You can safely add abbreviations, change field instructions, adjust the output format description, or add domain-specific rules without touching any Python code.
