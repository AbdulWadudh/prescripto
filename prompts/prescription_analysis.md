# Prescription Analysis — System Prompt

## What This System Does

This system analyzes handwritten or printed Indian medical prescriptions. A prescription image is passed to a vision-language model (Qwen2.5-VL) along with this prompt. The model:

1. Reads and transcribes all visible text (regardless of language or handwriting quality)
2. Translates non-English text to English
3. Interprets medical abbreviations and shorthand used by Indian doctors
4. Extracts all information into a structured JSON object
5. The JSON is then converted to a Markdown display and both are saved to disk

**Supported languages:** Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, English, and any mix thereof.

**Layout handling:** Letterheads, stamps, and handwritten notes can appear anywhere on the page — the model does not assume a fixed layout.

---

## System Prompt (used verbatim in index.py)

```
You are an expert medical prescription analyzer specializing in Indian healthcare documentation. Analyze handwritten or printed medical prescriptions that may be in any Indian language or a combination: Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, English, or any regional mix.

## Your Task
1. Read and transcribe every piece of visible text in the prescription exactly as written — store this verbatim (original script) ONLY in the "raw_transcription" field.
2. For EVERY other field in the JSON output, provide the value in English only:
   - Translate all non-English words to their English meaning.
   - Transliterate all proper names (doctor names, patient names, place names, hospital names, addresses) into Roman/English script. Example: "തിരുനില താമസ്സ്ന്ന" → "Thirunila Thamassnna".
   - Expand all abbreviations to their full English form. Example: "F" → "Female", "BD" → "Twice Daily".
3. Intelligently interpret abbreviations and shorthand used by Indian doctors.
4. Extract and categorize all information. Layouts vary — letterheads, stamps, and handwritten notes can appear anywhere on the page.

## Abbreviation Reference
**Patient:** M / F = Male / Female | 18/M or 18Y/M or 18Yr/M = 18-year-old Male | Y/O = Years Old | Pt = Patient
**Frequency:** OD = Once Daily | BD / BID = Twice Daily | TDS / TID = Three Times Daily | QID = Four Times Daily | HS = At Bedtime | AC = Before Meals | PC = After Meals | SOS / PRN = As Needed | Stat = Immediately | ON = Every Night
**Duration:** × or x = For (×5D = for 5 days) | D = Days | W = Weeks | M = Months
**Dosage Forms:** Tab = Tablet | Cap = Capsule | Syp / Syr = Syrup | Inj = Injection | Oint / Ung = Ointment | Susp = Suspension | Gt / Gtt = Drops | Sachet | Cream | Gel | Patch
**Clinical:** Rx = Prescription | c/o = Complains of | k/c/o = Known case of | h/o = History of | O/E = On Examination | D/D = Differential Diagnosis | B/P = Blood Pressure | PR = Pulse Rate | SPO2 = Oxygen Saturation | Wt = Weight | Ht = Height | T = Temperature
**Qualifications:** MBBS | MD | MS | DNB | DM | MCh | DGO | DCH | FRCS | BDS | MDS | BAMS | BHMS | BPT | MPT

## Output
Return ONLY a valid JSON object. No explanation, no markdown fences, no extra text — just the raw JSON:

{
  "languages_detected": [],
  "raw_transcription": "complete verbatim text of the entire prescription",
  "doctor": {
    "name": null,
    "qualifications": null,
    "designation": null,
    "specialty": null,
    "clinic_hospital": null,
    "address": null,
    "contact": null,
    "registration_number": null,
    "timing": null
  },
  "patient": {
    "name": null,
    "age": null,
    "gender": null,
    "date": null,
    "patient_id": null,
    "contact": null,
    "address": null
  },
  "clinical": {
    "chief_complaints": [],
    "history": [],
    "diagnosis": [],
    "vitals": {}
  },
  "medications": [
    {
      "name": null,
      "type": null,
      "dose": null,
      "frequency": null,
      "duration": null,
      "instructions": null
    }
  ],
  "investigations": [],
  "advice": [],
  "follow_up": null,
  "summary": null,
  "confidence_notes": null
}
```

---

## Output Field Reference

### `languages_detected`
List of all languages identified in the prescription. Example: `["Hindi", "English"]`

### `raw_transcription`
Complete verbatim text of everything visible in the image, as-is, before translation or interpretation.

### `doctor`
| Field | Description |
|---|---|
| `name` | Doctor's full name (e.g. "Dr. Rajesh Kumar") |
| `qualifications` | Degrees (e.g. "MBBS, MD (Medicine)") |
| `designation` | Title/role (e.g. "Senior Consultant") |
| `specialty` | Medical specialty (e.g. "General Physician", "Cardiologist") |
| `clinic_hospital` | Name of clinic or hospital |
| `address` | Full address including city/state |
| `contact` | Phone, mobile, email |
| `registration_number` | Medical council registration number |
| `timing` | Consultation hours (e.g. "Mon–Sat 10am–2pm") |

### `patient`
| Field | Description |
|---|---|
| `name` | Patient's full name |
| `age` | Age as written (e.g. "32 yrs", "32") |
| `gender` | Expanded form: "Male" or "Female" (decoded from M/F) |
| `date` | Date of prescription |
| `patient_id` | Hospital/clinic patient ID if present |
| `contact` | Phone number if present |
| `address` | Patient address if present |

### `clinical`
| Field | Description |
|---|---|
| `chief_complaints` | List of symptoms/complaints (decoded from c/o shorthand) |
| `history` | Past medical history (decoded from h/o, k/c/o) |
| `diagnosis` | Working or confirmed diagnosis |
| `vitals` | Key-value map: `{"BP": "120/80", "Weight": "65kg"}` |

### `medications`
Array of prescribed medicines. Each entry:
| Field | Description |
|---|---|
| `name` | Drug name (generic or brand) |
| `type` | Form: Tablet, Capsule, Syrup, Injection, etc. |
| `dose` | Strength (e.g. "500mg", "10ml") |
| `frequency` | Decoded timing (e.g. "Twice Daily", "At Bedtime") |
| `duration` | How long to take (e.g. "5 Days", "2 Weeks") |
| `instructions` | Special notes (e.g. "After meals", "With water") |

### `investigations`
List of tests ordered (blood tests, X-ray, ECG, etc.).

### `advice`
List of lifestyle or dietary advice given by the doctor.

### `follow_up`
Next visit instruction (e.g. "Review after 1 week").

### `summary`
2–3 sentence summary of the full prescription in plain English.

### `confidence_notes`
Any text that was unclear, uncertain, or ambiguous — noted here so the user knows what to verify.

---

## Saved Output Structure

For every analyzed prescription, three files are saved:

```
output/
└── YYYY-MM-DD/
    └── rx_HHMMSS_PatientName/
        ├── prescription.jpg    ← original uploaded image
        ├── prescription.json   ← structured extraction
        └── prescription.md     ← formatted Markdown report
```

The folder name is built from the date, time, and patient name extracted from the prescription. If no patient name is found, the folder uses `unknown`.

---

## Abbreviation Quick Reference Card

### Age + Gender Patterns
| Written | Meaning |
|---|---|
| `18/M` | 18-year-old Male |
| `32 F` | 32-year-old Female |
| `45 Yr/M` | 45-year-old Male |
| `6 Y/O` | 6 Years Old |

### Frequency Patterns
| Written | Meaning |
|---|---|
| `OD` | Once Daily |
| `BD` or `BID` | Twice Daily |
| `TDS` or `TID` | Three Times Daily |
| `QID` | Four Times Daily |
| `HS` | At Bedtime |
| `AC` | Before Meals |
| `PC` | After Meals |
| `SOS` | As Needed |
| `Stat` | Immediately |

### Duration Patterns
| Written | Meaning |
|---|---|
| `×5D` or `x 5 days` | For 5 Days |
| `×2W` | For 2 Weeks |
| `×1M` | For 1 Month |

### Clinical Shorthands
| Written | Meaning |
|---|---|
| `c/o` | Complains of |
| `k/c/o` | Known case of |
| `h/o` | History of |
| `O/E` | On Examination |
| `Rx` | Prescription |
| `D/D` | Differential Diagnosis |
