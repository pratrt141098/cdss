# eval/query_set.py
#
# 20 clinical queries grounded in the 10 held-out patients (last 10 subject_ids
# from the cdss_147 collection, sorted ascending). Two queries per patient cover
# different clinical sections so the eval spans medications, diagnoses,
# allergies, and history of present illness.
#
# "Held-out" means these patients' chunks ARE in cdss_147 (we do not re-ingest),
# but they are reserved exclusively as the eval ground-truth set and are never
# used to build or fine-tune the retriever. The first 20 patients (by sorted
# subject_id) are treated as the training/ingested cohort.

from dataclasses import dataclass, field

HELD_OUT_SUBJECT_IDS = [
    "10001884", "10001919", "10002013", "10002131", "10002155",
    "10002167", "10002221", "10002348", "10002428", "10002430",
]


@dataclass
class EvalQuery:
    query_id: str
    query: str
    subject_id: str
    hadm_id: str
    expected_section: str                    # section slug matching ChromaDB metadata
    expected_answer_keywords: list[str] = field(default_factory=list)
    notes: str = ""                          # free-text rationale / source excerpt


QUERY_SET: list[EvalQuery] = [
    # ── Patient 10001884 / HADM 21268656 ────────────────────────────────────────
    EvalQuery(
        query_id="Q01",
        query="What is the discharge diagnosis for this patient?",
        subject_id="10001884",
        hadm_id="21268656",
        expected_section="discharge_diagnosis",
        expected_answer_keywords=["coronary artery disease"],
        notes="Discharge diagnosis is Coronary Artery Disease.",
    ),
    EvalQuery(
        query_id="Q02",
        query="What inhalers or respiratory medications were prescribed at discharge?",
        subject_id="10001884",
        hadm_id="21268656",
        expected_section="discharge_medications",
        expected_answer_keywords=["albuterol", "fluticasone", "salmeterol"],
        notes="Discharge meds include albuterol sulfate and fluticasone-salmeterol.",
    ),

    # ── Patient 10001919 / HADM 29897682 ────────────────────────────────────────
    EvalQuery(
        query_id="Q03",
        query="What is this patient's primary diagnosis?",
        subject_id="10001919",
        hadm_id="29897682",
        expected_section="discharge_diagnosis",
        expected_answer_keywords=["gastric cancer", "metastatic", "stage IV"],
        notes="Primary dx: Metastatic gastric cancer stage IV (T3N2M1).",
    ),
    EvalQuery(
        query_id="Q04",
        query="What medications was this patient taking on admission?",
        subject_id="10001919",
        hadm_id="29897682",
        expected_section="medications_on_admission",
        expected_answer_keywords=["enoxaparin", "omeprazole"],
        notes="Admission meds include enoxaparin 120 mg SC daily and omeprazole.",
    ),

    # ── Patient 10002013 / HADM 21763296 ────────────────────────────────────────
    EvalQuery(
        query_id="Q05",
        query="Does this patient have any documented drug allergies?",
        subject_id="10002013",
        hadm_id="21763296",
        expected_section="allergies",
        expected_answer_keywords=["lisinopril"],
        notes="Patient is allergic to lisinopril.",
    ),
    EvalQuery(
        query_id="Q06",
        query="What antibiotics were prescribed at discharge?",
        subject_id="10002013",
        hadm_id="21763296",
        expected_section="discharge_medications",
        expected_answer_keywords=["ciprofloxacin", "clindamycin"],
        notes="Discharge antibiotics: ciprofloxacin 500 mg Q12H and clindamycin 300 mg Q6H.",
    ),

    # ── Patient 10002131 / HADM 24065018 ────────────────────────────────────────
    EvalQuery(
        query_id="Q07",
        query="What allergies are documented for this patient?",
        subject_id="10002131",
        hadm_id="24065018",
        expected_section="allergies",
        expected_answer_keywords=["penicillin", "dilantin", "zofran"],
        notes="Allergies: Penicillins, Dilantin Kapseal, Zofran.",
    ),
    EvalQuery(
        query_id="Q08",
        query="What is the primary discharge diagnosis for this patient?",
        subject_id="10002131",
        hadm_id="24065018",
        expected_section="discharge_diagnosis",
        expected_answer_keywords=["deep vein thrombosis", "congestive heart failure", "alzheimer"],
        notes="Primary dx: DVT. Secondary: CHF, Afib, Alzheimer's dementia.",
    ),

    # ── Patient 10002155 / HADM 20345487 ────────────────────────────────────────
    EvalQuery(
        query_id="Q09",
        query="What are this patient's documented drug allergies?",
        subject_id="10002155",
        hadm_id="20345487",
        expected_section="allergies",
        expected_answer_keywords=["codeine"],
        notes="Patient is allergic to codeine.",
    ),
    EvalQuery(
        query_id="Q10",
        query="What is the history of present illness for this patient?",
        subject_id="10002155",
        hadm_id="20345487",
        expected_section="history_of_present_illness",
        expected_answer_keywords=["lung cancer", "melena", "dyspnea", "cough"],
        notes="Stage IV NSCLC presenting with dyspnea, productive cough, and melena.",
    ),

    # ── Patient 10002167 / HADM 24023396 ────────────────────────────────────────
    EvalQuery(
        query_id="Q11",
        query="What is the discharge diagnosis for this patient?",
        subject_id="10002167",
        hadm_id="24023396",
        expected_section="discharge_diagnosis",
        expected_answer_keywords=["nausea", "vomiting", "band"],
        notes="Dx: nausea and vomiting due to tight lap band.",
    ),
    EvalQuery(
        query_id="Q12",
        query="What medications were prescribed at discharge?",
        subject_id="10002167",
        hadm_id="24023396",
        expected_section="discharge_medications",
        expected_answer_keywords=["lorazepam", "buspirone"],
        notes="Discharge meds: lorazepam 0.5 mg BID PRN anxiety; buspirone 5 mg TID.",
    ),

    # ── Patient 10002221 / HADM 20195471 ────────────────────────────────────────
    EvalQuery(
        query_id="Q13",
        query="What are this patient's drug allergies?",
        subject_id="10002221",
        hadm_id="20195471",
        expected_section="allergies",
        expected_answer_keywords=["codeine", "augmentin", "topamax"],
        notes="Allergies: Codeine, Augmentin, Topamax.",
    ),
    EvalQuery(
        query_id="Q14",
        query="Why was this patient admitted and what procedure did they undergo?",
        subject_id="10002221",
        hadm_id="20195471",
        expected_section="history_of_present_illness",
        expected_answer_keywords=["knee", "osteoarthritis", "arthroplasty"],
        notes="Left knee OA/pain, admitted for left total knee arthroplasty.",
    ),

    # ── Patient 10002348 / HADM 22725460 ────────────────────────────────────────
    EvalQuery(
        query_id="Q15",
        query="What medications was this patient taking on admission?",
        subject_id="10002348",
        hadm_id="22725460",
        expected_section="medications_on_admission",
        expected_answer_keywords=["aspirin", "alendronate", "levothyroxine", "lisinopril"],
        notes="Admission meds: ASA 81 mg, alendronate, vitamin D, levothyroxine, lisinopril.",
    ),
    EvalQuery(
        query_id="Q16",
        query="What is the primary diagnosis for this patient?",
        subject_id="10002348",
        hadm_id="22725460",
        expected_section="discharge_diagnosis",
        expected_answer_keywords=["brain tumor"],
        notes="Primary dx: Brain tumor (cerebellar lesion).",
    ),

    # ── Patient 10002428 / HADM 20321825 ────────────────────────────────────────
    EvalQuery(
        query_id="Q17",
        query="What is the history of present illness for this patient?",
        subject_id="10002428",
        hadm_id="20321825",
        expected_section="history_of_present_illness",
        expected_answer_keywords=["sjogren", "c. diff", "respiratory failure", "dyspnea"],
        notes="Sjogren's syndrome, recent sepsis from c. diff, presenting with worsening dyspnea.",
    ),
    EvalQuery(
        query_id="Q18",
        query="What is the discharge diagnosis for this patient?",
        subject_id="10002428",
        hadm_id="20321825",
        expected_section="discharge_diagnosis",
        expected_answer_keywords=["altered mental status", "hypoxia", "clostridium"],
        notes="Dx: Altered mental status, hypoxia, Clostridium difficile.",
    ),

    # ── Patient 10002430 / HADM 24513842 ────────────────────────────────────────
    EvalQuery(
        query_id="Q19",
        query="What are this patient's documented drug allergies?",
        subject_id="10002430",
        hadm_id="24513842",
        expected_section="allergies",
        expected_answer_keywords=["corgard", "vasotec"],
        notes="Allergies: Corgard, Vasotec.",
    ),
    EvalQuery(
        query_id="Q20",
        query="What medications were prescribed at discharge?",
        subject_id="10002430",
        hadm_id="24513842",
        expected_section="discharge_medications",
        expected_answer_keywords=["amiodarone", "apixaban", "aspirin", "losartan", "rosuvastatin"],
        notes="Discharge meds include amiodarone, apixaban, aspirin, losartan, rosuvastatin.",
    ),
]
