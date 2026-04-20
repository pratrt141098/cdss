# modules/ner.py
import spacy
from dataclasses import dataclass, field
from modules.base import BaseModule


@dataclass
class AnnotatedChunk:
    chunk_id:   str
    hadm_id:    int
    patient_id: int
    section:    str
    text:       str
    entities:   dict = field(default_factory=lambda: {
        "medications": [],
        "diseases":    [],
        "anatomy":     [],
        "procedures":  [],
    })


MEDICATION_SECTIONS = {"medications_on_admission", "discharge_medications", "allergies"}
DISEASE_SECTIONS    = {"history_of_present_illness", "discharge_diagnosis", "past_medical_history",
                       "chief_complaint", "family_history"}
ANATOMY_SECTIONS    = {"physical_exam", "review_of_systems"}
PROCEDURE_SECTIONS  = {"pertinent_results", "discharge_condition", "discharge_instructions"}

EMPTY_ENTITIES = lambda: {"medications": [], "diseases": [], "anatomy": [], "procedures": []}


class NERModule(BaseModule):
    """
    Uses en_core_sci_sm to detect clinical entity spans.
    sm/md/lg core models output flat ENTITY labels — entities are
    bucketed by section name as a heuristic. Compatible with typed
    labels from bc5cdr/bionlp models if swapped in later.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        model_name = self.config.get("spacy_model", "en_core_sci_sm")
        self.nlp = spacy.load(model_name, exclude=["parser", "tagger", "lemmatizer"])
        print(f"[{self.name}] Loaded: {model_name}")

    def process(self, input_data: list) -> list[AnnotatedChunk]:
        texts     = [chunk.text for chunk in input_data]
        annotated = []

        for i, (chunk, doc) in enumerate(
            zip(input_data, self.nlp.pipe(texts, batch_size=16))
        ):
            # skip header-only chunks
            if len(chunk.text.strip()) < 20:
                annotated.append(AnnotatedChunk(
                    chunk_id=chunk.chunk_id,
                    hadm_id=chunk.hadm_id,
                    patient_id=chunk.patient_id,
                    section=chunk.section,
                    text=chunk.text,
                    entities=EMPTY_ENTITIES(),
                ))
                continue

            entities = self._extract_entities(doc, chunk.section)
            annotated.append(AnnotatedChunk(
                chunk_id=chunk.chunk_id,
                hadm_id=chunk.hadm_id,
                patient_id=chunk.patient_id,
                section=chunk.section,
                text=chunk.text,
                entities=entities,
            ))

            if i % 25 == 0:
                print(f"[{self.name}] Annotated {i}/{len(input_data)} chunks")

        print(f"[{self.name}] Done — {len(annotated)} annotated chunks")
        return annotated

    def _extract_entities(self, doc, section: str) -> dict:
        entities      = EMPTY_ENTITIES()
        CLINICAL_NOISE = {
            "no known", "allergies", "adverse drug reactions", "attending",
            "addendum", "none", "none known", "nkda", "n/a", "unknown",
            "see above", "see below", "please see", "as above", "as below",
        }
        seen          = set()
        section_lower = section.lower()

        for ent in doc.ents:
            val = ent.text.strip()
            key = val.lower()
            if key in seen or len(val) < 4 or key in CLINICAL_NOISE:
                continue
            seen.add(key)

            label = ent.label_.upper()

            # typed labels — bc5cdr / bionlp models
            if label in ("CHEMICAL", "DRUG"):
                entities["medications"].append(val)
            elif label in ("DISEASE", "DISORDER"):
                entities["diseases"].append(val)
            elif label in ("BODY_PART", "TISSUE", "ORGAN", "CELL_TYPE"):
                entities["anatomy"].append(val)
            elif label in ("PROCEDURE", "TEST"):
                entities["procedures"].append(val)

            # flat ENTITY from sm/md/lg core models — route by section
            elif label == "ENTITY":
                if section_lower in MEDICATION_SECTIONS:
                    entities["medications"].append(val)
                elif section_lower in DISEASE_SECTIONS:
                    entities["diseases"].append(val)
                elif section_lower in ANATOMY_SECTIONS:
                    entities["anatomy"].append(val)
                elif section_lower in PROCEDURE_SECTIONS:
                    entities["procedures"].append(val)
                else:
                    entities["diseases"].append(val)

        return entities