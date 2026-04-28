# modules/query_router.py
#
# Detects structured clinical queries that map reliably to a known MIMIC-IV
# section slug. When matched, the caller should fetch chunks directly via
# VectorStoreModule.get_by_section() instead of running a vector search.
#
# Returns a section slug (str) if a route is matched, otherwise None.

import re

# Each entry: (compiled pattern, section slug)
# Patterns are checked in order; first match wins.
_MED = r"(med|drug|rx|antibiotic|prescri)"   # common medication terms

_ROUTES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"allerg",                                          re.I), "allergies"),
    # medications on admission — "admis" matches both "admit" and "admission"
    (re.compile(rf"{_MED}.{{0,60}}admis|admis.{{0,60}}{_MED}",     re.I), "medications_on_admission"),
    # discharge medications — "discharge" near medication term in either order
    (re.compile(rf"discharge.{{0,40}}{_MED}|{_MED}.{{0,40}}discharge", re.I), "discharge_medications"),
    (re.compile(r"discharge.{0,15}diagnos",                         re.I), "discharge_diagnosis"),
    (re.compile(r"primary.{0,15}diagnos",                           re.I), "discharge_diagnosis"),
]


def route_query(query: str) -> str | None:
    """
    Returns the matching section slug if the query maps deterministically
    to a structured MIMIC-IV section, otherwise returns None.
    """
    for pattern, section in _ROUTES:
        if pattern.search(query):
            return section
    return None
