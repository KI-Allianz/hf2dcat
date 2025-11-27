from enum import Enum

class Profile(str, Enum):
    DCAT_AP = "dcat-ap"
    DCAT_AP_DE = "dcat-ap-de"

class OutputFormat(Enum):
    JSONLD = "json-ld"
    TURTLE = "turtle"
    RDFXML = "xml"
    NTRIPLES = "nt"
