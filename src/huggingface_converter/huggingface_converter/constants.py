from rdflib import Namespace
from rdflib.namespace import DCAT, XSD

# -------------------------------------------------------------------
# RDF-Namespaces
# -------------------------------------------------------------------
SCHEMA = Namespace("https://schema.org/")
DCATAP = Namespace("http://data.europa.eu/r5r/")
DCATDE= Namespace("http://dcat-ap.de/def/dcatde/3.0.0#")
ADMS = Namespace("http://www.w3.org/ns/adms#")
MLS = Namespace("http://www.w3.org/ns/mls#")
VCARD = Namespace("http://www.w3.org/2006/vcard/ns#")
IT6 = Namespace("http://data.europa.eu/it6/")
LPWCC = Namespace("https://linkedpaperswithcode.com/class/")

# -------------------------------------------------------------------
# DCATAP and DCATAP DE controlled vovabularies 
# -------------------------------------------------------------------
DCATAP_CONTROLLED_VOCABULARY =  {
    "language": "http://publications.europa.eu/resource/authority/language",
    "license": "http://publications.europa.eu/resource/authority/licence",
    "frequency": "http://publications.europa.eu/resource/authority/frequency",
    "theme": "http://publications.europa.eu/resource/authority/data-theme",
    "file_type": "http://publications.europa.eu/resource/authority/file-type",
    "media_type": "http://www.iana.org/assignments/media-types",
    "access_rights": "http://publications.europa.eu/resource/authority/access-right",
    "dataset_type": "http://publications.europa.eu/resource/authority/dataset-type",
    "availability": "http://publications.europa.eu/resource/authority/planned-availability", 
    "theme": "http://publications.europa.eu/resource/authority/data-theme", 
    "accrual_periodicity": "http://publications.europa.eu/resource/authority/frequency", 
    "spatial_continent": "http://publications.europa.eu/resource/authority/continent",
    "spatial_country": "http://publications.europa.eu/resource/authority/country", 
    # "dataset_type": "http://publications.europa.eu/resource/authority/dataset-type"
}

DCATAP_DE_CONTROLLED_VOCABULARY = {
    "language": "http://publications.europa.eu/resource/authority/language",  
    "license": "http://dcat-ap.de/def/licenses",
    "frequency": "http://publications.europa.eu/resource/authority/frequency",
    "theme": "http://publications.europa.eu/resource/authority/data-theme",
    "file_type": "http://publications.europa.eu/resource/authority/file-type",
    "media_type": "http://www.iana.org/assignments/media-types",
    "access_rights": "http://publications.europa.eu/resource/authority/access-right",
    "dataset_type": "http://publications.europa.eu/resource/authority/dataset-type",
    "contributors": "http://dcat-ap.de/def/contributors",
    "geocoding_level": "http://dcat-ap.de/def/politicalGeocoding/Level",
    "geocoding_region": "http://dcat-ap.de/def/politicalGeocoding/regionalKey",
    "availability": "http://publications.europa.eu/resource/authority/planned-availability",
    "theme": "http://publications.europa.eu/resource/authority/data-theme", 
    "accrual_periodicity": "http://publications.europa.eu/resource/authority/frequency", 
    "spatial_continent": "http://publications.europa.eu/resource/authority/continent",
    "spatial_country": "http://publications.europa.eu/resource/authority/country", 
    #  "dataset_type": "http://publications.europa.eu/resource/authority/dataset-type"
}

# -------------------------------------------------------------------
# Others
# -------------------------------------------------------------------
RESOURCE_CONFIG = {
    "dataset": (DCAT.Dataset, "dataset"),
    "model": (DCAT.Dataset, "model"),
}
METRICS = {
    "likes": (SCHEMA.LikeAction, XSD.integer),
    "downloads": (SCHEMA.DownloadAction, XSD.integer),
    "downloads_all_time": (SCHEMA.DownloadAction, XSD.integer),
    "trending_score": (SCHEMA.InteractionCounter, XSD.float)  
}
# Map DE and EN mainly
LANG_CODE_MAPPINGS = {
    "en-us": "ENG",  
    "de-de": "DEU", 
    # Official EU languages
    "bg": "BGR",  # Bulgarian
    "cs": "CES",  # Czech
    "da": "DAN",  # Danish
    "de": "DEU",  # German
    "el": "ELL",  # Greek
    "en": "ENG",  # English
    "es": "SPA",  # Spanish
    "et": "EST",  # Estonian
    "fi": "FIN",  # Finnish
    "fr": "FRA",  # French
    "ga": "GLE",  # Irish
    "hr": "HRV",  # Croatian
    "hu": "HUN",  # Hungarian
    "it": "ITA",  # Italian
    "lt": "LIT",  # Lithuanian
    "lv": "LAV",  # Latvian
    "mt": "MLT",  # Maltese
    "nl": "NLD",  # Dutch
    "pl": "POL",  # Polish
    "pt": "POR",  # Portuguese
    "ro": "RON",  # Romanian
    "sk": "SLK",  # Slovak
    "sl": "SLV",  # Slovenian
    "sv": "SWE",  # Swedish
    
    # Other common European languages
    "is": "ISL",  # Icelandic
    "no": "NOR",  # Norwegian
    "mk": "MKD",  # Macedonian
    "sq": "SQI",  # Albanian
    "sr": "SRP",  # Serbian
    "tr": "TUR",  # Turkish
    "uk": "UKR",  # Ukrainian
    
    # Additional global languages
    "ar": "ARA",  # Arabic
    "zh": "ZHO",  # Chinese
    "ja": "JPN",  # Japanese
    "ru": "RUS",  # Russian
    "hi": "HIN",  # Hindi
    "ko": "KOR",
    "th": "THA", 
}
LANG_LABELS = {
    # Official EU languages
    "BGR": "Bulgarian",
    "CES": "Czech",
    "DAN": "Danish",
    "DEU": "German",
    "ELL": "Greek",
    "ENG": "English",
    "SPA": "Spanish",
    "EST": "Estonian",
    "FIN": "Finnish",
    "FRA": "French",
    "GLE": "Irish",
    "HRV": "Croatian",
    "HUN": "Hungarian",
    "ITA": "Italian",
    "LIT": "Lithuanian",
    "LAV": "Latvian",
    "MLT": "Maltese",
    "NLD": "Dutch",
    "POL": "Polish",
    "POR": "Portuguese",
    "RON": "Romanian",
    "SLK": "Slovak",
    "SLV": "Slovenian",
    "SWE": "Swedish",

    # Other common European languages
    "ISL": "Icelandic",
    "NOR": "Norwegian",
    "MKD": "Macedonian",
    "SQI": "Albanian",
    "SRP": "Serbian",
    "TUR": "Turkish",
    "UKR": "Ukrainian",

    # Additional global languages
    "ARA": "Arabic",
    "ZHO": "Chinese",
    "JPN": "Japanese",
    "RUS": "Russian",
    "HIN": "Hindi"
}
LANG_LABELS_MULTI = {
    "BGR": {"en": "Bulgarian", "de": "Bulgarisch"},
    "CES": {"en": "Czech", "de": "Tschechisch"},
    "DAN": {"en": "Danish", "de": "Dänisch"},
    "DEU": {"en": "German", "de": "Deutsch"},
    "ELL": {"en": "Greek", "de": "Griechisch"},
    "ENG": {"en": "English", "de": "Englisch"},
    "SPA": {"en": "Spanish", "de": "Spanisch"},
    "EST": {"en": "Estonian", "de": "Estnisch"},
    "FIN": {"en": "Finnish", "de": "Finnisch"},
    "FRA": {"en": "French", "de": "Französisch"},
    "GLE": {"en": "Irish", "de": "Irisch"},
    "HRV": {"en": "Croatian", "de": "Kroatisch"},
    "HUN": {"en": "Hungarian", "de": "Ungarisch"},
    "ITA": {"en": "Italian", "de": "Italienisch"},
    "LIT": {"en": "Lithuanian", "de": "Litauisch"},
    "LAV": {"en": "Latvian", "de": "Lettisch"},
    "MLT": {"en": "Maltese", "de": "Maltesisch"},
    "NLD": {"en": "Dutch", "de": "Niederländisch"},
    "POL": {"en": "Polish", "de": "Polnisch"},
    "POR": {"en": "Portuguese", "de": "Portugiesisch"},
    "RON": {"en": "Romanian", "de": "Rumänisch"},
    "SLK": {"en": "Slovak", "de": "Slowakisch"},
    "SLV": {"en": "Slovenian", "de": "Slowenisch"},
    "SWE": {"en": "Swedish", "de": "Schwedisch"},

    "ISL": {"en": "Icelandic", "de": "Isländisch"},
    "NOR": {"en": "Norwegian", "de": "Norwegisch"},
    "MKD": {"en": "Macedonian", "de": "Mazedonisch"},
    "SQI": {"en": "Albanian", "de": "Albanisch"},
    "SRP": {"en": "Serbian", "de": "Serbisch"},
    "TUR": {"en": "Turkish", "de": "Türkisch"},
    "UKR": {"en": "Ukrainian", "de": "Ukrainisch"},

    "ARA": {"en": "Arabic", "de": "Arabisch"},
    "ZHO": {"en": "Chinese", "de": "Chinesisch"},
    "JPN": {"en": "Japanese", "de": "Japanisch"},
    "RUS": {"en": "Russian", "de": "Russisch"},
    "HIN": {"en": "Hindi", "de": "Hindi"}
}
