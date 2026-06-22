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
LPWC = Namespace("https://linkedpaperswithcode.com/property/")
LPWCC = Namespace("https://linkedpaperswithcode.com/class/")
MLSO = Namespace("http://w3id.org/mlso/")
CR = Namespace("http://mlcommons.org/croissant/")

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

# ML_TASK_TYPES_TTL = "https://raw.githubusercontent.com/dtai-kg/MLSO/main/ontology/Taxonomies/mlso_ml_task_types_v2.ttl"
# ML_ALGORITHMS_TTL = "https://raw.githubusercontent.com/dtai-kg/MLSO/refs/heads/main/ontology/Taxonomies/mlso_ml_algorithms.ttl"

# HF Task from : https://huggingface.co/tasks
HF_TASKS: dict[str, dict[str, str]] = {

    # =========================
    # Multimodal
    # =========================
    "any-to-any": {
        "label": "Any-to-Any",
        "category": "multimodal",
    },
    "audio-text-to-text": {
        "label": "Audio-Text-to-Text",
        "category": "multimodal",
    },
    "document-question-answering": {
        "label": "Document Question Answering",
        "category": "multimodal",
    },
    "visual-document-retrieval": {
        "label": "Visual Document Retrieval",
        "category": "multimodal",
    },
    "image-text-to-text": {
        "label": "Image-Text-to-Text",
        "category": "multimodal",
    },
    "image-text-to-image": {
        "label": "Image-Text-to-Image",
        "category": "multimodal",
    },
    "image-text-to-video": {
        "label": "Image-Text-to-Video",
        "category": "multimodal",
    },
    "video-text-to-text": {
        "label": "Video-Text-to-Text",
        "category": "multimodal",
    },
    "visual-question-answering": {
        "label": "Visual Question Answering",
        "category": "multimodal",
    },
    # =========================
    # Natural Language Processing
    # =========================
    "feature-extraction": {
        "label": "Feature Extraction",
        "category": "natural-language-processing",
    },
    "fill-mask": {
        "label": "Fill-Mask",
        "category": "natural-language-processing",
    },
    "question-answering": {
        "label": "Question Answering",
        "category": "natural-language-processing",
    },
    "sentence-similarity": {
        "label": "Sentence Similarity",
        "category": "natural-language-processing",
    },
    "summarization": {
        "label": "Summarization",
        "category": "natural-language-processing",
    },
    "table-question-answering": {
        "label": "Table Question Answering",
        "category": "natural-language-processing",
    },
    "text-classification": {
        "label": "Text Classification",
        "category": "natural-language-processing",
    },
    "text-generation": {
        "label": "Text Generation",
        "category": "natural-language-processing",
    },
    "text-ranking": {
        "label": "Text Ranking",
        "category": "natural-language-processing",
    },
    "token-classification": {
        "label": "Token Classification",
        "category": "natural-language-processing",
    },
    "translation": {
        "label": "Translation",
        "category": "natural-language-processing",
    },
    "zero-shot-classification": {
        "label": "Zero-Shot Classification",
        "category": "natural-language-processing",
    },

    # =========================
    # Computer Vision
    # =========================
    "depth-estimation": {
        "label": "Depth Estimation",
        "category": "computer-vision",
    },
    "image-classification": {
        "label": "Image Classification",
        "category": "computer-vision",
    },
    "image-feature-extraction": {
        "label": "Image Feature Extraction",
        "category": "computer-vision",
    },
    "image-segmentation": {
        "label": "Image Segmentation",
        "category": "computer-vision",
    },
    "image-to-image": {
        "label": "Image-to-Image",
        "category": "computer-vision",
    },
    "image-to-text": {
        "label": "Image-to-Text",
        "category": "computer-vision",
    },
    "image-to-video": {
        "label": "Image-to-Video",
        "category": "computer-vision",
    },
    "keypoint-detection": {
        "label": "Keypoint Detection",
        "category": "computer-vision",
    },
    "mask-generation": {
        "label": "Mask Generation",
        "category": "computer-vision",
    },
    "object-detection": {
        "label": "Object Detection",
        "category": "computer-vision",
    },
    "video-classification": {
        "label": "Video Classification",
        "category": "computer-vision",
    },
    "text-to-image": {
        "label": "Text-to-Image",
        "category": "computer-vision",
    },
    "text-to-video": {
        "label": "Text-to-Video",
        "category": "computer-vision",
    },
    "unconditional-image-generation": {
        "label": "Unconditional Image Generation",
        "category": "computer-vision",
    },
    "video-to-video": {
        "label": "Video-to-Video",
        "category": "computer-vision",
    },
    "zero-shot-image-classification": {
        "label": "Zero-Shot Image Classification",
        "category": "computer-vision",
    },
    "zero-shot-object-detection": {
        "label": "Zero-Shot Object Detection",
        "category": "computer-vision",
    },
    "text-to-3d": {
        "label": "Text-to-3D",
        "category": "computer-vision",
    },
    "image-to-3d": {
        "label": "Image-to-3D",
        "category": "computer-vision",
    },

    # =========================
    # Audio
    # =========================
    "audio-classification": {
        "label": "Audio Classification",
        "category": "audio",
    },
    "audio-to-audio": {
        "label": "Audio-to-Audio",
        "category": "audio",
    },
    "automatic-speech-recognition": {
        "label": "Automatic Speech Recognition",
        "category": "audio",
    },
    "text-to-speech": {
        "label": "Text-to-Speech",
        "category": "audio",
    },

    # =========================
    # Tabular
    # =========================
    "tabular-classification": {
        "label": "Tabular Classification",
        "category": "tabular",
    },
    "tabular-regression": {
        "label": "Tabular Regression",
        "category": "tabular",
    },

    # =========================
    # Reinforcement Learning
    # =========================
    "reinforcement-learning": {
        "label": "Reinforcement Learning",
        "category": "reinforcement-learning",
    },
}
