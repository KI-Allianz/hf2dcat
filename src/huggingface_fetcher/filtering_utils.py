from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

NON_OPEN_LICENSES = {
    "apple-amlr", "apple-ascl", 
    "bigcode-openrail-m", "bigscience-bloom-rail-1.0", 
    "bigscience-openrail-m", "c-uda",
    "cc-by-nc-2.0", "cc-by-nc-3.0", "cc-by-nc-4.0",
    "cc-by-nc-nd-3.0", "cc-by-nc-nd-4.0",
    "cc-by-nc-sa-2.0", "cc-by-nc-sa-3.0", "cc-by-nc-sa-4.0",
    "cc-by-nd-4.0",  "creativeml-openrail-m", "deepfloyd-if-license",
    "fair-noncommercial-research-license", 
    "gemma", "h-research", "intel-research", 
    "llama2", "llama3", "llama3.1", "llama3.2",
    "llama3.3", "llama4",  "open-mdw", 
    "openrail", "openrail++", 
    "other", "unknown"
}
OPEN_LICENSES = {
    "afl-3.0", 'agpl-3.0', "apache-2.0", "artistic-2.0",
    "bsd", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear",
    "bsl-1.0", "cc", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "cc-by-sa-3.0", "cc-by-sa-4.0", "cc0-1.0", 
    "cdla-permissive-1.0", "cdla-permissive-2.0", "cdla-sharing-1.0",
    "ecl-2.0", "epl-1.0", "epl-2.0", 'etalab-2.0',
    "eupl-1.1", "eupl-1.2","gfdl", "gpl", "gpl-2.0", "gpl-3.0", 
    "isc", "lgpl", "lgpl-2.1", "lgpl-3.0", "lgpl-lr", "lppl-1.3c",
    "mit", "mpl-2.0", "ms-pl", "ncsa", "odbl", "odc-by", 
    "ofl-1.1", "osl-3.0", "pddl", "postgresql",
    "unlicense", "wtfpl", "zlib"
}
# Repo that has license info in CardData 
LICENSE_EXCEPTIONS = {
    "HuggingFaceM4/FineVision": "cc-by-4.0",
    "tencent/WildSpeech-Bench": "cc-by-4.0"
}

CONTENT_RESTRICTION_TAGS = {
    "not-for-all-audiences"  # HF tag for marking repository that contains "sensitive content and may contains potentionally harmful and sentitive information"
}

# EXCLUDED_DATASET_IDS = {
#     "KakologArchives/KakologArchives", # no English description
#     "ACCC1380/private-model", # no English description
#     "jamesqijingsong/chengyu", # no English description
#     "kuroneko5943/jd21", 
#     # "nvidia/Nemotron-Personas-Japan", 
#     # "liwu/MNBVC", # unsafe dataset files
#     "Derur/all-portable-apps-and-ai-in-one-url" # unsafe dataset files
# }

EXCLUDED_DATASETS = {
    "KakologArchives/KakologArchives": {
        "reason": "non-English metadata description"
    },
    # "ACCC1380/private-model": {
    #     "reason": "non-English metadata description not supported by current translation workflow"
    # },
    "jamesqijingsong/chengyu": {
        "reason": "non-English metadata description"
    },
    # "kuroneko5943/jd21": {
    #     "reason": "non-English metadata description not supported by current translation workflow"
    # },
    "Derur/all-portable-apps-and-ai-in-one-url": {
        "reason": "repository contains files flagged as unsafe"
    },
}

    
def normalize_license(license_field) -> list[str]:
    """Normalize HF license field into a list of lowercase strings."""
    if not license_field:
        return []
    if isinstance(license_field, str):
        return [license_field.lower().strip()]
    if isinstance(license_field, list):
        return [str(lic).lower().strip() for lic in license_field if lic]
    return []

def extract_license(item) -> list[str]:
    licenses = []

    for tag in (getattr(item, "tags", None) or []):
        if isinstance(tag, str) and tag.startswith("license:"):
            licenses.append(tag.split("license:", 1)[1].lower().strip())

    return licenses

def extract_tags(item) -> list[str]:
    return [
        tag.lower().strip()
        for tag in (getattr(item, "tags", None) or [])
        if isinstance(tag, str)
    ]

def get_restriction_reasons(
    item_id: str,
    obj_type: str,
    private: bool = False,
    gated: bool = False,
    disabled: bool = False,
    licenses: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[str]:
    reasons = []
    licenses = licenses or []
    tags = tags or []

    # Exclude a few datasets due to language, unsafe files 
    if obj_type == "dataset" and item_id in EXCLUDED_DATASETS:
        reasons.append(EXCLUDED_DATASETS[item_id]["reason"])

    # Check for private, gated and disabled tags
    if private:
        reasons.append("private=True")
    if gated:
        reasons.append("gated=True")
    if disabled:
        reasons.append("disabled=True")
    
    # Check if contents are "not-for-all-audiences"
    restricted_tags = [
        tag for tag in tags
        if tag.lower() in CONTENT_RESTRICTION_TAGS
    ]

    if restricted_tags:
        reasons.append(
            f"content restriction tag(s): {', '.join(restricted_tags)}"
        )

    if item_id not in LICENSE_EXCEPTIONS:
        if not licenses:
            reasons.append("no license info")
        else:
            closed = [lic for lic in licenses if lic in NON_OPEN_LICENSES]
            if closed:
                reasons.append(f"closed license(s): {', '.join(closed)}")

    return reasons

def is_valid_candidate(item, obj_type) -> tuple[bool, list[str]]:
    item_id = getattr(item, "id", "")

    reasons = get_restriction_reasons(
        item_id=item_id,
        obj_type=obj_type,
        private=bool(getattr(item, "private", False)),
        gated=bool(getattr(item, "gated", False)),
        disabled=bool(getattr(item, "disabled", False)),
        licenses=extract_license(item),
        tags=extract_tags(item)
    )

    return len(reasons) == 0, reasons

def is_restricted(item: Dict, obj_type: str) -> Tuple[bool, Optional[List[str]]]:
    if not item:
        return False, None

    item_id = item.get("id", "")

    reasons = get_restriction_reasons(
        item_id=item_id,
        obj_type=obj_type,
        private=bool(item.get("private", False)),
        gated=bool(item.get("gated", False)),
        disabled=bool(item.get("disabled", False)),
        licenses=normalize_license(item.get("license")),
        tags=[
            tag.lower().strip()
            for tag in item.get("tags", [])
            if isinstance(tag, str)
        ]
    )

    return bool(reasons), reasons or None

def apply_filter(items, kind: str):
    """Filter out restricted datasets or models"""
    kept, removed = [], []
    obj_type = "dataset" if kind == "datasets" else "model"

    for itm in items:
        restricted, reasons = is_restricted(itm, obj_type)
        if restricted:
            removed.append({
                "id": itm.get("id", "<unknown>"),
                "reasons": reasons,
                "stage": "post_fetch",
                "metadata": itm,
            })
        else:
            kept.append(itm)
    if removed:
        for r in removed:
            logger.info(f"Filtered out {kind} {r['id']} due to: {', '.join(r['reasons'])}")
    return kept, removed