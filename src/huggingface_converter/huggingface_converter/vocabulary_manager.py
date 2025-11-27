from .enums import Profile
from .constants import DCATAP_CONTROLLED_VOCABULARY, DCATAP_DE_CONTROLLED_VOCABULARY
from typing import Optional
from rdflib import URIRef
import re 


class VocabularyManager:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.vocabularies = {
            Profile.DCAT_AP: DCATAP_CONTROLLED_VOCABULARY,
            Profile.DCAT_AP_DE: DCATAP_DE_CONTROLLED_VOCABULARY
        }
        self.metric_translations = {
            "likes": "Likes",
            "downloads": "Downloads",
            "downloads_all_time": "Downloads (gesamt)",
            "trending_score": "Trend-Score"
        }

    def get_uri(self, field: str, value: str) -> Optional[URIRef]:
        if not value or not isinstance(value, str):
            return None

        base = self.vocabularies.get(self.profile, {}).get(field)
        if not base:
            return None
   
        return URIRef(f"{base}/{self._normalize_value(value)}")
    
    def get_dataset_type_uri(self, dataset_type: str) -> Optional[URIRef]:
        base = self.vocabularies[self.profile]["dataset_type"]
        return URIRef(f"{base}/{dataset_type}")
        
    def _normalize_value(self, value: str) -> str:
        value = re.sub(r'[^a-zA-Z0-9]', '_', str(value).strip())
        return value.upper()

    def get_metric_translation(self, metric: str) -> str:
        """Get German translation for metric names"""
        return self.metric_translations.get(metric.lower()) if isinstance(metric, str) else None