import iso639
import json
import re
import logging
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil import parser as date_parser
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse
from typing import Union, Dict, Any, List, Optional, Tuple
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import DCAT, DCTERMS, FOAF, RDF, XSD, SKOS, PROV, RDFS, OWL
from deep_translator import GoogleTranslator
from uuid import uuid4

from .enums import Profile, OutputFormat
from .shacl_validator import SHACLValidator, SHACLProfile
from .constants import (
    SCHEMA, DCATAP, DCATDE, ADMS, VCARD, MLS, IT6, LPWCC, 
    RESOURCE_CONFIG, METRICS, LANG_CODE_MAPPINGS, LANG_LABELS_MULTI
)
from .vocabulary_manager import VocabularyManager
from .translation_manager import TranslationManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


class HFToDCATConverter:

    def __init__(
        self,
        base_uri: str = "https://data.example.org/",
        profile: Profile = Profile.DCAT_AP,
        default_format: OutputFormat = OutputFormat.TURTLE,
        enable_translation: bool = True,
        validate_flag: bool = True
    ):
        self.base_uri = base_uri.rstrip("/") + "/"
        self.profile = profile
        self.vocab_manager = VocabularyManager(profile)
        self.enable_translation = enable_translation
        self.translator = TranslationManager(self.enable_translation) 
        self.default_format = default_format
  
        self.iso639_3_name_index = self._load_iso639_3_name_index(BASE_DIR.parent / "language_code_table" /"iso-639-3_Name_Index.tab")
        self.hf_license_mapping = self._load_hf_license_mapping(
                (BASE_DIR.parent / "license_code_table/hf2dcatap_de_license_mappings.json") if profile == Profile.DCAT_AP_DE 
                else (BASE_DIR.parent / "license_code_table/hf2dcatap_license_mappings.json")
            )
        self.hf_extension_mapping = self._load_hf_extension_mapping(
                BASE_DIR.parent / "extension_mappings/extension2_mediatype_filetype_mappings.json"
            )
        self.validate_flag = validate_flag

    def _bind_namespaces(self, g: Graph, profile: Profile) -> None:
        g.bind("dcat", DCAT)
        g.bind("dct", DCTERMS)
        g.bind("foaf", FOAF)
        g.bind("xsd", XSD)
        g.bind("schema", SCHEMA, override=True)
        g.bind("skos", SKOS)
        g.bind("prov", PROV)
        g.bind("vcard", VCARD)    
        g.bind("dcatap", DCATAP)
        g.bind("dcatde", DCATDE)
        g.bind("adms", ADMS)
        g.bind("mls", MLS)
        g.bind("owl", OWL)
        g.bind("it6", IT6)
        g.bind("lpwcc", LPWCC)
   
    def _load_hf_metadata(self, json_path: Union[str, Path]) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fetched = data.get("fetched_metadata", {})
    
        if not fetched or (
            not fetched.get("datasets") and not fetched.get("models")
        ):
            raise ValueError(
                f"No fetched datasets or models found in {json_path}"
            )

        return fetched

    def _add_translated_text(self, g: Graph, subject: URIRef, predicate: URIRef, text: str):
        """Add text with German translation if available"""
        if not isinstance(text, str):
            return

        g.add((subject, predicate, Literal(text, lang="en")))
        if self.translator.enabled:
            de_text = self.translator.translate_text(text)
            if isinstance(de_text, str) and de_text.strip().lower() != text.strip().lower():
                g.add((subject, predicate, Literal(de_text, lang="de")))

    def _load_iso639_3_name_index(self, filepath: str) -> dict:
        """
        Load ISO 639-3 Name Index from a .tab file into a dictionary.
        
        Args:
            filepath: Path to the iso-639-3_Name_Index.tab file.
            
        Returns:
            Dictionary mapping lowercase 3-letter language codes to human-readable English names.
            
        """
        name_index = {}
        
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                header = file.readline().strip()
                expected_header = "Id\tPrint_Name\tInverted_Name"
                if header != expected_header:
                    raise ValueError(
                        f"Invalid file header. Expected '{expected_header}', got '{header}'"
                    )
                
                for line_number, line in enumerate(file, start=2): 
                    line = line.strip()
                    if not line: 
                        continue
                        
                    parts = line.split("\t")
                    if len(parts) < 3:
                        logger.warning(f"Skipping malformed line {line_number}: {line}")
                        continue
                        
                    code = parts[0].strip().lower()
                    name = parts[1].strip()
                    
                    if len(code) != 3 or not code.isalpha():
                        logger.warning(f"Invalid language code format at line {line_number}: {code}")
                        continue
                        
                    name_index[code] = name

        except FileNotFoundError:
            logger.error(f"ISO 639-3 name index file not found: {filepath}")
            raise
        except UnicodeDecodeError:
            logger.error(f"Failed to decode file as UTF-8: {filepath}")
            raise ValueError("File must be UTF-8 encoded")
        except Exception as e:
            logger.error(f"Error processing ISO 639-3 file: {str(e)}")
            raise ValueError(f"Invalid file format: {str(e)}")
        
        # logger.info(f"Loaded {len(name_index)} language codes from ISO 639-3 name index")
        return name_index
    
    def _load_hf_license_mapping(self, mapping_path: Union[str, Path]) -> Dict[str, Any]:
        """Load license mapping"""
        try:
            with open(mapping_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"License mapping file not found: {mapping_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in license mapping file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading license mapping: {e}")
            raise

    def _load_hf_extension_mapping(self, mapping_path: Union[str, Path]) -> Dict[str, Any]:
        """Load extension mapping"""
        try:
            with open(mapping_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Extension mapping file not found: {mapping_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in extension mapping file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading extension mapping: {e}")
            raise

    def _create_bnode(self, resource_uri_id: str, suffix: str = "") -> BNode:
        """Create a unique BNode using dataset SHA with optional suffix"""
        return BNode(f"bn_{resource_uri_id}{f'_{suffix}' if suffix else ''}")

    def convert(self, g: Graph, resource_type: str, metadata: Dict[str, Any]) -> None:
        """
        Convert a Hugging Face dataset or model metadata entry into RDF triples.

        This method:
        - Generates a unique RDF URI for the dataset or model (based on SHA hash).
        - Adds basic RDF metadata (title, description, identifier, tags).
        - Adds domain-specific resource type information (e.g., ML model).
        - Attaches publisher info, vocab terms, metrics, and distributions.
        - Optionally validates the RDF graph after conversion.

        Args:
            g (Graph): The RDFLib graph to populate.
            resource_type (str): Either 'dataset' or 'model'.
            metadata (Dict[str, Any]): The input metadata for the resource.

        Returns:
            None
        """
        rdf_type, path_seg = RESOURCE_CONFIG[resource_type]
        resource_id = (metadata.get("id") or str(uuid4())).strip()
    
        # # Use an uuid as the main dataset URI
        # uri_id = uuid4()
        # dataset_uri = URIRef(f"{self.base_uri}{path_seg}/{uri_id}")   

        '''Use the original hugging face repo id as the uri (rather than a random uuid to ensure 
        that used model/datasets of a model can match the same model/datasets ingested as dcat:Dataset)'''
        dataset_uri = URIRef(f"{self.base_uri}data/hf_{path_seg}/{resource_id}")   
        # Add rdf type (DCAT.Dataset) 
        g.add((dataset_uri, RDF.type, rdf_type))
        # Add identifer and version 
        g.add((dataset_uri, DCTERMS.identifier, Literal(resource_id)))
        resource_sha = metadata.get("sha")
        # Add sha as dataset version
        if resource_sha and isinstance(resource_sha, str):
            g.add((dataset_uri, OWL.versionInfo, Literal(resource_sha)))
     
        hub_url = str(metadata.get("hub_url") or "").strip()
        # if hub_url.startswith(("http://", "https://")):
        #     # g.add((dataset_uri, OWL.sameAs, URIRef(hub_url)))
        #     g.add((dataset_uri, SKOS.exactMatch, URIRef(hub_url)))

        # Handle model-specific types 
        if resource_type == "model":
            g.add((dataset_uri, RDF.type, MLS.Model)) 
            g.add((dataset_uri, RDF.type, IT6.MachineLearningModel)) 
        elif  resource_type == "dataset":
            g.add((dataset_uri, RDF.type, MLS.Dataset)) 
     
        # Add basic metadata with translations
        self._add_basic_metadata(g, dataset_uri, metadata, resource_id, resource_type)
        
        # Add controlled vocabulary terms
        self._add_controlled_vocabulary_terms(g, dataset_uri, metadata)

        # Add metrics
        self._add_metrics(g, dataset_uri, resource_id, metadata)
        
        # Add publisher info
        self._add_publisher_info(g, dataset_uri, resource_id)

        self._add_citations_documentation(g, dataset_uri, resource_type, metadata)

        # Add creator info
        self._add_creator_info(g, dataset_uri, resource_id, metadata)

        # Add contactPoint and provenance:
        self._add_provenance(g, dataset_uri, resource_id)

        # Add distributions for models/datasets
        self._add_distributions(g, dataset_uri, resource_type, metadata, resource_id)        

        if self.validate_flag:
            if self._validate_graph(g):
                if self.profile == Profile.DCAT_AP:
                    dcat_ap_uri = URIRef("https://semiceu.github.io/DCAT-AP/releases/3.0.0/")
                    g.add((dataset_uri, DCTERMS.conformsTo, dcat_ap_uri))
                    g.add((dcat_ap_uri, RDF.type, DCTERMS.Standard))
                    g.add((dcat_ap_uri, RDFS.label, Literal("DCAT-AP 3.0.0", lang="en")))
                    if self.enable_translation: 
                        g.add((dcat_ap_uri, RDFS.label, Literal("DCAT-AP 3.0.0", lang="de")))
    
    def _add_property(self, g: Graph, subject: URIRef, resource_uri_id: str, name: str, value: str, category: str = None, multi: bool = False):
        """Add (name, value) pair of metadata for dataset/model as schema property"""
        if value is None:
            return
        
        if multi:
            bnode_key = f"{name}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:8]}"
        else:
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            bnode_key = safe_name
            
        bnode = self._create_bnode(resource_uri_id, bnode_key)

        g.add((subject, SCHEMA.additionalProperty, bnode))
        g.add((bnode, RDF.type, SCHEMA.PropertyValue))
        g.add((bnode, SCHEMA.name, Literal(name)))
        g.add((bnode, SCHEMA.value, Literal(value)))

        # Use propertyID as grouping/category identifier
        if category:
            g.add((bnode, SCHEMA.propertyID, Literal(category)))
            
    def _add_basic_metadata(self, g: Graph, subject: URIRef, metadata: Dict[str, Any], resource_id: str, resource_type: str):
        """
        Add basic metadata fields (title, description, tags, landing page, etc.) to the RDF graph.

        Handles:
        - Title and description (with translation & placeholder removal if applicable).
        - Keywords from tags.
        - Landing page URL.
        - Remaining metadata fields via HF_FIELD_MAPPING.
        - Boolean flags (e.g., 'private', 'gated', etc.)

        Args:
            g (Graph): RDFLib graph being constructed.
            subject (URIRef): The URIRef node representing the subject resource.
            metadata (Dict[str, Any]): Dictionary of HF resource metadata.
            resource_id (str): Resource ID, used for link/emoji cleanup decisions.

        """
        # Add id as title and keep the title the same in EN and DE
        title = remove_invalid_xml_chars(safe_get(metadata, "id", "pretty_name", "name", default="Untitled"))
        if isinstance(title, str): 
            g.add((subject, DCTERMS.title, Literal(title, lang="en")))
            # Add en title as de title as title itself is id (translation is unnrecessary)
            if self.enable_translation: 
                g.add((subject, DCTERMS.title, Literal(title, lang="de")))
        # if self.translator.enabled:
        #     de_title = self.translator.translate_text(title)
        #     # if de_title and de_title.lower() != title.lower():
        #     if de_title:
        #         g.add((subject, DCTERMS.title, Literal(de_title, lang="de")))

        # Add 'description' if available; if not, and 'readme_url' exists, point to the repository's README as the description.
        if not metadata.get("description"):
            readme_url = metadata.get("readme_url")
            if readme_url:
                metadata["description"] = (
                    f"See repository README for detailed guidance: {readme_url}"
                )
            else:
                metadata["description"] = "No description available."

        description = remove_invalid_xml_chars(metadata.get("description"))
        if isinstance(description, str):
            description = target_clean_description(description, resource_id)
            if description == "More information needed":
                if readme_url:= metadata.get("readme_url"):
                    description =  f"See repository README for detailed guidance: {readme_url}"
                else:
                    description = f"No description available."
            g.add((subject, DCTERMS.description, Literal(description, lang="en")))
            if self.translator.enabled:
                if description.startswith("See repository README for detailed guidance:"):
                    readme_url = description.split(":", 1)[-1].strip()
                    de_description = f"Siehe Repository-README für ausführliche Hinweise: {readme_url}"
                else:
                    de_description = self.translator.translate_text(description)

                if isinstance(de_description, str):
                    de_desc_stripped = de_description.strip()
                    desc_stripped = description.strip()
                    if (
                        de_desc_stripped
                        and de_desc_stripped.lower() != desc_stripped.lower()
                    ):
                        if de_desc_stripped.startswith((
                            "Eine detaillierte Anleitung finden Sie auf Repository Readme:",
                            "Siehe Repository-README für ausführliche Hinweise:"
                        )):
                            de_description_final = de_desc_stripped
                        elif desc_stripped == "No description available.":
                            de_description_final = "Keine Beschreibung verfügbar."
                        else:
                            TRANSLATION_NOTE = (
                                f"[Hinweis: Diese deutsche Beschreibung wurde maschinell aus der "
                                f"englischen Originalbeschreibung des "
                                f"{'Datensatzes' if resource_type == 'dataset' else 'Modells'} "
                                f"auf Hugging Face übersetzt.]")
                            de_description_final = f"{TRANSLATION_NOTE}\n\n{de_desc_stripped}"
                        g.add((subject, DCTERMS.description, Literal(de_description_final, lang="de")))

        # Add tags as keyword
        tags = set(as_array(metadata.get("tags", [])))
        if tags: 
            for tag in sorted(tags):
                # self._add_translated_text(g, subject, DCAT.keyword, tag)
                # Tags are kept as-is due to technical nature
                g.add((subject, DCAT.keyword, Literal(tag, lang="en")))

        # Add AI-related keyword (translated) for model
        if resource_type == "model":
            g.add((subject, DCAT.keyword, Literal("AI model", lang="en")))
            if self.enable_translation: 
                g.add((subject, DCAT.keyword, Literal("KI-Modell", lang="de")))

        # Add AI-related keyword (translated) for dataset
        elif resource_type == "dataset":
            g.add((subject, DCAT.keyword, Literal("AI dataset", lang="en")))
            if self.enable_translation: 
                g.add((subject, DCAT.keyword, Literal("KI-Datensatz", lang="de")))

        # Add hub_url as landingPage
        hub_url = str(metadata.get("hub_url") or "").strip()
        if hub_url.startswith(("http://", "https://")):
            g.add((subject, DCAT.landingPage, URIRef(hub_url)))
            g.add((subject, IT6.hasRepository, URIRef(hub_url)))
            g.add((URIRef(hub_url), RDF.type, FOAF.Document))

        # Add created_at and last_modified as issued and modified respectively
        self._add_dates(g, subject, metadata)
        # Add library, transformers, and config info for model 
        if resource_type == "model":
            self._add_library_transformers_config(g, subject, resource_id, metadata)
        # Add modality, task_categories, task_ids and size_category for datasets 
        if resource_type == "dataset":
            self._add_dataset_structured_keywords(g, subject, resource_id, metadata, tags)  
        # Add access rights and availability
        self._handle_boolean_flags(g, subject, metadata)
    
    def _add_task(self, g: Graph, subject: URIRef, task: str, origin: str):
        """Create or reuse an MLS Task and link it to the subject via dct:subject."""
        task_text = task.lower().strip()
        if not task_text:
            return
        
        task_slug = re.sub(r"[^a-z0-9]+", "-", task_text).strip("-")

        task_uri = URIRef(f"{self.base_uri}def/hf-ml-task/{task_slug}") # self defined uri for HF task

        g.add((subject, DCTERMS.subject, task_uri))

        # Link dataset task 
        if (task_uri, RDF.type, MLS.Task) not in g:
            g.add((task_uri, RDF.type, MLS.Task))
            g.add((task_uri, RDFS.label, Literal(task_text)))

        g.add((task_uri, SKOS.note, Literal(origin)))

    def _add_dataset_structured_keywords(self, g: Graph, subject: URIRef, resource_uri_id:str, metadata: Dict[str, Any], tags: List[str]) -> None:
        """Convert HF dataset modality, task_categories, task_ids, size_categories"""
        modalities = []
        size_categories = []
        # task_ids = []

        task_categories = metadata.get("task_categories", [])
        task_ids = metadata.get("task_ids", [])
    
        for tag in tags:
            if ":" not in tag:
                continue

            key, value = tag.split(":", 1)
            value = value.strip()
            if key == "modality":
                modalities.append(value)
            # elif key == "task_ids":
            #     task_ids.append(value)
            elif key == "size_categories":
                size_categories.append(value)
     
        # Add modality 
        multi = len(modalities) > 1
        for mod in modalities:
            self._add_property(g, subject, resource_uri_id, name="modality", value=mod, category="modality", multi=multi)

        # Add task category 
        multi = len(task_categories) > 1
        for task in task_categories:
            self._add_task(g, subject, task, "task_category")
 
        # Add task id 
        multi = len(task_ids) > 1
        for tid in task_ids:
            self._add_task(g, subject, tid, "task_id")
        
        # Add size_category 
        multi = len(size_categories) > 1
        for size in size_categories:
            self._add_property(g, subject, resource_uri_id, name="size_category", value=size, category="size", multi=multi)
   
    def _add_controlled_vocabulary_terms(self, g: Graph, subject: URIRef, metadata: Dict[str, Any]):
        """Add language, theme, accrualPeriodicity and spatial """
        # Add language
        self._add_language(g, subject, metadata)
     
        # Add theme
        self._add_theme(g, subject)

        # Add accrualPeriodicity (for all datasets & models) 
        freq_uri = self.vocab_manager.get_uri("accrual_periodicity", "IRREG")
        g.add((subject, DCTERMS.accrualPeriodicity,freq_uri))

        if self.profile == Profile.DCAT_AP:
            g.add((freq_uri, RDF.type, DCTERMS.Frequency))
        elif self.profile == Profile.DCAT_AP_DE:
            g.add((freq_uri, RDF.type, SKOS.Concept))
            g.add((freq_uri, SKOS.inScheme, URIRef(self.vocab_manager.vocabularies[self.profile]["accrual_periodicity"])))
        g.add((freq_uri, SKOS.prefLabel, Literal("Irregular", lang="en")))
        if self.enable_translation: 
            g.add((freq_uri, SKOS.prefLabel, Literal("Irregulär", lang="de")))

        # Add spatial
        region_value = None
        region = metadata.get("region", None)

        if isinstance(region, list):
            for entry in region:
                if isinstance(entry, str) and entry.strip():
                    region_value = entry.strip().lower()
                    break  
        elif isinstance(region, str) and region.strip():
            region_value = region.strip().lower()

        if region_value and region_value == "us":
            region_country_uri = self.vocab_manager.get_uri("spatial_country", "USA")
            g.add((subject, DCTERMS.spatial,region_country_uri))

            if self.profile == Profile.DCAT_AP:
                g.add((region_country_uri, RDF.type, DCTERMS.Location))
            elif self.profile == Profile.DCAT_AP_DE:
                g.add((region_country_uri, RDF.type, SKOS.Concept))
                g.add((region_country_uri, SKOS.inScheme, URIRef(self.vocab_manager.vocabularies[self.profile]["spatial_country"])))
            g.add((region_country_uri, SKOS.prefLabel, Literal("United States", lang="en")))
            if self.enable_translation: 
                g.add((region_country_uri, SKOS.prefLabel, Literal("Vereinigte Staaten von Amerika", lang="de")))

    def _add_language(self, g: Graph, subject: URIRef, metadata: Dict[str, Any])-> None:
        """
        Add language triples to the RDF graph.

        - Uses 'language' and 'language_bcp47' fields from metadata
        - Normalizes and maps languages to ISO 639-3 code and resolves codes to vocabulary URIs
        - Falls back to 'en' if nothing is provided
        - Marks dataset as multilingual if > LANGUAGE_LIMIT codes
        """
        language_codes = set()

        # Extract language values from metadata
        language_val = metadata.get("language")
        if isinstance(language_val, list):
            for code in language_val:
                if isinstance(code, str) and code.strip():
                    language_codes.add(code.strip().lower())
        elif isinstance(language_val, str) and language_val.strip():
            language_codes.add(language_val.strip().lower())

     
        bcp_language_val = metadata.get("language_bcp47")
        if isinstance(bcp_language_val, list):
            for code in bcp_language_val:
                base_code = self._extract_bcp47_base(code)
                if base_code:
                    language_codes.add(base_code)
        elif isinstance(bcp_language_val, str):
            base_code = self._extract_bcp47_base(bcp_language_val)
            if base_code:
                language_codes.add(base_code)

        # Fallback to English if no languages specified
        if not language_codes:
            logger.debug("No language specified, defaulting to 'en'")
            language_codes = {"en"}

        #Language threshold for listing all languanges in RDF 
        LANGUAGE_LIMIT = 10
        if len(language_codes) > LANGUAGE_LIMIT:
            # Too many languages: mark as multilingual
            mul_uri = self.vocab_manager.get_uri("language", "MUL")
            if mul_uri:
                g.add((subject, DCTERMS.language, mul_uri))
                if self.profile == Profile.DCAT_AP:
                    g.add((mul_uri, RDF.type, DCTERMS.LinguisticSystem))
                elif self.profile == Profile.DCAT_AP_DE:
                    g.add((mul_uri, RDF.type, SKOS.Concept))
                    g.add((mul_uri, SKOS.inScheme, URIRef(self.vocab_manager.vocabularies[self.profile]["language"])))

                # Add multilingual labels
                g.add((mul_uri, SKOS.prefLabel, Literal("Multilingual", lang="en")))
                if self.enable_translation: 
                    g.add((mul_uri, SKOS.prefLabel, Literal("Mehrsprachig", lang="de")))

                # logger.info(f"Dataset uses more than {LANGUAGE_LIMIT} languages. Marked as multilingual.")
            else:
                logger.warning("Could not mark as multilingual: 'MUL' URI not found in vocab manager.")               
        else: 
            language_uri_set = set()
            for code in language_codes:
                if code == "multilingual":
                    continue

                # Normalize 2-letter codes via helper function or LANG_CODE_MAPPINGS
                if len(code) == 2:
                    iso3_code = iso_2letter_to_3letter(code)
                    if iso3_code:
                        lang_code_normalized = iso3_code
                    else:
                        lang_code_normalized = LANG_CODE_MAPPINGS.get(code, code.upper())
                elif len(code) == 3:
                    lang_code_normalized = code.upper()
                else:
                    lang_code_normalized = code

                # Build EU language URI using normalized 3-letter code
                language_uri = self.vocab_manager.get_uri("language", lang_code_normalized)

                if not language_uri:
                    logger.warning(f"No URI found for language code: {code} (normalized: {lang_code_normalized})")
                    continue
    
                if language_uri in language_uri_set:
                    continue  # avoid duplicates

                language_uri_set.add(language_uri)
                g.add((subject, DCTERMS.language, language_uri))

                if self.profile == Profile.DCAT_AP:
                    g.add((language_uri, RDF.type, DCTERMS.LinguisticSystem))
                elif self.profile == Profile.DCAT_AP_DE:
                    g.add((language_uri, RDF.type, SKOS.Concept))
                    g.add((language_uri, SKOS.inScheme, URIRef(self.vocab_manager.vocabularies[self.profile]["language"])))

                # Retrieve human-readable label from ISO 639-3 or fallback
                labels_dict = LANG_LABELS_MULTI.get(lang_code_normalized)

                if not labels_dict:
                    # Attempt to load from ISO 639-3 Name Index
                    label_en = self.iso639_3_name_index.get(lang_code_normalized.lower(), lang_code_normalized)
                    labels_dict = {"en": label_en}

                    if self.translator.enabled:
                        try:
                            label_de = self.translator.translate_text(label_en)
                            if label_de:
                                labels_dict["de"] = label_de
                            else:
                                # Fallback to English
                                labels_dict["de"] = label_en
                        except TranslationError:
                            logger.info(f"No German translation for '{label_en}'. Using English only.")
                            labels_dict["de"] = label_en

                # Add labels in multiple languages
                for label_lang, label_text in labels_dict.items():
                    if label_text:  # Only add non-empty labels
                        g.add((language_uri, SKOS.prefLabel, Literal(label_text, lang=label_lang)))

    def _extract_bcp47_base(self, code: Any) -> Optional[str]:
        # Extract base language code from BCP 47 language tag
        if not isinstance(code, str) or not code.strip():
            return None
        base = code.strip().split("-")[0].strip().lower()
        if not re.match(r"^[a-z]{2,3}$", base):
            logger.debug(f"Invalid base code from BCP-47 tag: '{code}' → '{base}'")
            return None
        return base

    def _add_library_transformers_config(self, g: Graph, subject: URIRef, resource_uri_id:str, metadata: Dict[str, Any]) -> None:
        """Add library, transformers and config info"""
        # ------------------------
        # library and transformers info
        # ------------------------
        library_name = metadata.get("library_name")
        transformers_info = metadata.get("transformers_info", {})
        pipeline_tag = metadata.get("pipeline_tag")
        # Add library 
        if library_name and  isinstance(library_name, str):
            g.add((subject, SCHEMA.softwareRequirements, Literal(library_name.strip(), lang="en")))
            g.add((subject, IT6.modelArchitecture, Literal(library_name.strip(), lang="en")))
        # Add transformers info 
        if transformers_info:
            if auto_model := transformers_info.get("auto_model"):
                self._add_property(g, subject, resource_uri_id, "auto_model", auto_model, "transformers_info")
            if custom_class := transformers_info.get("custom_class"):
                self._add_property(g, subject, resource_uri_id, "custom_class", custom_class, "transformers_info")
            if  transformer_pipeline_tag := transformers_info.get("pipeline_tag"): 
                self._add_property(g, subject, resource_uri_id, "pipeline_tag", transformer_pipeline_tag, "transformers_info")       
            if  processor := transformers_info.get("processor"):
                self._add_property(g, subject, resource_uri_id, "processor", processor, "transformers_info")
           
        # Add fallback pipeline_tag when there is no transformers info
        elif pipeline_tag:
            self._add_property(g, subject, resource_uri_id, "pipeline_tag", pipeline_tag)

        # ------------------------
        # config info
        # ------------------------
        config = metadata.get("config", {})
        mask_token_added = False
        if config:
            # Add architectures 
            if (architectures := config.get("architectures", [])):
                if len(architectures) == 1:
                    self._add_property(g, subject, resource_uri_id, "architecture", architectures[0], "config", False)
                else: 
                    for arch in architectures:
                        self._add_property(g, subject, resource_uri_id, "architecture", arch, "config", True)

            # Add model_type
            if (model_type := config.get("model_type", "")):
                self._add_property(g, subject, resource_uri_id, "model_type", model_type, category="config")

            # Add tokenizer 
            if tokenizer := metadata.get("tokenizer_config", {}):
                for key, value in tokenizer.items():
                    # Skip anything not related to tokens
                    if "token" not in key:
                        continue
                    if key == "mask_token":
                        if isinstance(value, str):
                            self._add_property(g, subject, resource_uri_id, f"tokenizer_{key}", value, category="tokenizer_config")
                            mask_token_added = True
                            continue
                        if isinstance(value, dict):
                            if (content := value.get("content", "")):
                                self._add_property(g, subject, resource_uri_id, f"tokenizer_{key}", content, category="tokenizer_config")
                                mask_token_added = True
                            continue 
                       
                        continue

                    self._add_property(g, subject, resource_uri_id, f"tokenizer_{key}", value, category="tokenizer_config")
        
            if not mask_token_added:
                if (raw_mask_token := metadata.get("mask_token", "")):
                    if isinstance(raw_mask_token, str):
                            self._add_property(g, subject, resource_uri_id, "tokenizer_mask_token", raw_mask_token, category="tokenizer_config")
                    elif isinstance(raw_mask_token, dict):
                        if (content := value.get("content", "")):
                            self._add_property(g, subject, resource_uri_id, "tokenizer_mask_token", content, category="tokenizer_config")

    def _add_metrics(self, g: Graph, subject: URIRef, resource_uri_id: str, metadata: Dict[str, Any]):
        """Add metrics (likes, downloads) if available"""
        for i, (field, (action_term, dt)) in enumerate(METRICS.items()):
            count = metadata.get(field)
            if count is None:
                continue

            ic = self._create_bnode(resource_uri_id, f"metric_{i}")
            g.add((subject, SCHEMA.interactionStatistic, ic))
            g.add((ic, RDF.type, SCHEMA.InteractionCounter))
            g.add((ic, SCHEMA.name, Literal(field, lang="en")))
            de_name = self.vocab_manager.get_metric_translation(field)
            if self.enable_translation: 
                g.add((ic, SCHEMA.name, Literal(de_name, lang="de")))

            if action_term in (SCHEMA.LikeAction, SCHEMA.DownloadAction):
                g.add((ic, SCHEMA.interactionType, action_term))
            g.add((ic, SCHEMA.userInteractionCount, Literal(count, datatype=dt)))

    def _add_publisher_info(self, g: Graph, subject: URIRef, resource_id: str):
        pub_id = resource_id.split("/", 1)[0].strip()
        if not pub_id:
            return

        pub_path = quote(pub_id, safe="-_.")
        pub_uri = URIRef(f"https://huggingface.co/{pub_path}")
        g.add((subject, DCTERMS.publisher, pub_uri))
        g.add((pub_uri, RDF.type, FOAF.Agent))
        g.add((pub_uri, RDF.type, FOAF.Organization))
        g.add((pub_uri, FOAF.name, Literal(pub_id)))
        g.add((pub_uri, FOAF.homepage, pub_uri))
        # if self.profile == Profile.DCAT_AP_DE:
        #     g.add((pub_uri, RDF.type, SKOS.Concept))
        #     g.add((pub_uri, SKOS.prefLabel, Literal("Company", lang="en")))
        #     if self.enable_translation: 
        #         g.add((pub_uri, SKOS.prefLabel, Literal("Unternehmen", lang="de")))
        #     g.add((pub_uri, SKOS.inScheme, URIRef(self.vocab_manager.vocabularies[self.profile]["publisher_type"])))

    def _add_citations_documentation(self, g: Graph, subject: URIRef, resource_type: str, metadata: Dict[str, Any]):
        """
        Add publication links (arXiv, DOI) using dct:isReferencedBy and add README.md using FOAF.Documentation
        """
        if not isinstance(subject, URIRef):
            subject = URIRef(str(subject))

        # Process arXiv entries with validation
        arxiv_entries = metadata.get("arxiv", [])

        if isinstance(arxiv_entries, str):
            arxiv_entries = [arxiv_entries]
        elif isinstance(arxiv_entries, (float, int)):
            arxiv_entries = [str(arxiv_entries)]
        elif isinstance(arxiv_entries, list):
            arxiv_entries = [str(e) for e in arxiv_entries if isinstance(e, (str, int, float))]
        else:
            arxiv_entries = []

        for entry in arxiv_entries:
            if not isinstance(entry, str):
                continue
                
            arxiv_id = entry.lower().replace("arxiv:", "").strip()
            if not arxiv_id:
                continue
                
            # arXiv ID validation
            if re.fullmatch(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id) or \
                re.fullmatch(r'^[a-z-]+/\d{7}(v\d+)?$', arxiv_id):
                
                try:
                    arxiv_uri = URIRef(f"https://arxiv.org/abs/{arxiv_id}")
                    arxiv_triples = [
                        (subject, DCTERMS.isReferencedBy, arxiv_uri, g),
                        (arxiv_uri, RDF.type, RDFS.Resource, g),
                        (arxiv_uri, RDF.type, DCTERMS.BibliographicResource, g), 
                        (arxiv_uri, RDF.type, FOAF.Document, g), 
                        (arxiv_uri, RDFS.label, Literal(f"arXiv paper {arxiv_id}", lang="en"), g),
                    ]
                    if self.enable_translation:
                        arxiv_triples.append(
                            (arxiv_uri, RDFS.label, Literal(f"arXiv Papier {arxiv_id}", lang="de"), g)
                        )
                    g.addN(arxiv_triples)
                except Exception as e:
                    logger.warning(f"Failed to process arXiv entry {arxiv_id}: {str(e)}")
                    continue
                    

        # Process DOI entries with validation
        doi_entries = metadata.get("doi", [])
        if isinstance(doi_entries, str):
            doi_entries = [doi_entries]

        for entry in doi_entries:
            if not isinstance(entry, str):
                continue
                
            doi_id = entry.lower().replace("doi:", "").strip()
            if not doi_id:
                continue
                
            # DOI validation
            if re.fullmatch(r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$', doi_id):
                try:
                    doi_uri = URIRef(f"https://doi.org/{doi_id}")
                    doi_triples = [
                        (subject, DCTERMS.isReferencedBy, doi_uri, g),
                        (doi_uri, RDF.type, RDFS.Resource, g),
                        (doi_uri, RDF.type, DCTERMS.BibliographicResource, g), 
                        (doi_uri, RDF.type, FOAF.Document, g), 
                        (doi_uri, RDFS.label, Literal(f"DOI {doi_id}", lang="en"), g),
                    ]
                    if self.enable_translation:
                        doi_triples.append(
                            (doi_uri, RDFS.label, Literal(f"DOI {doi_id}", lang="de"), g)
                        )
                    g.addN(doi_triples)
                except Exception as e:
                    logger.warning(f"Failed to process DOI entry {doi_id}: {str(e)}")
                    continue

        # Process README URL
        readme_url = metadata.get("readme_url")
        if isinstance(readme_url, str) and readme_url.startswith(("http://", "https://")):
            try:
                readme_uri = URIRef(readme_url)
                g.add((subject, FOAF.page, readme_uri))
                if resource_type == "dataset":
                    label_en = "Dataset README"
                    description_en = "README file documenting the dataset."
                    if self.enable_translation: 
                        label_de = "Datensatz-README"
                        description_de = "README-Datei des Datensatzes."
                else:
                    label_en = "Model README" 
                    description_en = "README file documenting the model."
                    if self.enable_translation: 
                        label_de = "Modell-README"
                        description_de = "README-Datei des Modells."

                g.add((readme_uri, RDF.type, FOAF.Document))
                g.add((readme_uri, RDFS.label, Literal(label_en, lang="en"))) 
                g.add((readme_uri, DCTERMS.title, Literal(label_en, lang="en")))
                g.add((readme_uri, DCTERMS.description, Literal(description_en, lang="en")))

                if self.enable_translation:
                    g.add((readme_uri, RDFS.label, Literal(label_de, lang="de")))
                    g.add((readme_uri, DCTERMS.title, Literal(label_de, lang="de")))
                    g.add((readme_uri, DCTERMS.description, Literal(description_de, lang="de")))
            except Exception as e:
                logger.warning(f"Failed to process README URL {readme_url}: {str(e)}")
    
    def _add_creator_info(self, g: Graph, subject: URIRef, resource_id: str, metadata: Dict[str, Any]):
              
        creator_id = resource_id.split("/", 1)[0].strip()

        if not creator_id:
            return

        creator_path = quote(creator_id, safe="-_.")
        creator_uri = URIRef(f"https://huggingface.co/{creator_path}")

        g.add((subject, DCTERMS.creator, creator_uri))
        g.add((creator_uri, RDF.type, FOAF.Agent))
        g.add((creator_uri, RDF.type, FOAF.Organization))

        g.add((creator_uri, FOAF.name, Literal(creator_id)))
        g.add((creator_uri, FOAF.homepage, creator_uri))

        if self.profile == Profile.DCAT_AP_DE:
            g.add((creator_uri, RDF.type, SKOS.Concept))
            g.add((creator_uri, SKOS.prefLabel, Literal("Creator", lang="en")))
            if self.enable_translation: 
                g.add((creator_uri, SKOS.prefLabel, Literal("Ersteller", lang="de")))    

    def _add_provenance(self, g: Graph, subject: URIRef, resource_id: str):
        # Add provenance
        prov = self._create_bnode(resource_id, "prov")
        g.add((subject, DCTERMS.provenance, prov))
        g.add((prov, RDF.type, DCTERMS.ProvenanceStatement))
        g.add((prov, RDFS.label, Literal("The dataset was harvested from the Hugging Face website.", lang="en")))
        if self.enable_translation: 
            g.add((prov, RDFS.label, Literal("Der Datensatz wurde von der Hugging Face-Website geharvestet.", lang="de")))
      
    def _handle_boolean_flags(self, g: Graph, subject: URIRef, metadata: Dict[str, Any]):
        """Handle special boolean flags with their specific predicates"""
        # Determine if dataset is gated (restricted access)
        is_gated = metadata.get("gated") or metadata.get("private")
        
        if is_gated is not None:
            access_value = "RESTRICTED" if is_gated else "PUBLIC"
            base_uri = self.vocab_manager.vocabularies[self.profile]["access_rights"]
            access_uri = URIRef(f"{base_uri}/{access_value}")
            g.add((subject, DCTERMS.accessRights, access_uri))
            g.add((access_uri, RDF.type, DCTERMS.RightsStatement))
            if self.profile == Profile.DCAT_AP_DE:
                g.add((access_uri, RDF.type, SKOS.Concept))
                g.add((access_uri, SKOS.prefLabel, Literal(access_value.lower(), lang="en")))
                if self.enable_translation: 
                    if access_value == "RESTRICTED":
                        g.add((access_uri, SKOS.prefLabel, Literal("Eingeschränkt", lang="de")))
                    else:
                        g.add((access_uri, SKOS.prefLabel, Literal("Öffentlich", lang="de")))
                g.add((access_uri, SKOS.inScheme, URIRef(base_uri)))
        
        if "disabled" in metadata:
            availability_value = "UNAVAILABLE" if metadata["disabled"] else "AVAILABLE"
            base_uri = self.vocab_manager.vocabularies[self.profile]["availability"]
            availability_uri = URIRef(f"{base_uri}/{availability_value}")
            if self.profile == Profile.DCAT_AP:
                g.add((subject, DCATAP.availability, availability_uri))
            elif self.profile == Profile.DCAT_AP_DE:
                g.add((subject, DCATDE.availability, availability_uri))
                g.add((availability_uri, RDF.type, SKOS.Concept))
                g.add((availability_uri, SKOS.prefLabel, Literal(availability_value.lower(), lang="en")))
                if self.enable_translation: 
                    if availability_value == "UNAVAILABLE":
                        g.add((availability_uri, SKOS.prefLabel, Literal("Nicht verfügbar", lang="de")))
                    else:
                        g.add((availability_uri, SKOS.prefLabel, Literal("Verfügbar", lang="de")))
                g.add((availability_uri, SKOS.inScheme, URIRef(base_uri)))
        
    def _add_vocabulary_concept(self, g: Graph, subject: URIRef, p: URIRef, field: str, value: str):
        uri = self.vocab_manager.get_uri(field, value)
        if uri:
            g.add((subject, p, uri))
            g.add((uri, RDF.type, SKOS.Concept))
            g.add((uri, SKOS.prefLabel, Literal(value, lang="en")))

            if self.translator.enabled:
                de_label = self.translator.translate_text(value)
                if isinstance(de_label, str):
                    if de_label.strip().lower() != value.strip().lower():
                        g.add((uri, SKOS.prefLabel, Literal(de_label, lang="de")))

            if self.profile == Profile.DCAT_AP_DE:
                scheme = self.vocab_manager.vocabularies[self.profile].get(field)
                if scheme:
                    g.add((uri, SKOS.inScheme, URIRef(scheme)))


    def _add_distributions(self, g: Graph, subject: URIRef, resource_type: str, metadata: Dict[str, Any], resource_id: str) -> None:
        """Add distributions and related metadata for models or datasets."""
        # Handle license 
        license_uri = self._process_license(g, metadata.get("license"))

        # Use minted repo URI (for repo-level overall distribution)
        HF_FORMAT_URI = URIRef(f"{self.base_uri}/def/file-type/repository") # self defined uri
        g.add((HF_FORMAT_URI, RDF.type, DCTERMS.MediaTypeOrExtent))
        g.add((HF_FORMAT_URI, SKOS.exactMatch, LPWCC.Repository))

        label_base = resource_type.capitalize() 
        g.add((HF_FORMAT_URI, RDFS.label, Literal(f"Hugging Face {label_base} Repository", lang="en")))
        if self.enable_translation:
            de_label = "Hugging Face Datensatz-Repository" if label_base == "Dataset" else "Hugging Face Modell-Repository"
            g.add((HF_FORMAT_URI, RDFS.label, Literal(de_label, lang="de")))

        g.add((HF_FORMAT_URI, DCTERMS.description, Literal("Represents the full Hugging Face repository (not a single file).", lang="en")))
        if self.enable_translation: 
            g.add((HF_FORMAT_URI, DCTERMS.description, Literal("Repräsentiert das gesamte Hugging-Face-Repository (nicht nur eine einzelne Datei).", lang="de")))

        if resource_type == "model":
            self._add_model_distributions(g, subject, metadata, resource_id, HF_FORMAT_URI, license_uri)
        elif resource_type == "dataset":
            self._add_dataset_distributions(g, subject, metadata, resource_id, HF_FORMAT_URI,license_uri)

    def _process_license(self, g: Graph, license_value: Union[str, List[str], None]) -> Optional[URIRef]:
        """
        Process license information and add appropriate RDF triples.

        This method:
        - Normalizes the license value
        - Attempts to map it using `hf_license_mapping`
        - If no value is provided or no mapping is found, falls back to the 'unknown' license
        - Adds license metadata such as type, prefLabels, notes, exactMatch URIs, and inScheme (if DCAT-AP.DE)

        Args:
            g (Graph): The RDF graph to which triples are added.
            license_value (str | List[str] | None): The license name or list of license names.

        Returns:
            URIRef: The URI of the license used (mapped or 'unknown'), or None if all mappings fail.
        """
        if not license_value:
            license_value = "unknown"
        
        if isinstance(license_value, list):
            license_value = license_value[0]
        
        license_value = license_value.strip().lower()

        mapping = self.hf_license_mapping.get(license_value) or self.hf_license_mapping.get("unknown")
        if not mapping:
            logger.info(f"No license mapping found for: {license_value}, and no 'unknown' fallback present.")
            return None

        license_uri = URIRef(mapping["uri"])
        g.add((license_uri, RDF.type, DCTERMS.LicenseDocument))

        # Add prefLabel(s)
        if mapping.get("label_en"):
            g.add((license_uri, SKOS.prefLabel, Literal(mapping["label_en"], lang="en")))
        if self.enable_translation and mapping.get("label_de"):
            g.add((license_uri, SKOS.prefLabel, Literal(mapping["label_de"], lang="de")))

        # Add skos:note(s)
        if mapping.get("note_en"):
            g.add((license_uri, SKOS.note, Literal(mapping["note_en"], lang="en")))
        if self.enable_translation and mapping.get("note_de"):
            g.add((license_uri, SKOS.note, Literal(mapping["note_de"], lang="de")))

        # Add skos:exactMatch links
        for match in mapping.get("exact_matches", []):
            g.add((license_uri, SKOS.exactMatch, URIRef(match)))

        # Add inScheme for DCAT-AP.DE
        if self.profile == Profile.DCAT_AP_DE:
            g.add((license_uri, SKOS.inScheme, URIRef(self.vocab_manager.vocabularies[self.profile]["license"])))

        return license_uri

    def _add_model_distributions(self, g: Graph, subject: URIRef, metadata: Dict[str, Any], 
                            resource_id: str, hf_format_uri: URIRef, license_uri: Optional[URIRef]) -> None:
        """Add distributions and related info for models."""

        for dist in metadata.get("distributions", []):
             self._add_model_distribution(g, subject, dist, resource_id, metadata, hf_format_uri, license_uri)
        
        # Link used datasets
        used_datasets = as_array(metadata.get("datasets"))
        if used_datasets:
            self._link_used_datasets(g, subject, resource_id, used_datasets)
        
        # Link base model
        base_models = as_array(metadata.get("base_model"))
        if base_models:
            self._link_base_models(g, subject, resource_id, base_models)

    def _add_model_distribution(self, g: Graph, subject: URIRef, dist_meta: Dict[str, Any],
            resource_id: str, metadata: Dict[str, Any], hf_format_uri: URIRef, license_uri: Optional[URIRef],
                                ) -> None:
        """Add a model distribution."""      
        # dist_uri = URIRef(f"{subject}/distribution/{quote(dist_name)}")
        dist_slug_enc = quote(dist_meta.get("slug"), safe="/")
        dist_uri = URIRef(f"{subject}/distribution/{dist_slug_enc}")
        
        g.add((subject, DCAT.distribution, dist_uri))

        g.add((dist_uri, RDF.type, DCAT.Distribution))

        # Add accessURL and downloadURL
        for key, prop in [("accessURL", DCAT.accessURL), ("downloadURL", DCAT.downloadURL)]:
            url = dist_meta.get(key)
            if url:
                sanitized_url = sanitize_url_for_rdf(url)
                if sanitized_url:
                    g.add((dist_uri, prop, URIRef(sanitized_url)))
           
        if license_uri:
            g.add((dist_uri, DCTERMS.license, license_uri))
        
        self._add_dates(g, dist_uri, metadata)

        
        title_en = dist_meta.get("name", "")
        description_en = dist_meta.get("description", "")
        dist_type = dist_meta.get("type")
        g.add((dist_uri, DCTERMS.title, Literal(title_en, lang="en")))
        g.add((dist_uri, DCTERMS.description, Literal(description_en, lang="en")))

        ftype = dist_meta.get("type", "")
        if self.enable_translation:
            if ftype == "repo":
                title_de = title_en.replace("All files for", "Alle Dateien für")
            else:
                title_de = title_en
            description_de = translate_model_dist_description(dist_type, description_en)
            g.add((dist_uri, DCTERMS.title, Literal(title_de, lang="de")))
            g.add((dist_uri, DCTERMS.description, Literal(description_de, lang="de")))
           
        size = dist_meta.get("size")
        if size:
            self._add_byte_size(g, dist_uri, size)
        
        file_ext = dist_meta.get("fileExtension")
        if isinstance(file_ext, str) and file_ext.strip():
            file_ext = file_ext.lower().strip()
            if not file_ext.startswith('.'):
                file_ext = f".{file_ext}"
            self._add_file_media_type(g, dist_uri, file_ext)
       
        if dist_type == "repo":
            if hf_format_uri:
                g.add((dist_uri, DCTERMS.format, hf_format_uri))


    def _add_dataset_distributions(self, g: Graph, subject: URIRef, metadata: Dict[str, Any],
                                resource_id: str, hf_format_uri: URIRef, license_uri: Optional[URIRef]) -> None:
        """Add distributions and related info for datasets."""
        # Add file distributions
        for dist in metadata.get("distributions", []):
            self._add_dataset_distribution(g, subject, dist, resource_id, metadata, hf_format_uri, license_uri)
              
        # Add theme
        self._add_theme(g, subject)


    def _add_dataset_distribution(self, g: Graph, subject: URIRef, dist_meta: Dict[str, Any],
            resource_id: str, metadata: Dict[str, Any], hf_format_uri: URIRef, license_uri: Optional[URIRef],
                                ) -> None:
        """Add a dataset distribution."""      
        # dist_uri = URIRef(f"{subject}/distribution/{quote(dist_name)}")
        dist_slug_enc = quote(dist_meta.get("slug"), safe="/")
        dist_uri = URIRef(f"{subject}/distribution/{dist_slug_enc}")
        
        g.add((subject, DCAT.distribution, dist_uri))
        g.add((dist_uri, RDF.type, DCAT.Distribution))

        # Add accessURL and downloadURL
        for key, prop in [("accessURL", DCAT.accessURL), ("downloadURL", DCAT.downloadURL)]:
            url = dist_meta.get(key)
            if url:
                sanitized_url = sanitize_url_for_rdf(url)
                if sanitized_url:
                    g.add((dist_uri, prop, URIRef(sanitized_url)))
           
        if license_uri:
            g.add((dist_uri, DCTERMS.license, license_uri))
        
        self._add_dates(g, dist_uri, metadata)
     
        title_en = dist_meta.get("name", "")
        description_en = dist_meta.get("description", "")
        dist_type = dist_meta.get("type")
        g.add((dist_uri, DCTERMS.title, Literal(title_en, lang="en")))
        g.add((dist_uri, DCTERMS.description, Literal(description_en, lang="en")))

        if self.enable_translation:
            title_de = translate_dataset_dist_title(dist_type, title_en)
            description_de = translate_dataset_dist_description(dist_type, description_en)
            g.add((dist_uri, DCTERMS.title, Literal(title_de, lang="de")))
            g.add((dist_uri, DCTERMS.description, Literal(description_de, lang="de")))
           
        size = dist_meta.get("size")
        if size:
            self._add_byte_size(g, dist_uri, size)
        
        file_ext = dist_meta.get("fileExtension")
        if isinstance(file_ext, str) and file_ext.strip():
            file_ext = file_ext.lower().strip()
            if not file_ext.startswith('.'):
                file_ext = f".{file_ext}"
            self._add_file_media_type(g, dist_uri, file_ext)

       
        if dist_type == "repo":
            if hf_format_uri:
                g.add((dist_uri, DCTERMS.format, hf_format_uri))
    
    def _add_file_media_type(self, g: Graph, dist_uri: URIRef, ext: str) -> None:
        """
        Add file type and media type information for a distribution based on file extension.
        
        Args:
            g: RDF graph
            dist_uri: Distribution URI
            ext: File extension (e.g., '.bin', '.h5')
        """
        ext = ext.lower().strip()
        if not ext:
            return
        
        extension_mapping = self.hf_extension_mapping.get(ext) 
       
        if extension_mapping: 
        
            file_type_uri = extension_mapping.get("file_type_uri")
            file_type_label = extension_mapping.get("file_type_label")
            file_type = extension_mapping.get("file_type")              
            media_type_uri = extension_mapping.get("media_type_uri")
            see_also = extension_mapping.get("see_also")
        
            if media_type_uri:
                g.add((dist_uri, DCAT.mediaType, URIRef(media_type_uri)))
                g.add((URIRef(media_type_uri), RDF.type, DCTERMS.MediaType))
        
            if file_type_uri:
                g.add((dist_uri, DCTERMS.format, URIRef(file_type_uri)))
                g.add((URIRef(file_type_uri), RDF.type, DCTERMS.MediaTypeOrExtent))

                if file_type_label:
                    g.add((URIRef(file_type_uri), SKOS.prefLabel, Literal(file_type_label, lang="en")))
                    if self.enable_translation:
                        # Add the same literal for de
                        g.add((URIRef(file_type_uri), SKOS.prefLabel, Literal(file_type_label, lang="de")))
                        
                if see_also:
                    g.add((URIRef(file_type_uri), RDFS.seeAlso, URIRef(see_also)))

        else:
            self._add_fallback_file_media_type(g, dist_uri, ext)
        
            
    def _add_fallback_file_media_type(self, g: Graph, dist_uri: URIRef, ext: str) -> None:
        """Add a generic binary file fallback."""
        clean_ext = ext.lstrip('.').upper()
        file_type_uri = URIRef(f"https://piveau.io/def/file-type/{clean_ext.replace(' ', '-')}") # self defined uri

        media_type_uri = URIRef("http://www.iana.org/assignments/media-types/application/octet-stream")
        g.add((dist_uri, DCAT.mediaType, media_type_uri))
        g.add((URIRef(media_type_uri), RDF.type, DCTERMS.MediaType))

        g.add((dist_uri, DCTERMS.format, file_type_uri))
        g.add((file_type_uri, RDF.type, DCTERMS.MediaTypeOrExtent))

        g.add((file_type_uri, SKOS.prefLabel, Literal(clean_ext, lang="en")))
        # g.add((file_type_uri, SKOS.note, Literal(
        #     f"Minted fallback URI for extension {clean_ext}, no official controlled vocabulary available", lang="en")))

        if self.enable_translation:
            g.add((file_type_uri, SKOS.prefLabel, Literal(clean_ext, lang="de")))
            # g.add((file_type_uri, SKOS.note, Literal(
            #     f"Erzeugte Fallback-URI für die Erweiterung {clean_ext}, kein offizielles kontrolliertes Vokabular verfügbar", lang="de")))

    def _add_dates(self, g: Graph, subject: URIRef, metadata: Dict[str, Any]) -> None:
        """Add creation and modification dates to a subject."""
        for field, predicate in [("created_at", DCTERMS.issued), ("last_modified", DCTERMS.modified)]:
            raw_value = metadata.get(field)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            
            formatted = format_datetime(raw_value)
            if not formatted:
                logger.warning(f"Unable to parse date from field '{field}': {raw_value}")
                continue

            g.add((subject, predicate, Literal(formatted["@value"], datatype=formatted["@type"])))

    def _add_theme(self, g: Graph, subject: URIRef) -> None:
        """Add theme information to a subject."""
        theme_uri = self.vocab_manager.get_uri("theme", "TECH")
        if not theme_uri:
            return
        
        g.add((subject, DCAT.theme, theme_uri))
        g.add((theme_uri, RDF.type, SKOS.Concept))
        g.add((theme_uri, SKOS.prefLabel, Literal("Science and technology", lang="en")))
        if self.enable_translation: 
            g.add((theme_uri, SKOS.prefLabel, Literal("Wissenschaft und Technologie", lang="de")))
        
        if self.profile == Profile.DCAT_AP_DE:
            g.add((theme_uri, SKOS.inScheme, URIRef(self.vocab_manager.vocabularies[self.profile]["theme"])))
    
    def _add_byte_size(self, g: Graph, subject: URIRef, size: Any):
        try:
            size_int = int(size)
            if size_int > 0:
                g.add((subject, DCAT.byteSize, Literal(size_int, datatype=XSD.nonNegativeInteger)))
            else:
                logger.info(f"Ignoring zero or negative byte size: {size}")
        except (ValueError, TypeError):
            logger.warning(f"Invalid byte size: {size}")

    def _link_used_datasets(self, g: Graph, subject: URIRef, resource_id: str, dataset_ids: List[str]) -> None:
        """Link model to datasets it uses via it6:trainedOn
        """
        if isinstance(dataset_ids, str):
            logger.exception("dataset_ids is a str rather than a list")
        for dataset_id in dataset_ids:
            if not isinstance(dataset_id, str):
                continue

            dataset_id = dataset_id.strip()
            safe_id = quote(dataset_id, safe="/-_.~")

            # Check if dataset_id includes user/repo structure
            if '/' not in dataset_id:
                # Fallback: use Hugging Face search URL
                hf_url = f"https://huggingface.co/datasets?search={quote(dataset_id)}"
                hash_suffix = hashlib.sha1(dataset_id.encode("utf-8")).hexdigest()[:8]
                dataset_uri = URIRef(f"{self.base_uri}data/hf_dataset/{safe_id}--{hash_suffix.lower()}")
                incomplete_id = True
            else:
                hf_url = f"https://huggingface.co/datasets/{safe_id}"
                dataset_uri = URIRef(f"{self.base_uri}data/hf_dataset/{safe_id}")
                incomplete_id = False

            # Add datasets as training datasets
            g.add((subject, IT6.trainedOn, dataset_uri))
            g.add((dataset_uri, RDF.type, MLS.Dataset))
              
            g.add((dataset_uri, DCTERMS.title, Literal(dataset_id, lang="en")))
            if self.enable_translation: 
                g.add((dataset_uri, DCTERMS.title, Literal(dataset_id, lang="de")))
  
            # Description with fallback warning if necessary
            if incomplete_id:
                desc_en = (
                    f"The dataset '{dataset_id}' was used in training the model '{resource_id}."                   
                    f"The identifier '{dataset_id}' may be incomplete."
                )
                desc_de = (
                    f"Der Datensatz '{dataset_id}' wurde beim Training des Modells '{resource_id}' verwendet."
                    f"Die ID '{dataset_id}' ist möglicherweise unvollständig. "
           
                )
            else:
                desc_en = f"The '{dataset_id}' dataset was used in training the model '{resource_id}'."
                desc_de = f"Der Datensatz '{dataset_id}' wurde beim Training des Modells '{resource_id}' verwendet."
                
                g.add((dataset_uri, SKOS.exactMatch, URIRef(hf_url)))    

            g.add((dataset_uri, DCTERMS.description, Literal(desc_en, lang="en")))
            if self.enable_translation: 
                g.add((dataset_uri, DCTERMS.description, Literal(desc_de, lang="de")))
    
    def _link_base_models(self, g: Graph, subject: URIRef, resource_id: str, base_models: List[Dict[str, Any]]) -> None:
        """
        Link a model to its base models using prov:wasDerivedFrom and dct:references.
        """
        for model in base_models:
            if not isinstance(model, Dict):
                continue
            model_id = (model.get("name") or "").strip()
            model_type = (model.get("type") or "").strip()
            if not model_id:
                continue 
            if ":" in model_id:
                model_id = split(":")[-1]

            safe_id = quote(model_id, safe="/-_.~")

            # Construct URL for the base model
            if '/' not in model_id:
                hf_url = f"https://huggingface.co/models?search={quote(model_id)}"      
                hash_suffix = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
                model_uri = URIRef(f"{self.base_uri}data/hf_model/{safe_id}--{hash_suffix.lower()}")
                incomplete_id = True
            else:
                hf_url = f"https://huggingface.co/{model_id}"
                model_uri =  URIRef(f"{self.base_uri}data/hf_model/{safe_id}")
                incomplete_id = False
            
            # Add base model provenance 
            g.add((subject, PROV.wasDerivedFrom, model_uri))

            g.add((model_uri, DCTERMS.title, Literal(model_id, lang="en")))
            if self.enable_translation:
                g.add((model_uri, DCTERMS.title, Literal(model_id, lang="de")))

            # Description
            if incomplete_id:
                desc_en = (
                    f"This model '{model_id}' served as the base for developing the model '{resource_id}'"
                    f"{f' ({model_type})' if model_type and model_type.lower() != 'base' else ''}."
                    f"The identifier '{model_id}' may be incomplete."
                )

                desc_de = (
                    f"Dieses Modell '{model_id}' diente als Basismodell für die Entwicklung des Modells '{resource_id}'"
                    f"{f' ({model_type})' if model_type and model_type.lower() != 'base' else ''}. "
                    f"Die ID '{model_id}' ist möglicherweise unvollständig."
                )
            else:
                desc_en = (
                    f"This model '{model_id}' served as the base for developing the model '{resource_id}'"
                    f"{f' ({model_type})' if model_type and model_type.lower() != 'base' else ''}."
                )
                desc_de = (
                    f"Dieses Modell '{model_id}' diente als Basismodell für die Entwicklung des Modells '{resource_id}'"
                    f"{f' ({model_type})' if model_type and model_type.lower() != 'base' else ''}. "
                )

            g.add((model_uri, DCTERMS.description, Literal(desc_en, lang="en")))
            if self.enable_translation:
                g.add((model_uri, DCTERMS.description, Literal(desc_de, lang="de")))

    def _validate_graph(self, g: Graph)-> bool:
        validator = SHACLValidator()
        profile_map = {
            Profile.DCAT_AP: SHACLProfile.DCAT_AP,
            Profile.DCAT_AP_DE: SHACLProfile.DCAT_AP_DE
        }
        result = validator.validate_graph(g, profile=profile_map[self.profile])
        if not result.conforms:
            logger.warning("❌ SHACL validation failed")
            for group in validator.group_validation_results(
                result.details, include_warnings=True, include_infos=False, max_focus_nodes=5
            ):
                logger.warning(
                    f"{group['severity']} {group['count']}× {group['message']} @ {group['path']}"
                )
            return False

        # logger.info("✅ SHACL validation passed")
        return True  

    def _thread_safe_convert(self, resource_type: str, item_data: Dict[str, Any]) -> Optional[Graph]:
        """Process one item in a thread-safe manner"""
        thread_g = Graph()
        self._bind_namespaces(thread_g, self.profile)
        
        try:
            thread_converter = HFToDCATConverter(
                base_uri=self.base_uri,
                profile=self.profile,
                enable_translation=self.enable_translation
            )
            thread_converter.convert(thread_g, resource_type, item_data)
            return thread_g
        except Exception as e:
            logger.exception(f"Failed to convert {item_data.get('id')}: {str(e)}")
            return None

    def run_parallel(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path] = None,
        output_base: Optional[str] = None,
        output_format: Optional[List[OutputFormat]] = None,
        split_output: bool = True,
        max_workers: int = 8
    ) -> List[Path]:
        """
        Converts Hugging Face datasets/models metadata in parallel to RDF and exports them.

        Args: 
            input_path (str, Path): Path to the input JSON file.
            output_dir (str, Path): Directory where output files will be saved. Defaults to ./output.
            output_base (str): Custom base filename (without extension). If not provided, auto-generated.
            output_format (List): Formats to export to (default: [self.default_format]).
            split_output (bool): If True, saves datasets and models into separate files.
            max_workers : Number of worker threads to use.

        Returns: 
            List[Path]: List of created output file paths.
        """   
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        fetched = self._load_hf_metadata(input_path)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

        output_dir = Path(output_dir or "output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Archive any existing output files
        archive_dir = output_dir / "archive"
        archive_old_outputs(output_dir, archive_dir)
        # purge_output_dir(output_dir)

        dataset_items = fetched.get("datasets", [])
        model_items = fetched.get("models", [])
        total_count = len(dataset_items) + len(model_items)

        # Grouping logic
        if split_output:
            groups = []
            if dataset_items:
                groups.append(("datasets", dataset_items))
            if model_items:
                groups.append(("models", model_items))
        else:
            combined_items = []
            if dataset_items:
                combined_items.append(("datasets", dataset_items))
            if model_items:
                combined_items.append(("models", model_items))
            if combined_items:
                groups = [("all", combined_items)]
            else:
                logger.warning("No metadata items found to convert.")
                return []

        all_output_files: List[Path] = []
        multi_groups = len(groups) > 1

        for group_name, items in groups:
            merged_graph = Graph()
            self._bind_namespaces(merged_graph, self.profile)
            error_found = False
            # Parallel conversion
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if group_name == "all":
                    for resource_group, items_list in items:
                        resource_type = resource_group.rstrip("s")
                        futures = [
                            executor.submit(self._thread_safe_convert, resource_type, item)
                            for item in items_list
                        ]
                else:
                    resource_type = group_name.rstrip("s")
                    futures = [
                        executor.submit(self._thread_safe_convert, resource_type, item)
                        for item in items
                    ]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()  
                        if result is not None:
                            for triple in result:
                                merged_graph.add(triple)
                    except Exception as e:
                        error_found = True
                        logger.exception(f"❌ Parallel worker failed in converting '{group_name}': {e}")
                    
            if error_found:
                error_msg = f"❌ Conversion failed for one or more items in '{group_name}'."
                if multi_groups:
                    logger.error(error_msg + f" Skipping validation and output for {group_name}.")
                    continue
                else:
                    raise RuntimeError(error_msg + f" Skipping validation and output.")

            if len(merged_graph) == 0:
                error_msg = f"No triples were produced for '{group_name}'. Skipping validation and output."
                if multi_groups:
                    logger.error(error_msg)
                    continue
                else:
                    raise RuntimeError(error_msg)

            # Validate and output
            if self.validate_flag:
                try:
                    if not self._validate_graph(merged_graph):
                        error_msg = f"❌ Final SHACL validation failed for converted {group_name}."
                        if multi_groups:
                            logger.error(error_msg + f"Skipping output for {group_name}")
                            continue
                        else:
                            raise RuntimeError(error_msg + f"Skipping output.")
                    else:
                        logger.info(f"✅ Final SHACL validation passed for converted {group_name}")
                except Exception as e:
                    if multi_groups:
                        logger.exception(f"❌ Validation crashed for converted'{group_name}': {e}")
                        continue
                    else:
                        raise
            else:
                logger.info("⚠️ SHACL validation skipped (validate_flag is False)")

            count = len(items) if group_name != "all" else total_count
            if output_base:
                base_name = f"{output_base}_{group_name}"
            else:
                base_name = f"{self.profile.name.lower()}_{group_name}_{count}_{timestamp_str}"

            base_path = output_dir / base_name
            formats = output_format or [self.default_format]

            for fmt in formats:
                ext = {
                    OutputFormat.RDFXML: ".rdf",
                    OutputFormat.TURTLE: ".ttl",
                    OutputFormat.JSONLD: ".jsonld",
                    OutputFormat.NTRIPLES: ".nt"
                }.get(fmt, f".{fmt.value}")

                full_path = base_path.with_suffix(ext)
                merged_graph.serialize(destination=str(full_path), format=fmt.value)
                # logger.info(f"✅ Successfully wrote output to {full_path}")
                all_output_files.append(full_path)

        return all_output_files

   

# Utility functions
def safe_get(data: Dict, *keys, default=None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default

def as_array(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]

def format_datetime(date_str: str, fmt: str = "iso") -> Optional[Dict[str, str]]:
    try:
        dt = date_parser.parse(date_str)
        if fmt == "iso":
            if dt.time() == datetime.min.time():
                return {"@value": dt.date().isoformat(), "@type": str(XSD.date)}
            return {"@value": dt.isoformat(timespec="seconds"), "@type": str(XSD.dateTime)}
        return {"@value": dt.strftime(fmt), "@type": str(XSD.dateTime)}
    except Exception:
        return None

def sanitize_url_for_rdf(url: str | None) -> str | None:
    return url.strip().replace(" ", "%20") if url else url

def remove_invalid_xml_chars(text: str) -> str:
    """Remove characters that are not allowed in XML 1.0."""
    if not isinstance(text, str):
        return text
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

def target_clean_description(text: str, dataset_id: str) -> str:
    """
    Cleans edge-case description content for specific Hugging Face models.

    This function applies model-specific cleaning steps, such as:
    - Removing emojis from Phi-3 model descriptions (which may include markdown-adjacent symbols).
    - Flattening specific markdown links (e.g., for the PHI Standardization paper) in the nvidia/C-RADIOv2-VLM-H-RC3 model description.

    Args:
        text (str): The original English description.
        dataset_id (str): The Hugging Face model ID (e.g., "microsoft/Phi-3-mini-128k-instruct").

    Returns:
        str: The cleaned description string, with dataset-specific fixes applied.
    """
    text = remove_emojis_for_phi(text, dataset_id)
    text = demarkdown_links_for_cradiov2(text, dataset_id)
    return text

def remove_emojis_for_phi(text: str, dataset_id: str) -> str:
    if not dataset_id.startswith("microsoft/Phi-"):
        return text

    KNOWN_EMOJIS = ["👩‍🍳", "🖥️", "🛠️", "📖", "📰", "🏡", "📱"]

    for emoji in KNOWN_EMOJIS:
        text = text.replace(emoji, "")
    return text

def demarkdown_links_for_cradiov2(text: str, dataset_id: str) -> str:
    if dataset_id != "nvidia/C-RADIOv2-VLM-H-RC3":
        return text  

    pattern_map = {
        r'\[PHI Standardization\]\(([^)]+)\)': r'PHI Standardization (\1)',
    }

    for pattern, replacement in pattern_map.items():
        text = re.sub(pattern, replacement, text)
    return text

def iso_2letter_to_3letter(code_2: str) -> Optional[str]:
    """Convert 2-letter ISO 639-1 code to 3-letter ISO 639-3 code 
    
    Args:
        code_2: 2-letter language code (e.g., 'de', 'en')
    
    Returns:
        Uppercase 3-letter code (e.g., 'DEU', 'ENG') or None if invalid
    
    Raises:
        ValueError: If input is not a 2-letter string
    """
    if not isinstance(code_2, str) or len(code_2.strip()) != 2:
        logger.error(f"Invalid language code format: {code_2}")
    
    code_2 = code_2.strip().lower()
    
    try:
        lang = iso639.Language.from_part1(code_2)
        if not lang:
            return None
            
        if hasattr(lang, 'part3') and lang.part3:
            return lang.part3.upper()
            
        # Fallback to terminologic code (ISO 639-2/T)
        if hasattr(lang, 'part2t') and lang.part2t:
            return lang.part2t.upper()
            
        return None       
    except KeyError:
        return None
    except AttributeError as e:
        logging.warning(f"Unexpected language structure for {code_2}: {str(e)}")
        return None

def archive_old_outputs(output_dir: Path, archive_dir: Path) -> Optional[Path]:
    """
    Move existing RDF output files into a timestamped subfolder under archive_dir.
    
    Returns:
        Path to the archive directory if successful, None otherwise.
    """
    try:
        # Gather all existing RDF outputs
        old_files = []
        for ext in [".ttl", ".rdf", ".jsonld", ".nt"]:
            old_files.extend(output_dir.glob(f"*{ext}"))

        if not old_files:
            logging.info("No old RDF files found to archive.")
            return None

        # Build archive folder 
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_path = archive_dir / f"run_{timestamp}"
        archive_path.mkdir(parents=True, exist_ok=True)

        # Move all files
        for file in old_files:
            try:
                shutil.move(str(file), archive_path / file.name)
            except Exception as e:
                logging.warning(f"Could not move {file.name} to archive: {e}")

        logging.info(f"Archived old outputs to {archive_path}")
        return archive_path

    except Exception as e:
        logging.error(f"Failed to archive old output files: {e}")
        return None

def purge_output_dir(output_dir: Path, extensions=None) -> None:
    """
    Delete stale RDF files before a new run.
    
    Args:
        output_dir: Directory where output files are stored.
        extensions: List of extensions to delete (defaults to RDF types).
    """
    try:
        extensions = extensions or [".ttl", ".rdf", ".jsonld", ".nt"]
        deleted_any = False

        for ext in extensions:
            for file in output_dir.glob(f"*{ext}"):
                try:
                    file.unlink()
                    deleted_any = True
                    logging.debug(f"Deleted: {file}")
                except Exception as e:
                    logging.warning(f"Could not delete {file}: {e}")

        if deleted_any:
            logging.info("Purged old RDF files from output directory.")
        else:
            logging.info("No old RDF files found to purge.")

    except Exception as e:
        logging.error(f"Error while purging output directory: {e}")

def translate_dataset_dist_description(dist_type: str, description_en: str) -> str:
    if dist_type == "parquet-file":
        return description_en \
            .replace("Config:", "Konfiguration:") \
            .replace("Size:", "Größe:").replace("bytes", "Bytes")

    elif dist_type == "parquet-aggregate":
        return description_en \
            .replace("All Parquet files", "Alle Parquet-Dateien") \
            .replace("files", "Dateien") \
            .replace("total size", "Gesamtgröße").replace("bytes", "Bytes")

    elif dist_type == "file":
        return description_en \
            .replace("File:", "Datei:") \
            .replace("Size:", "Größe:").replace("bytes", "Bytes")

    elif dist_type == "repo":
        return description_en \
            .replace("Browse the repository and access",
                     "Durchsuchen Sie die Repository und greifen Sie auf") \
            .replace("dataset files", "Datensatz-Dateien") \
            .replace("Total size", "Gesamtgröße").replace("bytes", "Bytes")

    return description_en  

def translate_dataset_dist_title(dist_type:str, title_en: str) -> str:
    if dist_type == "parquet-aggregate":
        return title_en.replace("All Parquet files for", "Alle Parquet-Dateien für")
    elif dist_type == "repo":
        return title_en.replace("All files for", "Alle Dateien für")
    return title_en

def translate_model_dist_description(dist_type: str, description_en: str) -> str:
    if dist_type == "weight":
        return description_en \
            .replace("Model weight file:", "Modell-Gewichtsdatei:") \
            .replace("Size:", "Größe:").replace("bytes", "Bytes")

    elif dist_type == "config":
        return description_en \
            .replace("Configuration file:", "Konfigurationsdatei:") \
            .replace("Size:", "Größe:").replace("bytes", "Bytes")

    elif dist_type == "tokenizer":
        return description_en \
            .replace("Tokenizer file:", "Tokenizer-Datei:") \
            .replace("Size:", "Größe:").replace("bytes", "Bytes")

    elif dist_type == "additional":  # non-core, small repo
        return description_en \
            .replace("Additional file:", "Zusätzliche Datei:") \
            .replace("Size:", "Größe:").replace("bytes", "Bytes")

    elif dist_type == "repo":  # repo-level fallback
        return description_en \
            .replace("Browse the repository and access",
                     "Durchsuchen Sie das Repository und greifen Sie auf") \
            .replace("all", "alle").replace("files", "Dateien") \
            .replace("Total size", "Gesamtgröße").replace("bytes", "Bytes")

    return description_en

def translate_model_dist_title(dist_type: str, title_en: str) -> str:
    if dist_type == "repo":
        return title_en.replace("All files for", "Alle Dateien für")
    elif dist_type == "weight":
        return title_en.replace("Model weight file:", "Modell-Gewichtsdatei:")
    elif dist_type == "config":
        return title_en.replace("Configuration file:", "Konfigurationsdatei:")
    elif dist_type == "tokenizer":
        return title_en.replace("Tokenizer file:", "Tokenizer-Datei:")
    elif dist_type == "additional":
        return title_en.replace("File:", "Datei:")
    return title_en

def run_converter(
        input_path: Path,
        output_dir: Optional[Path] = Path("output"),
        output_base: Optional[str] = None, 
        base_uri: str = "https://piveau.io/set",
        profile: Profile = Profile.DCAT_AP,
        output_format: Optional[List[Union[str, OutputFormat]]] = None,
        enable_translation: bool = True
    ) -> List[Path]:
        """
        Convert Hugging Face datasets/models metadata to DCAT-AP RDF.

        Args:
            input_path (Path): Path to the input JSON file.
            output_dir (Path, optional): Directory where output files will be saved (default: ./output).
            output_base (str, optional): Optional base filename (without extension). If omitted, a name is auto-generated.
            base_uri (str): Base URI used as namespace for generated resources.
            profile: DCAT application profile (default: Profile.DCAT_AP.) 
            output_format (list of OutputFormat), optional Formats to export to (default: OutputFormat.RDFXML and OutputFormat.TURTLE).
            enable_translation (bool, optional): Whether to enable translation of text fields (default True).

        Returns:
            A list of paths to all created output files.
        """
        try:
            fmt_list: List[OutputFormat] = []
            if output_format:
                for fmt in output_format:
                    fmt_list.append(fmt if isinstance(fmt, OutputFormat) else OutputFormat[fmt])
            else:
                fmt_list = [OutputFormat.RDFXML, OutputFormat.TURTLE]

            converter = HFToDCATConverter(
                base_uri=base_uri,
                profile=profile,
                enable_translation=enable_translation
            )

            created_files = converter.run_parallel(
                input_path=input_path,
                output_dir=output_dir,
                output_base=output_base,
                output_format=fmt_list
            )

            if created_files:
                files_list = "\n    - ".join(f.name for f in created_files)
                logger.info(
                    f"✅ Successfully generated {len(created_files)} DCAT RDF output files in: {output_dir}\n"
                    f"    - {files_list}"
                )
            else:
                logger.error("❌ Conversion completed but no RDF output files were created.")

            return created_files
        
        except FileNotFoundError as e:
            logger.error(f"❌ Conversion failed due to error: {e}")
            return []

        except Exception as e:
            logger.exception(f"❌ Conversion failed: {e}")
            return []

