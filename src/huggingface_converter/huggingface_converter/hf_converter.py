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
from rdflib import Graph, URIRef, Literal, Namespace, BNode, Node
from rdflib.namespace import DCAT, DCTERMS, FOAF, RDF, XSD, SKOS, PROV, RDFS, OWL
from deep_translator import GoogleTranslator
from uuid import uuid4

from .enums import Profile, OutputFormat
from .shacl_validator import SHACLValidator, SHACLProfile
from .constants import (
    SCHEMA, DCATAP, DCATDE, ADMS, VCARD, MLS, IT6, LPWC, LPWCC, MLSO, CR,
    RESOURCE_CONFIG, METRICS, LANG_CODE_MAPPINGS, HF_TASKS, 
    LANG_LABELS_MULTI
)
from .vocabulary_manager import VocabularyManager
from .translation_manager import TranslationManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ML_TASK_TYPES_TTL = BASE_DIR.parent / "mlso_ttl" / "mlso_ml_task_types_v2.ttl"
ML_ALGORITHMS_TTL = BASE_DIR.parent / "mlso_ttl" / "mlso_ml_algorithms.ttl"
ML_FIELD_TTL = BASE_DIR.parent / "mlso_ttl" / "mlso_ml_fields.ttl"


class HFToDCATConverter:

    def __init__(
        self,
        base_uri: str = "https://data.example.org/",
        profile: Profile = Profile.DCAT_AP,
        default_format: OutputFormat = OutputFormat.TURTLE,
        enable_translation: bool = True,
        validate_flag: bool = True, 
        add_public_keyword: bool = False
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
        self.add_public_keyword = add_public_keyword
        
        # self.mlso_graph = Graph()
        # self.mlso_graph.parse(ML_TASK_TYPES_TTL, format="turtle")
        self.mlso_task_lookup = self._build_mlso_task_lookup(ML_TASK_TYPES_TTL)
        self.mlso_algorithm_lookup = self._build_mlso_algorithm_lookup(ML_ALGORITHMS_TTL)
        self.mlso_field_lookup = self._build_mlso_field_lookup(ML_FIELD_TTL)

        self.MODEL_IMPLEMENTATION = URIRef(f"{self.base_uri}def/ModelImplementation")

        # self.HAS_IMPLEMENTATION = URIRef(f"{self.base_uri}def/hasImplementation")
        # self.HAS_PROCESSOR = URIRef(f"{self.base_uri}def/hasProcessor")
        # self.HAS_MODEL_TYPE = URIRef(f"{self.base_uri}def/hasModelType")
        # self.HAS_EXECUTION_TASK = URIRef(f"{self.base_uri}def/hasExecutionTask")
      
        self.PYTHON_MODULE = URIRef(f"{self.base_uri}def/pythonModule")
        self.PYTHON_CLASS = URIRef(f"{self.base_uri}def/pythonClass")

        self.translation_cache = {}

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
        g.bind("lpwc", LPWC)
        g.bind("lpwcc", LPWCC)
        g.bind("mlso", MLSO)
        g.bind("phf", Namespace(f"{self.base_uri}def/"))
        # g.bind("cr", CR)
   
    def _load_hf_metadata(self, json_path: Union[str, Path]) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get("metadata", {})
        status = metadata.get("status")

        if status == "failed":
            raise RuntimeError(
                metadata.get("error", f"Fetch operation failed for {json_path}")
            )

        if status == "empty":
            raise ValueError(
                metadata.get("message", f"No datasets or models were kept in {json_path}")
            )

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
        # return BNode(f"bn_{resource_uri_id}{f'_{suffix}' if suffix else ''}")
        # sanitize dataset/model id 
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", resource_uri_id)

        return BNode(f"bn_{safe_id}{f'_{suffix}' if suffix else ''}")

    
    def _slugify_lookup_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")

    def _build_mlso_skos_lookup(
        self,
        ttl_path: str,
        *,
        rdf_types: tuple[URIRef, ...] = (SKOS.Concept, OWL.NamedIndividual),
        restrict_to_collection: URIRef | None = None,
        include_alt_labels: bool = True,
        include_uri_fragment: bool = True,
    ) -> dict[str, URIRef]:
        g = Graph()
        g.parse(ttl_path, format="turtle")

        if restrict_to_collection:
            subjects = set(g.objects(restrict_to_collection, SKOS.member))
        else:
            subjects = set()
            for rdf_type in rdf_types:
                subjects.update(g.subjects(RDF.type, rdf_type))

        lookup: dict[str, URIRef] = {}

        for s in subjects:
            labels = list(g.objects(s, SKOS.prefLabel))
            if include_alt_labels:
                labels += list(g.objects(s, SKOS.altLabel))

            for label in labels:
                key = self._slugify_lookup_key(str(label))
                if key:
                    lookup.setdefault(key, s)

            if include_uri_fragment:
                fragment = str(s).rstrip("/#").split("/")[-1].split("#")[-1]
                key = self._slugify_lookup_key(fragment)
                if key:
                    lookup.setdefault(key, s)

        return lookup

    def _build_mlso_algorithm_lookup(self, ttl_path: str) -> dict[str, URIRef]:
        return self._build_mlso_skos_lookup(
            ttl_path,
            rdf_types=(SKOS.Concept,),
            restrict_to_collection=None,
        )
    
    def _build_mlso_task_lookup(self, ttl_path: str) -> dict[str, URIRef]:
        return self._build_mlso_skos_lookup(
            ttl_path,
            restrict_to_collection=URIRef(
                "http://w3id.org/mlso/vocab/ml_task/MachineLearningTask"
            ),
        )

    def _build_mlso_field_lookup(self, ttl_path: str) -> dict[str, URIRef]:
        return self._build_mlso_skos_lookup(
            ttl_path,
            restrict_to_collection=URIRef(
                "http://w3id.org/mlso/vocab/ml_field/MachineLearningField"
            ),
        )
    # def _build_mlso_task_lookup(self, ttl_path: str) -> dict:
    #     """
    #     Load MLSO machine learning task type TTL and build lookup mapping normalized_label to URIRef
    #     """
    #     g = Graph()
    #     g.parse(ttl_path, format="turtle")

    #     lookup = {}

    #     for s in g.subjects(RDF.type, OWL.NamedIndividual):
    #         label = g.value(s, SKOS.prefLabel)

    #         if label:
    #             slug = label.lower().replace(" ", "-").replace("_", "-").strip()
    #             lookup[slug] = s

    #     return lookup
    
    # def _build_mlso_algorithm_lookup(self, ttl_path: str) -> Dict[str, URIRef]:
    #     """
    #     Load MLSO machine learning algorithm type TTL and build lookup mapping normalized_label to URIRef
    #     """
    #     g = Graph()
    #     g.parse(ttl_path, format="turtle")

    #     lookup = {}

    #     for s in g.subjects(RDF.type, SKOS.Concept):
    #         labels = list(g.objects(s, SKOS.prefLabel)) + list(g.objects(s, SKOS.altLabel))

    #         for label in labels:
    #             key = str(label).strip().lower()
    #             lookup[key] = s

    #     return lookup
    
    # def _build_mlso_field_lookup(self, ttl_path: str) -> dict[str, URIRef]:
    #     g = Graph()
    #     g.parse(ttl_path, format="turtle")

    #     lookup = {}

    #     for s in g.subjects(RDF.type, OWL.NamedIndividual):
    #         labels = list(g.objects(s, SKOS.prefLabel)) + list(g.objects(s, SKOS.altLabel))

    #         for label in labels:
    #             key = re.sub(r"[^a-z0-9]+", "-", str(label).lower()).strip("-")
    #             lookup[key] = s

    #     return lookup

    def _init_ml_classes(self, g: Graph):
        """Initialize custom ML implementation subclasses """
      
        if (self.MODEL_IMPLEMENTATION, RDF.type, RDFS.Class) not in g:
            g.add((self.MODEL_IMPLEMENTATION, RDF.type, RDFS.Class))
            g.add((self.MODEL_IMPLEMENTATION, RDFS.subClassOf, MLS.Implementation))
            g.add((self.MODEL_IMPLEMENTATION, RDFS.label, Literal("Model Implementation", lang="en")))
            g.add((self.MODEL_IMPLEMENTATION, RDFS.comment,
               Literal("Concrete implementation of a model architecture (e.g., BertForMaskedLM).", lang="en")))
            
            if self.enable_translation:
                g.add((self.MODEL_IMPLEMENTATION, RDFS.label, Literal("Modellimplementierung", lang="de")))
                g.add((self.MODEL_IMPLEMENTATION, RDFS.comment,
                    Literal("Konkrete Implementierung einer Modellarchitektur (z. B. BertForMaskedLM).", lang="de")))

    def _init_ml_properties(self, g: Graph):
        """Initialize ML-related properties """

        def add_property(prop, label_en, comment_en, label_de=None, comment_de=None):
            if (prop, RDF.type, RDF.Property) not in g:
                g.add((prop, RDF.type, RDF.Property))
                g.add((prop, RDFS.label, Literal(label_en, lang="en")))
                g.add((prop, RDFS.comment, Literal(comment_en, lang="en")))

                if self.enable_translation and label_de:
                    g.add((prop, RDFS.label, Literal(label_de, lang="de")))
                    if comment_de:
                        g.add((prop, RDFS.comment, Literal(comment_de, lang="de")))

        # add_property(
        #     self.HAS_MODEL_TYPE,
        #     "has model family",
        #     "Links a model to its model family (e.g., BERT, GPT), derived directly from the Hugging Face metadata field model_type.",
        #     "hat Modellfamilie",
        #     "Verknüpft ein Modell mit seiner Modellfamilie (z. B. BERT, GPT), die direkt aus dem Hugging-Face-Metadatenfeld model_type abgeleitet wird."
        # )

        # add_property(
        #     self.HAS_EXECUTION_TASK,
        #     "has execution task",
        #     "Links a model to an execution-level task (e.g., feature-extraction) derived from the Hugging Face metadata field transformersInfo.pipeline_tag.",
        #     "hat Ausführungsaufgabe",
        #     "Verknüpft ein Modell mit einer laufzeitspezifischen Aufgabe (z. B. Feature-Extraction), die aus dem Hugging-Face-Metadatenfeld transformersInfo.pipeline_tag abgeleitet wird."
        # )

        # add_property(
        #     self.HAS_PROCESSOR,
        #     "has processor",
        #     "Links a model to its preprocessing component (e.g., tokenizer or feature extractor) derived from the Hugging Face metadata field transformersInfo.processor.",
        #     "hat Prozessor",
        #     "Verknüpft ein Modell mit seiner Vorverarbeitungskomponente (z. B. Tokenizer oder Feature-Extraktor), die aus dem Hugging-Face-Metadatenfeld transformersInfo.processor abgeleitet wird."
        # )

        add_property(
            self.PYTHON_MODULE,
            "python module",
            "Specifies the Python module used to load or execute a model implementation (e.g., from transformers or custom code).",
            "Python-Modul",
            "Gibt das Python-Modul an, das zum Laden oder Ausführen einer Modellimplementierung verwendet wird (z. B. aus transformers oder benutzerdefiniertem Code)."
        )

        add_property(
            self.PYTHON_CLASS,
            "python class",
            "Specifies the Python class used to load or execute a model implementation (e.g., AutoModel or custom classes) derived from Hugging Face metadata.",
            "Python-Klasse",
            "Gibt die Python-Klasse an, die zum Laden oder Ausführen einer Modellimplementierung verwendet wird (z. B. AutoModel oder benutzerdefinierte Klassen), abgeleitet aus Hugging-Face-Metadaten."
        )

    def _init_ml_skos(self, g: Graph):
        custom_schemes = {
            "hf-task-type": {
                "en": "Hugging Face Task Types",
                "de": "Hugging Face Aufgabentypen"
            },
            "hf-modality": {
                "en": "Hugging Face Modalities",
                "de": "Hugging Face Modalitäten"
            },
            "hf-size": {
                "en": "Hugging Face Size Categories",
                "de": "Hugging Face Größenkategorien"
            },
            "hf-task-category": {
                "en": "Hugging Face Task Categories",
                "de": "Hugging Face Aufgabenkategorien"
            }
        }

        for slug, labels in custom_schemes.items():
            uri = URIRef(f"{self.base_uri}def/{slug}")

            g.add((uri, RDF.type, SKOS.ConceptScheme))

            for lang, label in labels.items():
                g.add((uri, SKOS.prefLabel, Literal(label, lang=lang)))
                g.add((uri, RDFS.label, Literal(label, lang=lang)))
                g.add((uri, DCTERMS.title, Literal(label, lang=lang)))

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

            # ML_LIBRARY = URIRef(f"{self.base_uri}def/MLLibrary")
            # MODEL_IMPLEMENTATION = URIRef(f"{self.base_uri}def/ModelImplementation")

        elif  resource_type == "dataset":
            g.add((dataset_uri, RDF.type, MLS.Dataset)) 
            croissant_meta = metadata.get("croissant")
            license_meta = metadata.get("license")
            if croissant_meta and isinstance(croissant_meta, dict):
                self._add_croissant(g, dataset_uri, resource_id, croissant_meta, license_meta)

        # Add basic metadata with translations
        self._add_basic_metadata(g, dataset_uri, metadata, resource_id, resource_type)
        
        # Add controlled vocabulary terms
        self._add_controlled_vocabulary_terms(g, dataset_uri, metadata)

        # Add metrics
        self._add_metrics(g, dataset_uri, resource_id, resource_type, metadata)
        
        self._add_citations_documentation(g, dataset_uri, resource_type, metadata)

        # Add creator info
        self._add_creator_info(g, dataset_uri, resource_id, metadata)
        # Add publisher info
        self._add_publisher_info(g, dataset_uri, resource_id)

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
    
    def _normalize_croissant_urls(self, g: Graph) -> None:
        """Convert string URLs in Croissant RDF to URIRefs where appropriate."""

        URL_PREDICATES = {
            SCHEMA.contentUrl,
            SCHEMA.url,
        }

        for s, p, o in list(g.triples((None, None, None))):
            if p in URL_PREDICATES and isinstance(o, Literal):
                url = str(o).strip()

                if url.startswith("http"):
                    g.remove((s, p, o))
                    g.add((s, p, URIRef(url)))
    
    def _normalize_schema_org(self, g: Graph):
        """ Normalize schema in Croissant RDF """
        SCHEMA_HTTP = "http://schema.org/"
        SCHEMA_HTTPS = "https://schema.org/"

        for s, p, o in list(g):
            s_new = s
            p_new = p
            o_new = o

            if isinstance(s, URIRef) and str(s).startswith(SCHEMA_HTTP):
                s_new = URIRef(str(s).replace(SCHEMA_HTTP, SCHEMA_HTTPS))

            if isinstance(p, URIRef) and str(p).startswith(SCHEMA_HTTP):
                p_new = URIRef(str(p).replace(SCHEMA_HTTP, SCHEMA_HTTPS))

            if isinstance(o, URIRef) and str(o).startswith(SCHEMA_HTTP):
                o_new = URIRef(str(o).replace(SCHEMA_HTTP, SCHEMA_HTTPS))

            if (s_new, p_new, o_new) != (s, p, o):
                g.remove((s, p, o))
                g.add((s_new, p_new, o_new))
    
    def _replace_croissant_blank_nodes(self, graph: Graph, base_uri: URIRef) -> Graph:
        """ Replace all blank nodes in croissant rdf """

        base = str(base_uri).rstrip("/") + "/"

        # Assign stable local IDs to blank nodes ---
        bnode_ids = {}
        counter = 0

        def get_bnode_id(b):
            nonlocal counter
            if b not in bnode_ids:
                counter += 1
                bnode_ids[b] = counter
            return bnode_ids[b]

        def normalize_term(term):
            if isinstance(term, Literal):
                if term.datatype:
                    return f"literal:{term.value}:{term.datatype}"
                elif term.language:
                    return f"literal:{term.value}:{term.language}"
                return f"literal:{term.value}"
            elif isinstance(term, URIRef):
                return f"uri:{str(term)}"
            elif isinstance(term, BNode):
                return f"bnode:{get_bnode_id(term)}"
            else:
                return str(term)

        # Collect context of each blank node 
        bnode_contexts = {}

        for s, p, o in graph:
            if isinstance(s, BNode):
                bnode_contexts.setdefault(s, []).append(("subject", p, o))
            if isinstance(o, BNode):
                bnode_contexts.setdefault(o, []).append(("object", s, p))

        # Generate stable URIs
        bnode_map = {}

        for bnode, contexts in bnode_contexts.items():
            context_items = []

            for role, p, term in contexts:
                context_items.append((
                    role,
                    str(p),
                    normalize_term(term)
                ))

            # Deterministic hash
            context_str = json.dumps(sorted(context_items), sort_keys=True)
            hash_val = hashlib.sha256(context_str.encode()).hexdigest()[:12]

            stable_uri = URIRef(f"{base}node/{hash_val}")
            bnode_map[bnode] = stable_uri

        # Rewirte graph
        new_graph = Graph()

        # Preserve namespace bindings
        for prefix, namespace in graph.namespaces():
            new_graph.bind(prefix, namespace)

        # Bind croissant namespace if desired
        new_graph.bind("crdf", base_uri)

        for s, p, o in graph:
            new_s = bnode_map.get(s, s)
            new_o = bnode_map.get(o, o)
            new_graph.add((new_s, p, new_o))

        return new_graph

    def _add_croissant(self, g: Graph, subject: URIRef, dataset_id: str, croissant_meta: Dict[str, Any], license_meta):

        # Add croissant as distribution
        dist_uri = URIRef(f"{subject}/distribution/croissant")
  
        g.add((subject, DCAT.distribution, dist_uri))
        g.add((dist_uri, RDF.type, DCAT.Distribution))

        license_uri = self._process_license(g, license_meta)   
        if license_uri:
            g.add((dist_uri, DCTERMS.license, license_uri))

        g.add((dist_uri, DCTERMS.title, Literal("Croissant metadata (JSON-LD)", lang="en")))
        g.add((
            dist_uri, 
            DCTERMS.description, 
            Literal("Dataset metadata following the Croissant (MLCommons) schema in JSON-LD format.",
                lang="en"
            )
        ))
        if self.enable_translation:
            g.add((dist_uri, DCTERMS.title, Literal("Croissant-Metadaten (JSON-LD)", lang="de")))
            g.add((
                dist_uri,
                DCTERMS.description,
                Literal(
                    "Datensatzmetadaten gemäß dem Croissant-Standard (MLCommons) im JSON-LD-Format.", 
                    lang="de"
                )
            ))

        croissant_uri = URIRef(f"https://huggingface.co/api/datasets/{dataset_id}/croissant")
        # Add access URL for croissant metadata JSON-LD
        g.add((dist_uri, DCAT.accessURL, croissant_uri))
    
        # Add format .jsonld 
        self._add_file_media_type(g, dist_uri, ".jsonld")
        g.add((dist_uri, DCTERMS.format, URIRef("http://publications.europa.eu/resource/authority/file-type/JSON_LD")))

        # conformsTo (from croissant)
        conforms_to = croissant_meta.get("conformsTo")
        if not conforms_to:
            conforms_to = "https://mlcommons.org/croissant/1.1" 

        conform_uri = URIRef(conforms_to)
        g.add((dist_uri, DCTERMS.conformsTo, conform_uri))       
        # g.add((subject, DCTERMS.conformsTo, conform_uri))

        ml_version = conforms_to.rstrip("/").split("/")[-1]
        ml_label = f"MLCommons Croissant {ml_version}" if ml_version else "MLCommons Croissant"

        if (conform_uri, RDF.type, DCTERMS.Standard) not in g:
            g.add((conform_uri, RDF.type, DCTERMS.Standard))
            g.add((conform_uri, RDFS.label, Literal(ml_label, lang="en")))

            if self.enable_translation:
                g.add((conform_uri, RDFS.label, Literal(ml_label, lang="de")))
        
        # try:
        #     croissant_graph = Graph()
        #     croissant_rdf_base_uri = URIRef(f"{subject}/croissant/")
        #     croissant_rdf_uri = URIRef(croissant_rdf_base_uri.rstrip("/"))
        
        #     croissant_graph.bind("crdf", croissant_rdf_base_uri)
        #     croissant_meta_enriched = dict(croissant_meta)
        #     # Add @id for identification
        #     croissant_meta_enriched["@id"] = str(croissant_rdf_uri)

        #     croissant_graph.parse(
        #         data=json.dumps(croissant_meta_enriched),
        #         format="json-ld",
        #         base=croissant_rdf_base_uri
        #     )

        #     self._normalize_schema_org(croissant_graph)
        #     self._normalize_croissant_urls(croissant_graph)

        #     # # Look for blank nodes in cross graph
        #     # blank_nodes = [s for s in croissant_graph.subjects() if isinstance(s, BNode)]
        #     # if blank_nodes:
        #     #     croissant_graph = self._replace_croissant_blank_nodes(croissant_graph, croissant_rdf_base_uri)
        #     #     blank_nodes = [s for s in croissant_graph.subjects() if isinstance(s, BNode)]
        #     #     print(f"blank nodes: {blank_nodes}")

        #     if len(croissant_graph) > 0:
        #         for triple in croissant_graph:
        #             g.add(triple)
        #         g.add((subject, DCTERMS.relation, croissant_rdf_uri))                        
        #         g.add((croissant_rdf_uri, DCTERMS.hasFormat, dist_uri))
        #         g.add((dist_uri, DCTERMS.isFormatOf, croissant_rdf_uri))

        # except Exception as e:
        #     logger.exception(f"Croissant parsing failed for {dataset_id}: {e}")
    
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

        if self.add_public_keyword and "public" not in {t.lower() for t in tags}:
            g.add((subject, DCAT.keyword, Literal("public", lang="en")))

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
            repo_uri = URIRef(hub_url)
            g.add((subject, DCAT.landingPage, repo_uri))
            g.add((repo_uri, RDF.type, FOAF.Document))  
            if resource_type == "model":
                g.add((subject, IT6.hasRepository, repo_uri))
                g.add((repo_uri, RDF.type, LPWCC.Repository))
            if isinstance(title, str):
                g.add((repo_uri, RDFS.label, Literal(title, lang="en")))
                if self.enable_translation:
                    g.add((repo_uri, RDFS.label, Literal(title, lang="de")))

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
    
    def _add_skos_concept(
        self,
        g: Graph,
        subject: URIRef,
        value: str,
        scheme_slug: str,
        relation: URIRef,
        *,
        rdf_types: list = None,
        exact_match: URIRef = None,
        translate: bool = False,
    ):
        """Generic SKOS concept creator and linker for custom schemes"""

        text = value.strip()
        if not text:
            return None

        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

        concept_uri = URIRef(f"{self.base_uri}def/{scheme_slug}/{slug}")
        scheme_uri = URIRef(f"{self.base_uri}def/{scheme_slug}")

        label = re.sub(r"[-_]+", " ", text).strip()
        if not label.isupper():
            label = label[:1].upper() + label[1:]

        # Link subject to concept
        g.add((subject, relation, concept_uri))

        is_new = (concept_uri, RDF.type, SKOS.Concept) not in g

        # Add concept if not already in graph
        if is_new:
            g.add((concept_uri, RDF.type, SKOS.Concept))
            g.add((concept_uri, SKOS.prefLabel, Literal(label, lang="en")))
            g.add((concept_uri, RDFS.label, Literal(label, lang="en")))
            g.add((concept_uri, SKOS.inScheme, scheme_uri))

            # Optional semantic typing 
            if rdf_types:
                for t in rdf_types:
                    g.add((concept_uri, RDF.type, t))

            # Optional translation
            if translate and self.enable_translation:
                translated = None 
                if label in self.translation_cache:
                    translated = self.translation_cache[label]
                else:  
                    try: 
                        translated = self.translator.translate_text(label)
                        self.translation_cache[label] = translated
                    except Exception:
                        pass
                
                if translated and translated != label:
                    g.add((concept_uri, SKOS.prefLabel, Literal(translated, lang="de")))
                    g.add((concept_uri, RDFS.label, Literal(translated, lang="de")))

        # Optional external mapping
        if exact_match:
            g.add((concept_uri, SKOS.exactMatch, exact_match))

        return concept_uri

    # def _add_task(self, g: Graph, subject: URIRef, task: str, relation: URIRef, source:Optional[str|None]):
    #     """Create MLS Task instance and SKOS task type with optional MLSO alignment."""

    #     task_text = task.strip()
    #     if not task_text:
    #         return

    #     task_slug = re.sub(r"[^a-z0-9]+", "-", task_text.lower()).strip("-")
    #     task_node = URIRef(f"{self.base_uri}def/task/{task_slug}")

    #     # Link model to task instance
    #     g.add((subject, relation, task_node))
    #     g.add((task_node, RDF.type, MLS.Task))

    #     # Lookup MLSO
    #     mlso_concept = self.mlso_task_lookup.get(task_slug)

    #     # Create SKOS task type
    #     concept_uri = self._add_skos_concept(
    #         g,
    #         subject=task_node, 
    #         value=task,
    #         scheme_slug="hf-task-type",
    #         relation=MLSO.hasTaskType,
    #         exact_match=mlso_concept,
    #         translate=True,
    #     )

    #     if source == "hf_pipeline_tag" and concept_uri:
    #         if (concept_uri, RDFS.comment, None) not in g:
    #             g.add((concept_uri, RDFS.comment, Literal(
    #                 "Hugging Face pipeline_tag describing the intended use or main task of the model.",
    #                 lang="en"
    #             )))

    def _slug_to_mlso_camel(self, slug: str) -> str:
        """Convert HF task category slug to MLSO ML field format."""

        special = {
            "dna": "DNA",
        }

        parts = re.split(r"[-_\s]+", slug.strip())
        return "".join(
            special.get(part.lower(), part[:1].upper() + part[1:])
            for part in parts
            if part
        )

    def _add_task(
        self,
        g: Graph,
        subject: URIRef,
        task: str,
        relation: URIRef,
        source: str | None = None,
        source_data: URIRef | list[URIRef] | None = None,
    ) -> tuple[URIRef | None, URIRef | None]:
        """Create Task instance and add Task Type ."""

        task_text = task.strip()
        if not task_text:
            return None, None

        task_slug = re.sub(r"[^a-z0-9]+", "-", task_text.lower()).strip("-")
        task_node = URIRef(f"{self.base_uri}def/hf-task/task-{task_slug}")

        task_info = HF_TASKS.get(task_slug)
        task_label = task_info["label"] if task_info else task_text

        # Link resource to task instance
        g.add((subject, relation, task_node))
        g.add((task_node, RDF.type, IT6.Task))
        g.add((task_node, DCTERMS.title, Literal(task_label, lang="en")))
        if self.enable_translation:
            g.add((task_node, DCTERMS.title, Literal(task_label, lang="de")))
      
        # if source_data:
        #     source_data_items = (
        #         source_data if isinstance(source_data, list) else [source_data]
        #     )
        #     valid_dataset_uris = [
        #         ds for ds in source_data_items if isinstance(ds, URIRef)
        #     ]

        #     # Only add sourceData when a single dataset is clearly associated
        #     # Avoid overclaiming when the model was trained on more than one dataset 
        #     # (task directly associated with a specific dataset)
        #     if len(valid_dataset_uris) == 1:
        #         g.add((task_node, IT6.sourceData, valid_dataset_uris[0]))

            # for dataset_uri in source_data_items:
            #     if isinstance(dataset_uri, URIRef):
            #         g.add((task_node, IT6.sourceData, dataset_uri))

        source_comments = {
            "hf_pipeline_tag": (
                "Task from Hugging Face model metadata field 'pipeline_tag'. "
                "Represents the model's primary intended task."
            ),

            "hf_transformers_pipeline_tag": (
                "Task from Hugging Face model metadata field "
                "'transformersInfo.pipeline_tag'. "
                "Represents the Transformers pipeline task used to run the model."
            ),

            "hf_dataset_task_category": (
                "Task from Hugging Face dataset metadata field 'task_categories'. "
                "Represents tasks the dataset is intended or suitable for."
            ),
        }

        if source in source_comments:
            g.add((task_node, RDFS.comment, Literal(source_comments[source], lang="en")))
            if self.enable_translation:
                de_comment = {
                    "hf_pipeline_tag": (
                        "Aufgabe aus dem Hugging Face-Modellmetadatenfeld pipeline_tag abgeleitet. "
                        "Sie beschreibt die primäre vorgesehene Aufgabe des Modells."
                    ),
                    "hf_transformers_pipeline_tag": (
                        "Aufgabe aus dem Hugging Face-Feld transformersInfo.pipeline_tag abgeleitet. "
                        "Sie beschreibt die Transformers-Laufzeit-Pipeline-Aufgabe."
                    ),
                    "hf_dataset_task_category": (
                        "Aufgabe aus dem Hugging Face-Datensatzmetadatenfeld task_categories abgeleitet. "
                        "Sie beschreibt Aufgaben, für die der Datensatz vorgesehen oder geeignet ist."
                    ),
                }.get(source)

                if de_comment:
                    g.add((task_node, RDFS.comment, Literal(de_comment, lang="de")))

        # Lookup MLSO task type
        mlso_task_type = self.mlso_task_lookup.get(task_slug)

        task_type_uri = self._add_skos_concept(
            g,
            subject=task_node,
            value=task_label,
            scheme_slug="hf-task-type",
            relation=IT6.hasTaskType,
            rdf_types=[IT6.TaskType],
            exact_match=mlso_task_type,
            translate=True,
        )

        if not task_type_uri:
            return task_node, None

        g.add((task_type_uri, SKOS.notation, Literal(task_slug)))

        # # Optional MLSO compatibility
        # g.add((task_node, MLSO.hasTaskType, task_type_uri))

        # Add HF task page only for known HF task slugs
        if task_slug in HF_TASKS:
            g.add((
                task_type_uri,
                RDFS.seeAlso,
                URIRef(f"https://huggingface.co/tasks/{quote(task_slug, safe='-_')}")
            ))

        # Add HF category as SKOS broader grouping
        if task_info and task_info.get("category"):
            category_slug = task_info["category"]
            category_label = task_info.get(
                "category_label",
                category_slug.replace("-", " ").title(),
            )

            category_uri = URIRef(
                f"{self.base_uri}def/hf-task-category/{category_slug}"
            )
            category_scheme = URIRef(f"{self.base_uri}def/hf-task-category")

            if (category_uri, RDF.type, SKOS.Concept) not in g:
                g.add((category_uri, RDF.type, SKOS.Concept))
                g.add((category_uri, SKOS.prefLabel, Literal(category_label, lang="en")))
                g.add((category_uri, RDFS.label, Literal(category_label, lang="en")))
                g.add((category_uri, SKOS.notation, Literal(category_slug)))
                g.add((category_uri, SKOS.inScheme, category_scheme))

            g.add((task_type_uri, SKOS.broader, category_uri))
            g.add((category_uri, SKOS.narrower, task_type_uri))

            # Map HF task category to MLSO ML field
            camel_key = self._slug_to_mlso_camel(category_slug)
            normalized_camel_key = self._slugify_lookup_key(camel_key)

            field_uri = (
                self.mlso_field_lookup.get(normalized_camel_key)
                or self.mlso_field_lookup.get(category_slug)
                or self.mlso_field_lookup.get(camel_key)
            )

            if isinstance(field_uri, URIRef):
                g.add((task_node, MLSO.relatedToField, field_uri))
                g.add((category_uri, SKOS.exactMatch, field_uri))

        return task_node, task_type_uri
    
    def _add_modality(self, g: Graph, subject: URIRef, modality: str):
        """Create modality concept (SKOS + optional MLSO typing)."""

        self._add_skos_concept(
            g,
            subject=subject,
            value=modality,
            scheme_slug="hf-modality",
            relation=MLSO.hasModality,
            rdf_types=[MLSO.DataModality],
            translate=True,
        )
    
    def _add_size_category(self, g: Graph, subject: URIRef, size: str):
        """Create size category as SKOS concept."""

        size_text = size.strip()
        if not size_text:
            return

        # Normalize label
        clean_label = size_text.strip()

        # Case 3: 1K<n<10K
        clean_label = re.sub(r"\s*<\s*n\s*<\s*", " – ", clean_label)
        # Case 1: n<1K 
        clean_label = re.sub(r"n\s*<\s*", "< ", clean_label)
        # Case 2: n>1M 
        clean_label = re.sub(r"n\s*>\s*", "> ", clean_label)

        # Normalize spacing
        clean_label = re.sub(r"\s+", " ", clean_label).strip()


        concept_uri = self._add_skos_concept(
            g,
            subject=subject,
            value=size_text,  
            scheme_slug="hf-size",
            relation=DCTERMS.subject,
            translate=True,
        )

        if not concept_uri:
            return

        g.set((concept_uri, SKOS.prefLabel, Literal(clean_label, lang="en")))
        g.set((concept_uri, RDFS.label, Literal(clean_label, lang="en")))

        if size_text != clean_label:
            g.add((concept_uri, SKOS.altLabel, Literal(size_text, lang="en")))
    
    def _add_dataset_structured_keywords(self, g: Graph, subject: URIRef, resource_uri_id:str, metadata: Dict[str, Any], tags: List[str]) -> None:
        """Convert HF dataset modality, task_categories, task_ids, size_categories"""
        modalities = []
        size_categories = []
        libraries = []
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
            elif key == "library":
                libraries.append(value)
     
        # Add modality 
        multi = len(modalities) > 1
        for mod in modalities:
            # self._add_property(g, subject, resource_uri_id, name="modality", value=mod, category="modality", multi=multi)
            self._add_modality(g, subject, mod)

        # Add task category 
        multi = len(task_categories) > 1
        for task in task_categories:
            self._add_task(g, subject, task, LPWC.usedForTask, "hf_dataset_task_category", source_data=subject)

        # Add task id 
        multi = len(task_ids) > 1
        for tid in task_ids:
            # self._add_task(g, subject, tid, "task_id")
            self._add_property(g, subject, resource_uri_id, "task_id", tid)
        
        # Add size_category 
        multi = len(size_categories) > 1
        for size in size_categories:
            self._add_size_category(g, subject, size)
        
        # Add dataset libraries
        for lib in libraries:
            self._add_library(g, subject, resource_uri_id, lib, comment="Library used to load, access, and process datasets.")
   
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
    
    def _add_library(
        self, 
        g: Graph,
        subject: URIRef,
        resource_uri_id: str,
        library_name: Optional[str],
        comment: Optional[str] = None,
        infrastructure_label: str = "Hugging Face runtime environment"
    ) -> Optional[URIRef]:
        """Add library information."""

        if not isinstance(library_name, str):
            return None

        lib = library_name.strip()
        if not lib or lib.lower() in {"null", "none", "generic"}:
            return None

        lib_slug = quote(lib.lower(), safe="-_")
        lib_uri = URIRef(f"{self.base_uri}def/library/{lib_slug}")

        # Link model/dataset to runtime/software environment
        infra_slug = self._slugify_lookup_key(resource_uri_id)
        infra_uri = URIRef(f"{self.base_uri}def/computer-infrastructure/{infra_slug}")
        g.add((subject, DCTERMS.relation, infra_uri))

        # Define MLDCAT-AP ComputerInfrastructure node 
        if (infra_uri, None, None) not in g:
            g.add((infra_uri, RDF.type, IT6.ComputerInfrastructure))
            g.add((infra_uri, DCTERMS.title, Literal(infrastructure_label, lang="en")))
            g.add((infra_uri, RDFS.label, Literal(infrastructure_label, lang="en")))

            if self.enable_translation:
                g.add((infra_uri, DCTERMS.title, Literal("Hugging-Face-Laufzeitumgebung", lang="de")))
                g.add((infra_uri, RDFS.label, Literal("Hugging-Face-Laufzeitumgebung", lang="de")))

        # ComputerInfrastructure has Library
        g.add((infra_uri, IT6.hasLibrary, lib_uri))

        g.add((subject, SCHEMA.softwareRequirements, lib_uri))
      
        if (lib_uri, None, None) not in g:
            g.add((lib_uri, RDF.type, SCHEMA.SoftwareApplication))
            g.add((lib_uri, RDF.type, IT6.Library))

            g.add((lib_uri, SCHEMA.name, Literal(lib, lang="en")))
            g.add((lib_uri, DCTERMS.title, Literal(lib, lang="en")))
            g.add((lib_uri, RDFS.label, Literal(lib, lang="en")))

            if comment:
                g.add((lib_uri, RDFS.comment, Literal(comment, lang="en")))

                if self.enable_translation:
                    if "model" in comment.lower():
                        g.add((lib_uri, RDFS.comment, Literal(
                            "Bibliothek zum Laden und Ausführen von Machine-Learning-Modellen.",
                            lang="de"
                        )))
                    elif "dataset" in comment.lower():
                        g.add((lib_uri, RDFS.comment, Literal(
                            "Bibliothek zum Laden, Zugreifen und Verarbeiten von Datensätzen.",
                            lang="de"
                        )))

            if self.enable_translation:
                g.add((lib_uri, DCTERMS.title, Literal(lib, lang="de")))
                g.add((lib_uri, RDFS.label, Literal(lib, lang="de")))

        return lib_uri

    def _get_high_level_architecture(
        self, 
        model_type: Optional[str], 
        architectures: Optional[list] = None, 
        auto_model: Optional[str] = None
    ) -> Optional[str]:
        """ Infer high-level model architecture using architectures, auto_model and model_type) """

        def normalize(s: str) -> str:
            return s.lower().replace("_", "").replace("-", "")

        # architectures 
        if isinstance(architectures, list):
            for arch in architectures:
                if not isinstance(arch, str):
                    continue
                a = normalize(arch)

                # Hybrid / state-space
                if any(x in a for x in ["jamba", "mamba", "rwkv"]):
                    return "Hybrid State Space Model"

                # Multimodal
                if any(x in a for x in ["llava", "blip", "clip", "visiontext"]):
                    return "Multimodal Transformer Model"

                # Whisper (special case)
                if "whisper" in a:
                    return "Encoder-Decoder Transformer"

                # Encoder-decoder 
                if any(x in a for x in ["seq2seq", "conditionalgeneration", "encoderdecoder", "visionencoderdecoder"]):
                    return "Encoder-Decoder Transformer"

                # Audio 
                if any(x in a for x in ["wav2vec", "hubert", "unispeech", "speech2text", "speechtotext"]):
                    return "Audio Transformer Model"

                # Vision
                if any(x in a for x in ["visionmodel", "visiontransformer", "vitmodel", "swin", 
                    "vit", "dinov2", "forimageclassification", "forobjectdetection", "fordepthestimation"
                ]):
                    return "Vision Transformer Model"

                # Decoder-only
                if "causallm" in a:
                    return "Decoder-only Transformer"

                # Encoder-only
                if any(x in a for x in ["maskedlm", "sequenceclassification", "tokenclassification"]):
                    return "Encoder-only Transformer"

        # auto_model 
        if isinstance(auto_model, str):
            am = normalize(auto_model)

            if any(x in am for x in ["imagetexttotext", "visualquestionanswering", "vision2seq"]):
                return "Multimodal Transformer Model"

            if "whisper" in am:
                return "Encoder-Decoder Transformer"

            if any(x in am for x in ["seq2seq", "conditionalgeneration"]):
                return "Encoder-Decoder Transformer"
            
            if any(x in am for x in ["speechseq2seq", "ctc"]):
                return "Audio Transformer Model"

            if any(x in am for x in [
                "imageclassification", "objectdetection", "depthestimation", "imagesegmentation"
            ]):
                return "Vision Transformer Model"

            if "causallm" in am:
                return "Decoder-only Transformer"

            if "maskedlm" in am:
                return "Encoder-only Transformer"

    
        # model_type 
        if not isinstance(model_type, str):
            return None

        mt = normalize(model_type)
        base = mt.split("-")[0].split("_")[0]

        # Special overrides
        if base.startswith(("jamba", "mamba", "rwkv")):
            return "Hybrid State Space Model"

        if base.startswith("apertus"):
            return "Decoder-only Transformer"

        if base.startswith("whisper"):
            return "Encoder-Decoder Transformer"

        # ===== Representative mapping sets =====
        encoder_only = {
            "bert", "roberta", "distilbert", "deberta", "albert", "electra",
            "mpnet", "longformer", "modernbert", "camembert", "xlmroberta"
        }

        decoder_only = {
            "gpt2", "gptneo", "gptneox", "llama", "mistral", "mixtral", "falcon", "mpt", "phi", 
            "phi3", "phi4", "gemma", "gemma2", "gemma3", "qwen2", "qwen3", "deepseek", "glm", 
            "opt", "granite", "cohere", "starcoder", "starcoder2","dbrx", "olmo"
        }

        encoder_decoder = {
            "t5", "mt5", "bart", "mbart", "marian", "pegasus", "ul2", "m2m100"
        }

        multimodal = {
            "clip", "blip", "blip2", "llava", "idefics", "florence",
            "internvl", "qwen2vl", "qwen3vl", "deepseekvl", "glm4v"
        }

        vision = {"vit", "swin", "swinv2", "dinov2", "sam", "beit", "siglip"}
        cnn = {"resnet", "efficientnet", "mobilenet", "resnext", "densenet"}
        audio = {"wav2vec2", "hubert", "parakeet", "speech2text"}

        # Matching
        if any(base.startswith(k) for k in multimodal):
            return "Multimodal Transformer Model"

        if any(base.startswith(k) for k in audio):
            return "Audio Transformer Model"

        if any(base.startswith(k) for k in vision):
            return "Vision Transformer Model"

        if any(base.startswith(k) for k in cnn):
            return "Convolutional Neural Network"

        if any(base.startswith(k) for k in encoder_only):
            return "Encoder-only Transformer"

        if any(base.startswith(k) for k in decoder_only):
            return "Decoder-only Transformer"

        if any(base.startswith(k) for k in encoder_decoder):
            return "Encoder-Decoder Transformer"

        return None

    def _add_library_transformers_config(
        self,
        g: Graph,
        subject: URIRef,
        resource_uri_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """ Add Hugging Face model semantic metadata for DCAT + ML loading."""

        # Link used datasets
        used_dataset_uris = []

        used_datasets = as_array(metadata.get("datasets"))
        if used_datasets:
            used_dataset_uris = self._link_used_datasets(
                g,
                subject,
                resource_uri_id,
                used_datasets,
            )

        # Add library_name 
        lib_uri = self._add_library(
            g, 
            subject, 
            resource_uri_id, 
            metadata.get("library_name"), 
            comment="Library used to load and run machine learning models."
        )

        # Add pipeline_tag as task 
        root_pipeline = metadata.get("pipeline_tag")
        if isinstance(root_pipeline, str) and root_pipeline.strip():
            # LPWC.usedForTask is defined as the relation between dataset and task but here use it also for model because of strong semantic match
            self._add_task(g, subject, root_pipeline, LPWC.usedForTask, "hf_pipeline_tag", source_data=used_dataset_uris) 
        
        config = metadata.get("config", {})
        transformers_info = metadata.get("transformers_info", {})
        model_type = config.get("model_type") if isinstance(config, dict) else None
        architectures = config.get("architectures") if isinstance(config, dict) else None
        auto_model = transformers_info.get("auto_model") if isinstance(transformers_info, dict) else None
       
        model_type_uri = None 
        algo_uri = None 
        if isinstance(model_type, str) and model_type.strip():
            mt = model_type.strip()
            # key = mt.lower()
            key = self._slugify_lookup_key(mt)

            slug = quote(key, safe="-_")
            model_type_uri = URIRef(f"{self.base_uri}def/model-type/{slug}")

            if (model_type_uri, None, None) not in g:
                g.add((model_type_uri, RDF.type, SKOS.Concept))
                g.add((model_type_uri, SKOS.prefLabel, Literal(mt, lang="en")))
                g.add((model_type_uri, RDFS.label, Literal(mt, lang="en")))
                g.add((model_type_uri, SKOS.notation, Literal(key)))

                # Semantic interpretation
                g.add((
                    model_type_uri,
                    SKOS.scopeNote,
                    Literal(
                    "Represents a Hugging Face model family specified in "
                    "the config.model_type field.", 
                        lang="en",
                    )
                ))

                # Metadata provenance
                # g.add((model_type_uri, DCTERMS.source, Literal("config.model_type"))) # DCATAP dct:source not applicable here 
            
                if self.enable_translation:
                    g.add((model_type_uri, SKOS.prefLabel, Literal(mt, lang="de")))
                    g.add((model_type_uri, RDFS.label, Literal(mt, lang="de")))

                    g.add((
                        model_type_uri,
                        SKOS.scopeNote,
                        Literal(
                            "Repräsentiert eine Hugging-Face-Modellfamilie "
                            "aus dem Feld config.model_type.",
                            lang="de",
                        )
                    ))


            # # Link model to model type:
            # g.add((subject, self.HAS_MODEL_TYPE, model_type_uri))
            # Add generic interoperable relation from model to model type
            g.add((subject, DCTERMS.relation, model_type_uri))

            algo_uri = self.mlso_algorithm_lookup.get(key)
            if isinstance(algo_uri, URIRef):
                g.add((model_type_uri, SKOS.exactMatch, algo_uri))

        # Add high level architecture
        high_level_arch = self._get_high_level_architecture(    
            model_type=model_type,
            architectures=architectures,
            auto_model=auto_model
        )
        if high_level_arch:
            g.add((subject, IT6.modelArchitecture, Literal(high_level_arch)))

        # config architectures is mls#Implementation          
        if isinstance(architectures, list):
            for arch in architectures:
                if isinstance(arch, str) and arch.strip():
                    arch_label = arch.strip()
                    arch_slug = quote(arch_label.lower(), safe="-_")
                    arch_impl_uri = URIRef(f"{self.base_uri}def/implementation/{arch_slug}")

                    # Link model to implementation 
                    # g.add((subject, self.HAS_IMPLEMENTATION, arch_impl_uri))
                    # g.add((subject, MLS.hasPart, arch_impl_uri))
                    # Link model to implementation as a generic relation 
                    g.add((subject, DCTERMS.relation, arch_impl_uri))


                    # Define implementation node
                    if (arch_impl_uri, None, None) not in g:
                        g.add((arch_impl_uri, RDF.type, MLS.Implementation))
                        g.add((arch_impl_uri, RDF.type, self.MODEL_IMPLEMENTATION))
                        g.add((arch_impl_uri, RDFS.label, Literal(arch_label, lang="en")))
                        # g.add((arch_impl_uri, DCTERMS.source, Literal("config.architectures"))) # DCATAP dct:source not applicable here 

                        g.add((arch_impl_uri, RDFS.comment, Literal(
                            "Model implementation class specified in the Hugging Face config.architectures field.",
                            lang="en",
                        )))

                        if self.enable_translation:
                            g.add((arch_impl_uri, RDFS.label, Literal(arch_label, lang="de")))
                            g.add((arch_impl_uri, RDFS.comment, Literal(
                                "Modellimplementierungsklasse aus dem Hugging-Face-Feld config.architectures.",
                                lang="de",
                            )))
                        

                    # # Link implementation to model type-related algorithm if available, otherwise use relation to link to model_type 
                    # added note: implements concept is not valid. so skip the relationship between arch_impl and algo_uri
                    # if isinstance(algo_uri, URIRef):
                    #     g.add((arch_impl_uri, MLS.implements, algo_uri))
                    # elif model_type_uri:
                    #     g.add((arch_impl_uri, DCTERMS.relation, model_type_uri))
                    if  model_type_uri:
                        g.add((arch_impl_uri, DCTERMS.relation, model_type_uri))

                    # # Link implementation to library if available
                    # if lib_uri:
                    #     g.add((arch_impl_uri, DCTERMS.isPartOf, lib_uri))
        
        # Add total number of parameters 
        safetensors = metadata.get("safetensors", {})
        params = safetensors.get("parameters", {})
        total_params = params.get("total")
        if isinstance(total_params, int) and total_params >= 0:
            g.add((subject, IT6.numberOfParameters, Literal(total_params, datatype=XSD.nonNegativeInteger)))
        
        # Add transformers info
        if isinstance(transformers_info, dict):
            auto_model = transformers_info.get("auto_model")
            custom_class = transformers_info.get("custom_class")

            def _add_loader(class_name: str, module_name: str, source_field: str, comment_en: str, comment_de: str):
                class_slug = quote(class_name.lower(), safe="-_")
                loader_uri = URIRef(f"{self.base_uri}def/implementation/{class_slug}")

                # Link model to implementation (used for loading)
                # g.add((subject, self.HAS_IMPLEMENTATION, loader_uri))
                # g.add((subject, MLS.hasPart, loader_uri))
                g.add((subject, DCTERMS.relation, loader_uri))

                # Define node
                if (loader_uri, None, None) not in g:
                    g.add((loader_uri, RDF.type, MLS.Implementation))
                    g.add((loader_uri, RDF.type, self.MODEL_IMPLEMENTATION))
                    if model_type_uri: 
                        g.add((loader_uri, MLS.implements, model_type_uri))
                    g.add((loader_uri, RDFS.label, Literal(class_name, lang="en")))
                    g.add((loader_uri, RDFS.comment, Literal(comment_en, lang="en")))
                    # g.add((loader_uri, DCTERMS.source, Literal(source_field)))  # DCATAP dct:source not applicable here 

                    if self.enable_translation:
                        g.add((loader_uri, RDFS.label, Literal(class_name, lang="de")))
                        g.add((loader_uri, RDFS.comment, Literal(comment_de, lang="de")))

                    if module_name:
                        g.add((loader_uri, self.PYTHON_MODULE, Literal(module_name)))
                        g.add((loader_uri, self.PYTHON_CLASS, Literal(class_name)))

                # # Link to library
                # if isinstance(lib_uri, Node):
                #     g.add((loader_uri, DCTERMS.isPartOf, lib_uri))

            # Add custom_class 
            if isinstance(custom_class, str) and "." in custom_class:
                module_name, class_name = custom_class.rsplit(".", 1)

                _add_loader(
                    class_name=class_name,
                    module_name=module_name,
                    source_field="transformersInfo.custom_class",
                    comment_en="Custom implementation class used to instantiate or load the model.",
                    comment_de="Benutzerdefinierte Implementierungsklasse zum Laden oder Instanziieren des Modells."
                )

            # Add auto_model (generic loader) ---
            if isinstance(auto_model, str) and auto_model.strip():
                class_name = auto_model.strip()

                _add_loader(
                    class_name=class_name,
                    module_name="transformers",
                    source_field="transformersInfo.auto_model",
                    comment_en="Hugging Face AutoModel class used to instantiate or load the model.",
                    comment_de="Hugging-Face-AutoModel-Klasse zum Laden oder Instanziieren des Modells."
                )

            # Add processor 
            processor = transformers_info.get("processor")
         
            if isinstance(processor, str) and processor.strip():
                proc = processor.strip()

                proc_slug = quote(proc.lower(), safe="-_")
                proc_uri = URIRef(f"{self.base_uri}def/processor/{proc_slug}")

                # Link model to processor
                # g.add((subject, self.HAS_PROCESSOR, proc_uri))
                # Generic interoperable relation
                g.add((subject, DCTERMS.relation, proc_uri))

                # Define processor node
                if (proc_uri, None, None) not in g:
                    g.add((proc_uri, RDF.type, SCHEMA.SoftwareApplication))
                    g.add((proc_uri, RDFS.label, Literal(proc, lang="en")))
                    # g.add((proc_uri, DCTERMS.source, Literal("transformersInfo.processor")))  # DCATAP dct:source not applicable here 

                    # Default module = transformers (safe assumption here)
                    g.add((proc_uri, self.PYTHON_MODULE, Literal("transformers")))
                    g.add((proc_uri, self.PYTHON_CLASS, Literal(proc)))

                    g.add((proc_uri, RDFS.comment, Literal(
                        "Processor specified in the transformersInfo.processor field, "
                        "used for preprocessing such as tokenization or feature extraction.",
                        lang="en",
                    )))

                    if self.enable_translation:
                        g.add((proc_uri, RDFS.label, Literal(proc, lang="de")))

                        g.add((proc_uri, RDFS.comment, Literal(
                            "Prozessor aus dem Feld transformersInfo.processor zur "
                            "Vorverarbeitung wie Tokenisierung oder Merkmalsextraktion.",
                            lang="de",
                        )))

                # # Link processor to library
                # if lib_uri:
                #     g.add((proc_uri, DCTERMS.isPartOf, lib_uri))


            t_pipeline = transformers_info.get("pipeline_tag")

            if isinstance(t_pipeline, str) and t_pipeline.strip():
                t_pipeline = t_pipeline.strip()

                same_as_root = (
                    isinstance(root_pipeline, str)
                    and t_pipeline.lower() == root_pipeline.strip().lower()
                )

                if not same_as_root:
                    self._add_task(g, subject, t_pipeline, LPWC.usedForTask, "hf_transformers_pipeline_tag")

    def _add_model_engagement(
        self,
        g: Graph,
        subject: URIRef,
        resource_uri_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Add MLDCAT-AP engagement information for Hugging Face models."""

        likes = metadata.get("likes")
        downloads = metadata.get("downloads")

        if likes is None and downloads is None:
            return

        engagement_slug = self._slugify_lookup_key(resource_uri_id)
        engagement_uri = URIRef(
            f"{self.base_uri}def/engagement/{engagement_slug}"
        )

        g.add((subject, IT6.hasEngagement, engagement_uri))
        g.add((engagement_uri, RDF.type, IT6.Engagement))

        g.add((
            engagement_uri,
            DCTERMS.title,
            Literal("Hugging Face engagement metrics", lang="en")
        ))

        if self.enable_translation:
            g.add((
                engagement_uri,
                DCTERMS.title,
                Literal("Hugging Face Engagement-Metriken", lang="de")
            ))

        if isinstance(likes, int) and likes >= 0:
            g.add((
                engagement_uri,
                IT6.like,
                Literal(likes, datatype=XSD.nonNegativeInteger)
            ))

        if isinstance(downloads, int) and downloads >= 0:
            g.add((
                engagement_uri,
                IT6.download,
                Literal(downloads, datatype=XSD.nonNegativeInteger)
            ))

            g.add((
                engagement_uri,
                RDFS.comment,
                Literal(
                    "Downloads are counted over the last 30 days",
                    lang="en"
                )
            ))
            if self.enable_translation:
                g.add((
                    engagement_uri,
                    RDFS.comment,
                    Literal(
                        "Downloads werden für die letzten 30 Tage gezählt.",
                        lang="de"
                    )
                ))

    def _add_metrics(self, g: Graph, subject: URIRef, resource_uri_id: str, resource_type: str, metadata: Dict[str, Any]):
        """Add metrics (likes, downloads) if available"""

        if resource_type == "model":
            self._add_model_engagement(g, subject, resource_uri_id, metadata)

        metric_labels = {
            "likes": {
                "en": "likes",
                "de": "Likes",
                "comment_en": "Total likes received.",
                "comment_de": "Gesamtzahl der erhaltenen Likes."
            },
            "downloads": {
                "en": "downloads",
                "de": "Downloads",
                "comment_en": "Downloads in the last 30 days.", 
                "comment_de": "Downloads der letzten 30 Tage."
            },
        }

        for i, (field, (action_term, dt)) in enumerate(METRICS.items()):
            count = metadata.get(field)
            if count is None:
                continue

            ic = self._create_bnode(resource_uri_id, f"metric_{i}")
            g.add((subject, SCHEMA.interactionStatistic, ic))
            g.add((ic, RDF.type, SCHEMA.InteractionCounter))

            label_info = metric_labels.get(field, {"en": field})
            g.add((ic, SCHEMA.name, Literal(label_info["en"], lang="en")))

            if self.enable_translation and label_info.get("de"):
                g.add((ic, SCHEMA.name, Literal(label_info["de"], lang="de")))

            if label_info.get("comment_en"):
                g.add((ic, RDFS.comment, Literal(label_info["comment_en"], lang="en")))

            if self.enable_translation and label_info.get("comment_de"):
                g.add((ic, RDFS.comment, Literal(label_info["comment_de"], lang="de")))

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
        existing_names = list(g.objects(pub_uri, FOAF.name))

        pub_id_norm = pub_id.strip().lower()
        should_add = True

        for name in existing_names:
            name_str = str(name).strip()
            name_norm = name_str.lower()

            # Case 1: existing name is longer 
            if len(name_str) > len(pub_id):
                should_add = False
                break

            # Case 2: same length AND same (case-insensitive) 
            if len(name_str) == len(pub_id) and name_norm == pub_id_norm:
                should_add = False
                break

        if should_add:
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
                        # (arxiv_uri, RDF.type, RDFS.Resource, g),
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
                        # (doi_uri, RDF.type, RDFS.Resource, g),
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
    
    def _add_croissant_creators(self, g: Graph, subject: URIRef, creators):
        """ Add creator info based on croissant creator metadata """

        if isinstance(creators, dict):
            creators = [creators]

        for creator in creators:

            if not isinstance(creator, dict):
                continue

            name = creator.get("name")
            url = creator.get("url")
            ctype = creator.get("@type")

            if not name:
                continue

            name = name.strip()

            if url:
                creator_node = URIRef(url)
            else:
                creator_node = BNode()

            # Link to dataset
            g.add((subject, DCTERMS.creator, creator_node))
            g.add((creator_node, RDF.type, FOAF.Agent))
            if ctype == "Person":
                g.add((creator_node, RDF.type, FOAF.Person))
            elif ctype == "Organization":
                g.add((creator_node, RDF.type, FOAF.Organization))
         
            # Name
            g.add((creator_node, FOAF.name, Literal(name)))

            # Homepage 
            if url:
                g.add((creator_node, FOAF.homepage, URIRef(url)))
            
    def _add_creator_info(self, g: Graph, subject: URIRef, resource_id: str, metadata: Dict[str, Any]):
        
        croissant = metadata.get("croissant")
        croissant_creators = croissant.get("creator") if isinstance(croissant, dict) else None
        if croissant_creators:
            self._add_croissant_creators(g, subject, croissant_creators)
        else:
            creator_id = resource_id.split("/", 1)[0].strip()

            if not creator_id:
                return

            creator_path = quote(creator_id, safe="-_.")
            creator_uri = URIRef(f"https://huggingface.co/{creator_path}")

            g.add((subject, DCTERMS.creator, creator_uri))
            g.add((creator_uri, RDF.type, FOAF.Agent))
            g.add((creator_uri, FOAF.name, Literal(creator_id)))
            g.add((creator_uri, FOAF.homepage, creator_uri))

            # if self.profile == Profile.DCAT_AP_DE:
            #     g.add((creator_uri, RDF.type, SKOS.Concept))
            #     g.add((creator_uri, SKOS.prefLabel, Literal("Creator", lang="en")))
            #     if self.enable_translation: 
            #         g.add((creator_uri, SKOS.prefLabel, Literal("Ersteller", lang="de")))    

    def _add_provenance(self, g: Graph, subject: URIRef, resource_id: str):
        # Add provenance
        prov = self._create_bnode(resource_id, "prov")
        g.add((subject, DCTERMS.provenance, prov))
        g.add((prov, RDF.type, DCTERMS.ProvenanceStatement))
        g.add((prov, RDFS.label, Literal("The metadata was harvested from the Hugging Face platform.", lang="en")))
        if self.enable_translation: 
            g.add((prov, RDFS.label, Literal("Die Metadaten wurde von der Hugging Face-Plattform geharvestet.", lang="de")))
      
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
        HF_FORMAT_URI = URIRef(f"{self.base_uri}def/file-type/repository") # self defined uri
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
        
        # # Link used datasets
        # used_datasets = as_array(metadata.get("datasets"))
        # if used_datasets:
        #     self._link_used_datasets(g, subject, resource_id, used_datasets)
        
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
        dataset_uris: list[URIRef] = []

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
            # g.add((subject, PROV.used, dataset_uri))  
            dataset_uris.append(dataset_uri)

            g.add((subject, IT6.trainedOn, dataset_uri))
            # g.add((dataset_uri, RDF.type, MLS.Dataset))
            # g.add((dataset_uri, RDF.type, DCAT.Dataset))
              
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
        
        return dataset_uris
    
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
                enable_translation=self.enable_translation,
                add_public_keyword=self.add_public_keyword
            )
            thread_converter.convert(thread_g, resource_type, item_data)
            return thread_g
        except Exception as e:
            logger.exception(f"Failed to convert {item_data.get('id')}: {str(e)}")
            return None

    def _needs_ml_classes(self, g):
        """ Check whether ModelImplementation class is used in the graph. """
        return (None, RDF.type, self.MODEL_IMPLEMENTATION) in g

    def _needs_ml_properties(self, g):
        return any(
            p in {
                # self.HAS_IMPLEMENTATION,
                # self.HAS_PROCESSOR,
                self.PYTHON_MODULE,
                self.PYTHON_CLASS
            }
            for _, p, _ in g
        )

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
        
        try:
            fetched = self._load_hf_metadata(input_path)
        except ValueError as e:
            logger.warning(str(e))
            return []

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
            self._init_ml_skos(merged_graph)
       
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
            
            
            # Remove the code that adds standalone vocabulary definitions to catalog RDF.
            # if self._needs_ml_classes(merged_graph):
            #     self._init_ml_classes(merged_graph)

            # if self._needs_ml_properties(merged_graph):
            #     self._init_ml_properties(merged_graph)

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
                # merged_graph.serialize(destination=str(full_path), format=fmt.value)
                if fmt == OutputFormat.JSONLD:
                    context = build_context()

                    merged_graph.serialize(
                        destination=str(full_path),
                        format="json-ld",
                        context=context,
                        indent=2,
                        auto_compact=True
                    )
                else:
                    merged_graph.serialize(
                        destination=str(full_path),
                        format=fmt.value
                    )
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

def build_context() -> dict:
    return {
        "@context": {
            "dcat": "http://www.w3.org/ns/dcat#",
            "dct": "http://purl.org/dc/terms/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "schema": "https://schema.org/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "prov": "http://www.w3.org/ns/prov#",
            "vcard": "http://www.w3.org/2006/vcard/ns#",
            "dcatap": "http://data.europa.eu/r5r/",
            "dcatde": "http://dcat-ap.de/def/dcatde/",
            "adms": "http://www.w3.org/ns/adms#",
            "mls": "http://www.w3.org/ns/mls#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "it6": "http://data.europa.eu/it6/",
            "lpwcc": "https://linkedpaperswithcode.com/class/",
        }
    }

def run_converter(
        input_path: Path,
        output_dir: Optional[Path] = Path("output"),
        output_base: Optional[str] = None, 
        base_uri: str = "https://piveau.io/set",
        profile: Profile = Profile.DCAT_AP,
        output_format: Optional[List[Union[str, OutputFormat]]] = None,
        enable_translation: bool = True,
        add_public_keyword: bool = False
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
            add_public_keyword (bool, optional): Whether to inject dcat:keyword "public" into all generated dataset/model records.

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
                enable_translation=enable_translation, 
                add_public_keyword=add_public_keyword
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

