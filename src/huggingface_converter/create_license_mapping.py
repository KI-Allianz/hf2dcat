import logging
import json
from typing import Set, Dict, Any, Optional, Union
from huggingface_hub import HfApi
from rdflib import Graph, RDF, SKOS
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_hf_unique_licenses(
    limit: int = 5000,
    params: Optional[Dict[str, Any]] = {'sort': 'downloads', 'direction': -1}
) -> Set[str]:
    """
    Get unique licenses from Hugging Face models and datasets.
    
    Args:
        limit: Maximum number of items to check (default: 5000)
        params: Dictionary of API parameters (e.g., {'sort': 'downloads', 'direction': -1})
        
    Returns:
        Set of unique license strings
    """
    try: 
        api = HfApi()
        params = params or {'sort': 'downloads', 'direction': -1}
        
        def _extract_licenses(items) -> Set[str]:
            licenses = set()
            for item in items:
                tags = getattr(item, "tags", [])
                if isinstance(tags, list):
                    for tag in tags:
                        if tag.startswith("license:"):
                            licenses.add(tag.split(":", 1)[1].strip().lower())    
                elif isinstance(tags, str) and tags.startswith("license:"):
                    licenses.add(tags.split(":", 1)[1].strip().lower())
            return licenses
        
        models = api.list_models(limit=limit, **params)
        datasets = api.list_datasets(limit=limit, **params)
        return _extract_licenses(models) | _extract_licenses(datasets)
    except Exception as e:
        logger.exception(f"Failed to fetch Hugging Face licenses: {e}")
        return set()
    

def load_rdf_graph(data_dir: Union[Path, str] = Path("./license_code_table")) -> Graph:
    """Load RDF file from specified directory."""
    g = Graph()
    try:
        rdf_path = Path(data_dir) / "licences-skos.rdf"
        if not rdf_path.exists():
            raise FileNotFoundError(f"RDF file not found at {rdf_path}")
        g.parse(rdf_path, format="xml")
    except Exception as e:
        logger.exception(f"Failed to load RDF graph: {e}")
    
    return g

def create_eu_license_mapping(g: Graph) -> Dict[str, Dict[str, Any]]:
    """Generate license mapping from EU licenses-skos RDF graph."""
    license_mapping = {}
    try:
        for s in g.subjects(RDF.type, SKOS.Concept):
            spdx_notation = None
            en_label = None
            de_label = None
            exact_matches = []

            for notation in g.objects(s, SKOS.notation):
                spdx_notation = str(notation).lower().strip()

            for label in g.objects(s, SKOS.prefLabel):
                if label.language == "en":
                    en_label = str(label)
                elif label.language == "de":
                    de_label = str(label)

            for match in g.objects(s, SKOS.exactMatch):
                exact_matches.append(str(match))

            if spdx_notation:
                license_mapping[spdx_notation] = {
                    "eu_uri": str(s),
                    "exact_matches": exact_matches,
                    "label_en": en_label,
                    "label_de": de_label
                }
    except Exception as e:
        logger.exception(f"Error during RDF license parsing: {e}")
    
    return license_mapping

def save_results(file_path: Union[Path, str],results: Dict[str, Any]) -> None:
    """Save results to JSON file in specified directory."""
    try:
        output_path = Path(file_path) 
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.exception(f"Failed to save results to {file_path}: {e}")

def create_hf_dcatap_mapping(
    hf_licenses: Set[str], 
    data_dir: Union[Path, str] = Path("./license_code_table"), 
    output_file_path: Union[Path, str] = Path("./license_code_table/hf2dcatap_license_mappings.json")
):
    """
    Create a comprehensive Hugging Face license mapping to DCAT-AP 

    Args:
        hf_licenses: A set of Hugging Face license identifiers
        data_dir: Directory containing the RDF input file and where to save the output JSON.
        output_file_path: Path to the output mapping JSON        
    """  
    try:
        eu_license_mappings = create_eu_license_mapping(load_rdf_graph(data_dir))
    
        # Manual fallback mappings to EU licenses control vocabularies
        manual_eu_mappings = {
            "gpl-3.0": {
                "target": "gpl-3.0-or-later",
                "note_en": "Original Hugging Face license: 'gpl-3.0'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'gpl-3.0'"
            },
            "agpl-3.0": {
                "target": "agpl-3.0-or-later",
                "note_en": "Original Hugging Face license: 'agpl-3.0'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'agpl-3.0'"
            },
            "lgpl-3.0": {
                "target": "lgpl-3.0-or-later",
                "note_en": "Original Hugging Face license: 'lgpl-3.0'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'lgpl-3.0'"
            },     
            # general licenses convert to specific licenses
            "odc-by": {
                "target": "odc-by-1.0",
                "note_en": "Original Hugging Face license: 'odc-by'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'odc-by'"
            },
            "odbl": {
                "target": "odbl-1.0",
                "note_en": "Original Hugging Face license: 'odbl'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'odbl'"
            },  
            "bsd": {
                "target": "bsd-3-clause",
                "note_en": "Original Hugging Face license: 'bsd'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'bsd'"
            }, 
            "cc":  {
                "target": "cc-by-4.0",
                "note_en": "Original Hugging Face license: 'cc'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'cc'"
            }, 
            "gpl": {
                "target": "gpl-3.0-or-later",
                "note_en": "Original Hugging Face license: 'gpl'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'gpl'"
            },
            "gfdl": {
                "target": "gfdl-1.3-or-later",
                "note_en": "Original Hugging Face license: 'gfdl'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'gfdl'"
            },
            "pddl": {
                "target": "pddl-1.0",
                "note_en": "Original Hugging Face license: 'pddl'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'pddl'"
            }
        }

        # Custom license mapping
        manual_custom_licenses = {
            "wtfpl": {
                "uri": "https://spdx.org/licenses/WTFPL.html",
                "exact_matches": [
                    "https://www.wtfpl.net/about/"
                ],
                "label_en": "Do What The F*ck You Want To Public License",
                "label_de": "Mach-was-du-verdammt-nochmal-willst-Lizenz"
            },
            "creativeml-openrail-m": {
                "uri": "https://raw.githubusercontent.com/easydiffusion/easydiffusion/main/CreativeML%20Open%20RAIL-M%20License",
                "label_en": "Creative Machine Learning (CreativeML) Open & Responsible AI License for Machine Learning Models (OpenRAIL‑M)",
                "label_de": "Creative Machine Learning (CreativeML) Offene Verantwortliche KI-Lizenz für ML-Modelle (OpenRAIL‑M)"
            },
            "llama4": {
                "uri": "https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE",
                "label_en": "Meta Llama 4 Community License Agreement",
                "label_de": "Meta Llama 4 Community-Lizenzvereinbarung"
            }, 
            "llama3.3": {
                "uri": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE",
                "label_en": "Meta Llama 3.3 Community License Agreement",
                "label_de": "Meta Llama 3.3 Community-Lizenzvereinbarung"
            }, 
            "llama3.2": {
                "uri": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE",
                "label_en": "Meta Llama 3.2 Community License Agreement",
                "label_de": "Meta Llama 3.2 Community-Lizenzvereinbarung"
            }, 
            "llama3.1": {
                "uri": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE",
                "label_en": "Meta Llama 3.1 Community License Agreement",
                "label_de": "Meta Llama 3.1 Community-Lizenzvereinbarung"
            }, 
            "llama3": {
                "uri": "https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE",
                "label_en": "Meta Llama 3 Community License Agreement",
                "label_de": "Meta Llama 3 Community-Lizenzvereinbarung"
            },
            "llama2": {
                "uri": "https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE",
                "label_en": "Meta Llama 2 Community License Agreement",
                "label_de": "Meta Llama 2 Community-Lizenzvereinbarung"
            }, 
            "c-uda": {
                "uri": "https://spdx.org/licenses/C-UDA-1.0.html",
                "exact_matches": [
                    "https://github.com/microsoft/Computational-Use-of-Data-Agreement/blob/master/C-UDA-1.0.md", 
                    "https://cdla.dev/computational-use-of-data-agreement-v1-0/"
                ],
                "label_en": "Computational Use of Data Agreement v1.0",
                "label_de": "Nutzungsvereinbarung für rechnergestützte Datenverarbeitung Version 1.0", 
                "note_en": "Original Hugging Face license: 'c-uda'",
                "note_de": "Ursprüngliche Hugging Face Lizenz: 'c-uda'"
            },
            "cdla-permissive-1.0": {
                "uri": "https://spdx.org/licenses/CDLA-Permissive-1.0.html",
                "exact_matches": [
                    "https://cdla.io/permissive-1-0"
                ],
                "label_en": "Community Data License Agreement Permissive 1.0",
                "label_de": "Community-Datenlizenzvereinbarung - Permissive 1.0"
            },
            "cdla-permissive-2.0": {
                "uri": "https://spdx.org/licenses/CDLA-Permissive-2.0.html",
                "exact_matches": [
                    "https://cdla.dev/permissive-2-0/"
                ],
                "label_en": "Community Data License Agreement Permissive 2.0",
                "label_de": "Community-Datenlizenzvereinbarung - Permissive 2.0"
            },
            "cdla-sharing-1.0": {
                "uri": "https://spdx.org/licenses/CDLA-Sharing-1.0.html",
                "exact_matches": [
                    "https://cdla.dev/sharing-1-0/"
                ],
                "label_en": "Community Data License Agreement Sharing 1.0",
                "label_de": "Community-Datenlizenzvereinbarung - Sharing 1.0"
            },
            "bigcode-openrail-m": {
                "uri": "https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement",
                "label_en": "BigCode Open & Responsible AI License for Machine Learning Models (OpenRAIL-M) v1 License Agreement",
                "label_de": "BigCode Offene Verantwortliche KI-Lizenz für ML-Modelle (OpenRAIL-M) v1 Lizenzvereinbarung"
            },
            "bigscience-openrail-m": {
                "uri": "https://bigscience.huggingface.co/blog/bigscience-openrail-m",
                "exact_matches": [
                    "https://www.licenses.ai/blog/2022/8/26/bigscience-open-rail-m-license"
                ],
                "label_en": "BigScience Open & Responsible AI License for Machine Learning Models (OpenRAIL-M) License",
                "label_de": "BigScience Offene Verantwortliche KI-Lizenz für ML-Modelle (OpenRAIL-M) Lizenz"
            },
            "bigscience-bloom-rail-1.0": {
                "uri": "https://huggingface.co/spaces/bigscience/license",
                "label_en": "BigScience RAIL License v1.0",
                "label_de": "BigScience RAIL Lizenz v1.0"
            },
            "openrail": {
                "uri": "https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses",
                "label_en": "Open & Responsible AI licenses",
                "label_de": "Offene Verantwortliche KI-Lizenz"
            },
            "openrail++": {
                "uri": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md",
                "label_en": "Open Rail++-M License",
                "label_de": "Offene Verantwortliche KI-Lizenz für Modelle (Open RAIL++-M)"
            },
            # "h-research",  # no info found online regarding this license
            "intel-research": {
                "uri": "https://huggingface.co/OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov/blob/main/LICENSE.md",
                "label_en": "Intel Research Use License Agreement",
                "label_de": "Intel Forschungsnutzungs-Lizenzvereinbarung"
            },
            "apple-ascl": {
                "uri": "https://developer.apple.com/support/downloads/terms/apple-sample-code/Apple-Sample-Code-License.pdf",
                "label_en": "Apple Sample Code license",
                "label_de": "Apple Beispielcode-Lizenz"
            },
            "apple-amlr": {
                "uri": "https://huggingface.co/apple/aimv2-large-patch14-native/blob/main/LICENSE",
                "label_en": "Apple Model License for Research",
                "label_de": "Apple Forschungs-KI-Modell-Lizenzvereinbarung"
            },
            "open-mdw": {
                "uri": "https://openmdw.ai/license/",
                "label_en": "Open Model, Data & Weights (OpenMDW) License Agreement, version 1.0 (OpenMDW-1.0) ",
                "label_de": "Open Model, Data & Weights (OpenMDW) Lizenzvereinbarung, Version 1.0 (OpenMDW‑1.0)"
            },
            "lgpl-lr": {
                "uri": "https://spdx.org/licenses/LGPLLR.html",
                "label_en": "Lesser General Public License For Linguistic Resources",
                "label_de": "Abgeschwächte Allgemeine Öffentliche Lizenz für sprachliche Ressourcen"
            },
            "lgpllr": {
                "uri": "https://spdx.org/licenses/LGPLLR.html",
                "label_en": "Lesser General Public License For Linguistic Resources",
                "label_de": "Abgeschwächte Allgemeine Öffentliche Lizenz für sprachliche Ressourcen"
            },
            "fair-noncommercial-research-license": {
                "uri": "https://huggingface.co/facebook/fair-noncommercial-research-license/blob/main/LICENSE",
                "label_en": "FAIR Noncommercial Research License",
                "label_de": "FAIR-Lizenz für nichtkommerzielle Forschung"
            },
            "gemma": {
                "uri": "https://ai.google.dev/gemma/terms",
                "label_en": "Gemma Terms of Use",
                "label_de": "Nutzungsbedingungen für Gemma"
            },
            "deepfloyd-if-license": {
                "uri": "https://huggingface.co/spaces/DeepFloyd/deepfloyd-if-license",
                "label_en": "DeepFloyd IF Research License Agreement",
                "label_de": "DeepFloyd IF Forschungsnutzungs-Lizenzvereinbarung"
            },
            "unknown": {
                "uri": "http://dcat-ap.de/def/licenses/other-closed", # Fallback to DCAT-AP.de other-closed 
                "label_en": "Unknown license",
                "label_de": "Unbekannte Lizenz"
            },
            "other": {
                "uri": "http://dcat-ap.de/def/licenses/other-closed", # Fallback to DCAT-AP.de other-closed 
                "label_en": "Other license",
                "label_de": "Andere Lizenz"
            }
        }

        final_mappings = {}
        for hf_license in hf_licenses:
            if hf_license in eu_license_mappings:
                data = eu_license_mappings[hf_license]
                final_mappings[hf_license] = {
                    "uri": data["eu_uri"],
                    "exact_matches": data["exact_matches"],
                    "label_en": data["label_en"],
                    "label_de": data["label_de"]
                }
            elif hf_license in manual_eu_mappings:
                target = manual_eu_mappings[hf_license]["target"]
                eu_data = eu_license_mappings.get(target)
                if eu_data:
                    final_mappings[hf_license] = {
                        "uri": eu_data["eu_uri"],
                        "exact_matches": eu_data["exact_matches"],
                        "label_en": eu_data["label_en"],
                        "label_de": eu_data["label_de"],
                        "note_en": manual_eu_mappings[hf_license]["note_en"],
                        "note_de": manual_eu_mappings[hf_license]["note_de"]
                    }
            elif hf_license in manual_custom_licenses:
                final_mappings[hf_license] = manual_custom_licenses[hf_license]

        save_results(output_file_path, final_mappings)
        return final_mappings
    except Exception as e:
        logger.exception(f"Error in creating Hugging Face license to DCAT-AP license mapping: {e}")
        return {}

def create_hf_dcatap_de_mapping(
    hf_licenses: Set[str], 
    hf_dcatap_mappings: Dict[str, Dict[str, Any]], 
    data_dir: Union[Path, str] = Path("./license_code_table"), 
    output_file_path: Union[Path, str] = Path("./license_code_table/hf2dcatap_de_license_mappings.json")
):
    """
    Create a comprehensive Hugging Face license mapping to DCAT-AP.de

    Args:
        hf_licenses: A set of Hugging Face license identifiers
        hf_dcatap_mapping: Dictionary of detailed HF license mapping (includes label_en, note_en, etc.)
        data_dir: Directory (unused for now, reserved for RDF compatibility)
        output_file_path: Path to the output mapping JSON        
    """
    try:
        # Direct mappings: HF license ID → DCAT-AP.de license code
        direct_mappings = {
            "apache-2.0": "apache",
            "bsd": "bsdlicense",
            "bsd-2-clause": "bsdlicense",
            "bsd-3-clause": "bsdlicense",
            "bsd-3-clause-clear": "bsdlicense",
            "cc0-1.0": "cc-zero",
            "cc-by-2.0": "cc-by",
            "cc-by-2.5": "cc-by",
            "cc-by-3.0": "cc-by/3.0",
            "cc-by-4.0": "cc-by/4.0",
            "cc-by-sa-3.0": "cc-by-sa/3.0",
            "cc-by-sa-4.0": "cc-by-sa/4.0",
            "cc-by-nc-2.0": "cc-by-nc",
            "cc-by-nc-3.0": "cc-by-nc/3.0",
            "cc-by-nc-4.0": "cc-by-nc/4.0",
            "cc-by-nd-3.0": "cc-by-nd/3.0",
            "cc-by-nd-4.0": "cc-by-nd/4.0",
            "gfdl": "gfdl",
            "gpl-3.0": "gpl/3.0",
            "mpl-2.0": "mozilla",
            "odbl": "odbl",
            "odc-by": "odby",
            "pddl": "odcpddl"
        }

        # Fallback category mapping
        fallback_classification = {
            "other-open": {
                "cc", "cdla-permissive-1.0", "cdla-permissive-2.0", "etalab-2.0",
                "eupl-1.1", "eupl-1.2", "open-mdw", "gfdl", "odc-by", "bigcode-openrail-m", 
                "bigscience-bloom-rail-1.0", "bigscience-openrail-m",
                "creativeml-openrail-m", "openrail", "openrail++", "open-mdw", 
            },
            "other-opensource": {
                "afl-3.0", "agpl-3.0", "artistic-2.0", "bsl-1.0", "ecl-2.0", "epl-1.0", "epl-2.0",
                "gpl", "gpl-2.0", "lgpl", "lgpl-2.1", "lgpl-3.0", "isc", "lppl-1.3c", "mit",
                "ncsa", "osl-3.0", "postgresql", "ofl-1.1", "unlicense", "wtfpl", "zlib", "lgpl-lr"
            },
            "other-closed": {
                "unknown", "other", "h-research", "gemma", "deepfloyd-if-license", 
                "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3", "llama4"
            },
            "other-commercial": {
                "apple-amlr", "apple-ascl", "intel-research"
            },
            "other-freeware": {
                "c-uda"
            }
        }

        # Generate final mapping: HF ID to DCAT-AP.de URI
        base_uri = "http://dcat-ap.de/def/licenses/"
        final_mappings = {}

        for hf_id in sorted(hf_licenses):
            if hf_id in direct_mappings:
                code = direct_mappings[hf_id]
            else:
                found_category = None
                for fallback_code, ids in fallback_classification.items():
                    if hf_id in ids:
                        found_category = fallback_code
                        break
                code = found_category or "other-closed"
            
            entry = {}
            entry["uri"] = base_uri + code.strip()

            # Mapping enrichment from hf_dcatap_mapping
            if hf_dcatap_mappings and hf_id in hf_dcatap_mappings:
                enrich = hf_dcatap_mappings[hf_id]
                for key in ["exact_matches", "label_en", "label_de", "note_en", "note_de"]:
                    value = enrich.get(key)
                    if value:
                        entry[key] = value

            final_mappings[hf_id] = entry
        
        save_results(output_file_path, final_mappings)
        return final_mappings
    except Exception as e:
        logger.exception(f"Error in create Hugging Face license to DCAT-AP-de license mapping: {e}")
        return {}

if __name__ == "__main__":
    # Fetch Hugging Face licenses from a limit of 5000 datasets and models
    hf_licenses = get_hf_unique_licenses()  
    logger.info(f"Found {len(hf_licenses)} Hugging Face licenses:\n{sorted(hf_licenses)}")
    # Add more Hugging Face licenses (See https://huggingface.co/docs/hub/repositories-licenses)
    other_hf_licenses = {
        "afl-3.0", "agpl-3.0", "cc", "cc-by-nc-nd-3.0", "cc-by-nc-nd-4.0", "cc-by-nc-sa-3.0",
        "cc-by-nc-sa-4.0", "c-uda", "apple-amlr", "artistic-2.0", "bigcode-openrail-m",
        "bigscience-bloom-rail-1.0", "bigscience-openrail-m", "cdla-permissive-1.0",
        "cdla-permissive-2.0", "cdla-sharing-1.0", "creativeml-openrail-m", "deepfloyd-if-license",
        "etalab-2.0", "eupl-1.1", "gemma", "gfdl", "gpl", "gpl-2.0", "lgpl-3.0", "llama2", "llama3",
        "llama3.1", "llama3.2", "llama3.3", "mit", "mpl-2.0", "odc-by", "openrail", "openrail++",
        "other", "pddl", "unknown", "unlicense", "wtfpl", "bsl-1.0", "ecl-2.0", "epl-1.0", "epl-2.0",
        "eupl-1.2", "lgpl", "lgpl-2.1", "isc", "h-research", "lppl-1.3c", "apple-ascl", "open-mdw",
        "osl-3.0", "postgresql", "ofl-1.1", "ncsa", "zlib", "lgpl-lr"
    }
    combined_hf_licenses = hf_licenses.union(other_hf_licenses)
    sorted_hf_licenses = sorted(combined_hf_licenses)
    logger.info(f"Combined {len(sorted_hf_licenses)} Hugging Face licenses:\n{sorted_hf_licenses}")

    # Maps Hugging Face license identifiers to DCAT-AP standard licenses 
    hf2dcatap_license_mappings = create_hf_dcatap_mapping(sorted_hf_licenses)

    # Maps HUgging Face license identifiers to DCAT-AP.de standard licenses 
    create_hf_dcatap_de_mapping(sorted_hf_licenses, hf2dcatap_license_mappings)

