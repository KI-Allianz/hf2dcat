import logging
import json
import re 
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from rdflib import Graph, RDF, SKOS, URIRef, Literal, Namespace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def has_digits(s: str) -> bool:
    return any(c.isdigit() for c in s)

FORBIDDEN_SUBSTRINGS = [
    "pdfa", "pdfx", "xslt","jsonld",
    "rdfxml", "gzip", "bzip", "zipx"
]

def valid_prefix(ext_n: str, label_n: str) -> bool:
    if not label_n.startswith(ext_n):
        return False
    suffix = label_n[len(ext_n):]
    return suffix[:1].isdigit() or suffix.startswith(("-", "_"))

def score_candidate(ext: str, label: str) -> Tuple:
    ext_n = normalize(ext)
    label_n = normalize(label)

    return (
        label_n != ext_n,                           # exact match
        not valid_prefix(ext_n, label_n),           # controlled prefix
        (not has_digits(ext_n) and has_digits(label_n)),
        len(label_n)                                # shorter = more generic
    )

def eliminate_unsafe_candidates(ext: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ext_n = normalize(ext)
    safe: List[Dict[str, Any]] = []

    for c in candidates:
        label_n = normalize(c["file_type_label"])

        if label_n == ext_n:
            safe.append(c)
            continue

        if has_digits(label_n) and not has_digits(ext_n):
            continue

        if any(x in label_n for x in FORBIDDEN_SUBSTRINGS):
            continue

        safe.append(c)

    return safe or candidates   

def select_best_eu_filetype(ext: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    safe_candidates = eliminate_unsafe_candidates(ext, candidates)
    return sorted(
        safe_candidates,
        key=lambda c: score_candidate(ext, c["file_type_label"])
    )[0]

def create_db_mapping(data_dir: Union[Path, str] = Path("./extension_mappings")) -> Dict:
    """Load db.json file and create a mapping from file extension to mime type."""
    # mine-db json file path (db.json downloaded from: https://cdn.jsdelivr.net/gh/jshttp/mime-db@master/db.json)
    db_filepath = Path(data_dir) / "db.json"

    if not db_filepath.exists():
        raise FileNotFoundError(f"db.json file not found at {db_filepath}")

    # Load mine-db JSON file
    with open(db_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter entries where source is 'iana' and there are extensions 
    iana_only_mime_db = {
        mime_type: info
        for mime_type, info in data.items()
        if info.get("source") == "iana" and isinstance(info.get("extensions"), list)
    }

    # Initialize mapping dictionaries
    ext_to_mime = {}
   
    # Build mappings using only IANA-valid entries with extensions
    for mime_type, info in iana_only_mime_db.items():
        extensions = info["extensions"]
        for ext in extensions:
            ext_lower = f".{ext.lower()}"
            if ext_lower not in ext_to_mime:
                ext_to_mime[ext_lower] = mime_type

    sorted_ext_to_mime = dict(sorted(ext_to_mime.items()))
    return sorted_ext_to_mime

def load_rdf_graph(data_dir: Union[Path, str] = Path("./extension_mappings")) -> Graph:
    """Load RDF file from specified directory."""
    g = Graph()
    try:
        rdf_path = Path(data_dir) / "filetypes-skos.rdf"
        if not rdf_path.exists():
            raise FileNotFoundError(f"RDF file not found at {rdf_path}")
        g.parse(rdf_path, format="xml")
    except Exception as e:
        logger.exception(f"Failed to load RDF graph: {e}")
    
    return g

def create_eu_extension_mapping(g: Graph) -> Dict[str, Dict[str, Any]]:
    """Generate extension mapping from file extension to media type and file type based on EU filetypes-skos RDF graph."""

    EUVOC = Namespace("http://publications.europa.eu/ontology/euvoc#")

    extension_candidates: Dict[str, List[Dict[str, Any]]] = {}
    try:       
        for concept in g.subjects(RDF.type, SKOS.Concept):
            file_type_uri = str(concept)
          
            # Get preferred English label
            label = None
            for _, _, o in g.triples((concept, SKOS.prefLabel, None)):
                if isinstance(o, Literal) and o.language == "en":
                    label = str(o)
                    break

            # Gather all extensions and IANA media types for this concept
            extensions: List[str]= []
            media_types: List[str] = []

            for _, _, o in g.triples((concept, SKOS.notation, None)):
                if isinstance(o, Literal):
                    datatype = o.datatype
                    value = str(o).strip()
                    if datatype == EUVOC.FILE_EXT:
                        raw = value.lower()
                        if " " in raw or "," in raw:
                            for ext in re.split(r"[,\s]+", raw):
                                if ext.startswith("."):
                                    extensions.append(ext)
                        else:
                            extensions.append(raw)
                    elif datatype == EUVOC.IANA_MT:
                        mt = value.lower()
                        if " " not in mt and mt not in media_types:
                            media_types.append(mt)

            for ext in extensions:
                extension_candidates.setdefault(ext, []).append({
                    "file_type_uri": file_type_uri,
                    "file_type_label": label,
                    "media_types": media_types,
                })
        
        final_mappings: Dict[str, Dict[str, Any]] = {}
        for ext, candidates in extension_candidates.items():
            final_mappings[ext] = select_best_eu_filetype(ext, candidates)

        return final_mappings
    except Exception as e:
        logger.exception(f"Error during RDF filetype parsing: {e}")
        return {}

def create_extension_mapping(data_dir: Union[Path, str] = Path("./extension_mappings"))-> Dict:
     # Create extension to IANA mime type mapping based on db.json from the specified directory
    db_mapping = create_db_mapping(data_dir)

    # Create extension to file type and media type mapping 
    eu_extension_mapping = create_eu_extension_mapping(load_rdf_graph(data_dir))
    
    if ".tar.gz" in eu_extension_mapping:
        eu_extension_mapping[".tar.gz"] = {
            "file_type": "TAR_GZ", 
            "file_type_uri": "http://publications.europa.eu/resource/authority/file-type/TAR_GZ", 
            "media_types": ["application/gzip"]
        }
    
    if ".yaml" in eu_extension_mapping:
        eu_extension_mapping[".yaml"] = {
            "media_types": ["application/yaml"], 
            "file_type": "YAML",
            "file_type_uri": "http://publications.europa.eu/resource/authority/file-type/YAML",
            "file_type_label": "YAML"
        }
    
    if ".yml" in eu_extension_mapping:
        eu_extension_mapping[".yml"] = {
            "media_types": ["application/yaml"], 
            "file_type": "YAML",
            "file_type_uri": "http://publications.europa.eu/resource/authority/file-type/YAML",
            "file_type_label": "YAML"
        }
        
    # Custom mappings for model weight file extensions
    custom_extension_mapping = {
        ".h5": {
            "media_type": "application/x-hdf",
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/x-hdf",          
            "file_type": "HDF",
            "file_type_uri": "http://publications.europa.eu/resource/authority/file-type/HDF",
        },
        ".bin": {
            "media_type": "application/octet-stream",  
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",          
            "file_type": "BIN",
            "file_type_uri": "http://publications.europa.eu/resource/authority/file-type/BIN",
        },
        ".jsonl": {
            "media_type": "application/x-ndjson",
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/x-ndjson",          
            "file_type": "JSON",
            "file_type_uri": "http://publications.europa.eu/resource/authority/file-type/JSON",
            "file_type_label": "JSON Lines"
        },
        ".jsonl.gz": {
            "media_type": "application/gzip",
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/gzip",          
            "file_type": "JSON",
            "file_type_uri": "http://publications.europa.eu/resource/authority/file-type/JSON",
            "file_type_label": "JSON Lines (compressed)",
        },
        ".msgpack": {
            # Reference: https://github.com/msgpack/msgpack/issues/194#issuecomment-2111445791 & https://www.iana.org/assignments/media-types/application/vnd.msgpack
            "media_type": "application/vnd.msgpack",  
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/vnd.msgpack",          
            "file_type": "MSGPACK",
            "file_type_uri": "https://piveau.io/def/file-type/MSGPACK",  # self defined uri
            "file_type_label": "MessagePack", 
            "see_also": "https://msgpack.org/"
        },
        ".safetensors": {
            "media_type": "application/octet-stream",  
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",          
            "file_type": "SafeTensors",
            "file_type_uri": "https://piveau.io/def/file-type/SAFETENSORS", # self defined uri 
            "file_type_label": "SafeTensors", 
            "see_also": "https://github.com/huggingface/safetensors"
        },
        ".onnx": {
            "media_type": "application/octet-stream",  
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",          
            "file_type": "ONNX",
            "file_type_uri": "https://piveau.io/def/file-type/ONNX", # self defined uri
            "file_type_label": "ONNX", 
            "see_also": "https://onnx.ai/onnx/intro/"
        }, 
        ".pth": {
            "media_type": "application/octet-stream",  
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",          
            "file_type": "PyTorch",
            "file_type_uri": "https://piveau.io/def/file-type/PYTORCH", # self defined uri
            "file_type_label": "PyTorch",
            "see_also": "https://docs.pytorch.org/docs/stable/index.html"  # or "https://docs.pytorch.org/docs/stable/notes/serialization.html"
        },
        ".ot": {
            "media_type": "application/octet-stream",
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",
            "file_type": "TorchScript (Rust)",
            "file_type_uri": "https://piveau.io/def/file-type/TORCHSCRIPT", # self defined uri
            "file_type_label": "TorchScript (Rust)", 
            "see_also": "https://pytorch.org/docs/stable/jit.html"
        },
        ".pt": {
            "media_type": "application/octet-stream",
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",
            "file_type": "TorchScript", 
            "file_type_uri": "https://piveau.io/def/file-type/TORCHSCRIPT", # self defined uri
            "file_type_label": "TorchScript Model",
            "see_also": "https://pytorch.org/docs/stable/jit.html"
        },
        ".mlmodel": {
            "media_type": "application/x-coreml-model",
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",
            "file_type": "COREML",
            "file_type_uri":  "https://piveau.io/def/file-type/COREML", # self defined uri
            "file_type_label": "Core ML Model", 
            "see_also": "https://apple.github.io/coremltools/docs-guides/"
        }, 
        ".gguf": {
            "media_type": "application/octet-stream",
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",
            "file_type": "GGUF",
            "file_type_uri": "https://piveau.io/def/file-type/GGUF", # self defined uri
            "file_type_label": "GGUF",
            "see_also": "https://github.com/ggml-org/ggml/blob/master/docs/gguf.md"
        },
        ".ggml": {
            "media_type": "application/octet-stream",
            "media_type_uri": "http://www.iana.org/assignments/media-types/application/octet-stream",
            "file_type": "GGML",
            "file_type_uri": "https://piveau.io/def/file-type/GGML", # self defined uri
            "file_type_label": "GGML", 
            "see_also": "https://huggingface.co/blog/introduction-to-ggml"
        }
    }

    final_mapping = {}
    unmapped_extensions = []

    all_extensions = set().union(
        eu_extension_mapping.keys(),
        db_mapping.keys(),
        custom_extension_mapping.keys()
    )

    for ext in sorted(all_extensions):
        entry = {}

        # Step 1: Try EU mappings
        eu_info = eu_extension_mapping.get(ext)
        if eu_info:
            entry["file_type_uri"] = eu_info.get("file_type_uri")
            entry["file_type_label"] = eu_info.get("file_type_label")

            media_types = eu_info.get("media_types")
            if media_types:
                entry["media_type"] = media_types[0]
                entry["media_type_uri"] = f"http://www.iana.org/assignments/media-types/{media_types[0]}"

        # Step 2: If media_type missing, try MIME-db
        if "media_type" not in entry:
            mime = db_mapping.get(ext)
            if mime:
                entry["media_type"] = mime
                entry["media_type_uri"] = f"http://www.iana.org/assignments/media-types/{mime}"

        # Step 3: If file_type still missing, try custom mapping
        custom_info = custom_extension_mapping.get(ext)
        if custom_info:
            if "file_type_uri" not in entry and "file_type_uri" in custom_info:
                entry["file_type_uri"] = custom_info["file_type_uri"]
            if "file_type_label" not in entry and "file_type_label" in custom_info:
                entry["file_type_label"] = custom_info["file_type_label"]
            if "file_type" not in entry and "file_type" in custom_info:
                entry["file_type"] = custom_info["file_type"]  
            if "media_type" not in entry and "media_type" in custom_info:
                entry["media_type"] = custom_info["media_type"]
                entry["media_type_uri"] = custom_info["media_type_uri"]
            if "see_also" not in entry and "see_also" in custom_info:
                entry["see_also"] = custom_info["see_also"]


        # Step 4: Fallback — log if still missing both media_type and file_type
        if "media_type_uri" not in entry and "file_type_uri" not in entry:
            logger.warning(f"No media_type or file_type found for extension: {ext}")
            unmapped_extensions.append(ext)
            continue  

        final_mapping[ext] = entry

    return (final_mapping, unmapped_extensions)

def save_results(file_path: Union[Path, str],results: Dict[str, Any]) -> None:
    """Save results to JSON file in specified directory."""
    try:
        output_path = Path(file_path) 
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.exception(f"Failed to save results to {file_path}: {e}")


if __name__ == "__main__":
    data_dir = Path("./extension_mappings")
    mapping_output_filepath = data_dir / "extension2_mediatype_filetype_mappings.json"

    final_mapping, unmapped_extensions = create_extension_mapping(data_dir)

    save_results(mapping_output_filepath, final_mapping)

    logger.info(f"unmapped extensions: {unmapped_extensions}")

 

