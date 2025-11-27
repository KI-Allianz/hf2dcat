import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from rdflib import Graph, RDF, SKOS, URIRef, Literal, Namespace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    extension_mappings = {}
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
            extensions = []
            media_types = []

            for _, _, o in g.triples((concept, SKOS.notation, None)):
                if isinstance(o, Literal):
                    datatype = o.datatype
                    value = str(o).strip()
                    if datatype == EUVOC.FILE_EXT:
                        extensions.append(value.lower())
                    elif datatype == EUVOC.IANA_MT:
                        media_types.append(value.lower())

            for ext in extensions:
                if ext not in extension_mappings:
                    extension_mappings[ext] = {
                        "file_type_uri": file_type_uri,
                        "file_type_label": label,
                        "media_types": media_types
                    }
                else:
                    # Merge media types if extension is shared
                    extension_mappings[ext]["media_types"].extend(
                        [mt for mt in media_types if mt not in extension_mappings[ext]["media_types"]]
                    )
        
    except Exception as e:
        logger.exception(f"Error during RDF filetype parsing: {e}")
    
    return extension_mappings

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
        },
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

