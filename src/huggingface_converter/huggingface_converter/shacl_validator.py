import logging
import json
import hashlib
import requests
import tempfile
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Iterable, NamedTuple
from enum import Enum, auto
from rdflib import Graph, URIRef
from rdflib.namespace import RDFS
from pyshacl import validate
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing_extensions import TypedDict
from dataclasses import dataclass
from html import escape

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent

class ReportFormat(Enum):
    """Supported report output formats"""
    JSON = "json"
    TXT = "txt"
    HTML = "html"

class ValidationDetail(TypedDict):
    """Detailed description of a single SHACL validation result item."""
    focusNode: str
    resultPath: str
    message: str
    sourceShape: str
    valueNode: str

class ValidationDetails(TypedDict):
    """Grouped SHACL validation result details, separated by severity."""
    conforms: bool
    violations: List[ValidationDetail]
    warnings: List[ValidationDetail]
    infos: List[ValidationDetail]
 
@dataclass
class ValidationResult:
    """Structured representation of SHACL validation output."""
    conforms: bool
    report: str
    details: ValidationDetails

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "conforms": self.conforms,
            "report": self.report,
            "details": self.details
        }
    
    def __str__(self) -> str:
        """Return a string summary of the validation result."""
        return (f"ValidationResult(conforms={self.conforms}, "
                f"violations={len(self.details['violations'])}, "
                f"warnings={len(self.details['warnings'])})")

class SHACLProfile(Enum):
    """Supported SHACL validation profiles"""
    DCAT_AP = auto()
    DCAT_AP_DE = auto()
    CUSTOM = auto()

class ShapeCacheKey(NamedTuple):
    """Cache key for SHACL shape graphs"""
    profile: SHACLProfile
    custom_shapes_hash: Optional[str] = None

class SHACLValidator:
    """
    A SHACL (Shapes Constraint Language) validator for RDF graphs.

    This class validates RDF data against SHACL shapes, supporting configurable
    validation options, batch processing, report generation, and reusable validation
    workflows. 

    Key Features:
    -------------
    - Supports SHACL profiles: DCAT-AP, DCAT-AP.de, and custom shapes
    - Configurable SHACL validation settings: inference, meta-SHACL, abort-on-first, warnings
    - Detailed structured results with `ValidationResult` and violation summaries
    - Output reports in JSON, TXT, or HTML formats
    - Batch validation with optional parallel processing
    - Caching for SHACL shape graphs and file hashes to avoid redundant processing
    - Progress hooks for integration into GUI, CLI, or monitoring systems
    - Cleaned up temporary files automatically via context management

    Usage Examples:
    ---------
    Basic usage:

        >>> validator = SHACLValidator()
        >>> result = validator.validate_file("data/catalog.ttl")
        >>> print(result.conforms)
        >>> print(result.to_dict())

    Using custom shapes and HTML report output:

        >>> result = validator.validate_file(
        ...     rdf_file="dataset.ttl",
        ...     custom_shapes=["custom_shapes.ttl"],
        ...     report_formats=[ReportFormat.HTML, ReportFormat.JSON]
        ... )

    Batch validation with parallel workers and inference disabled:

        >>> results = validator.validate_batch(
        ...     sources=["datasets/"],
        ...     profile=SHACLProfile.DCAT_AP,
        ...     inference="none",
        ...     parallel=True
        ... )
        >>> for r in results:
        ...     print(r["source"], "✓" if r["conforms"] else "✗")
    """

    _PROFILE_CONFIG = {
        SHACLProfile.DCAT_AP: {
            "shape_dir": BASE_DIR.parent / "shapes" / "DCAT_AP_SHACL_SHAPES",
            "repo_url": "https://raw.githubusercontent.com/SEMICeu/DCAT-AP/refs/heads/master/releases/3.0.0/shacl",
            "required_shapes": ["dcat-ap-SHACL.ttl"]
        },
        SHACLProfile.DCAT_AP_DE: {
            "shape_dir": BASE_DIR.parent / "shapes" / "DCAT_AP_DE_SHACL_SHAPES",
            "repo_url": "https://raw.githubusercontent.com/GovDataOfficial/DCAT-AP.de-SHACL-Validation/refs/heads/master/validator/resources/v3.0/shapes",
            "required_shapes": [
                "dcat-ap-SHACL-DE.ttl", "dcat-ap-de-imports.ttl",
                "dcat-ap-de-controlledvocabularies.ttl", "dcat-ap-de-deprecated.ttl",
                "dcat-ap-spec-german-additions.ttl"
            ],
            "base_profile": SHACLProfile.DCAT_AP
        }
    }

    _FORMAT_MAPPING = {
        ".ttl": "turtle",
        ".rdf": "xml",
        ".xml": "xml",
        ".nt": "nt",
        ".jsonld": "json-ld",
        ".nq": "nquads",
        ".trig": "trig"
    }

    def __init__(
        self,
        max_workers: int = 4,
        validation_start_hook: Optional[callable] = None,
        validation_complete_hook: Optional[callable] = None
    ):
        """
        Initialize the SHACLValidator.

        Args:
            max_workers: Maximum number of threads to use in parallel batch validation.
            validation_start_hook: Optional callback invoked with the input being validated.
            validation_complete_hook: Optional callback invoked after validation completes.

        Examples:
            >>> validator = SHACLValidator(max_workers=2)
        """
        self.max_workers = max_workers
        self.validation_start_hook = validation_start_hook
        self.validation_complete_hook = validation_complete_hook
        self._shape_cache: Dict[ShapeCacheKey, Graph] = {}
        self._processed_hashes: Set[str] = set()
        self._temp_dirs: List[Path] = []
        self._ensure_shape_dirs_exist()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up temporary directories"""
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
        self._temp_dirs.clear()

    def _ensure_shape_dirs_exist(self) -> None:
        """Create shape directories if they don't exist"""
        for config in self._PROFILE_CONFIG.values():
            try:
                config["shape_dir"].mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured shape directory exists: {config['shape_dir']}")
            except OSError as e:
                logger.exception(f"Failed to create shape directory {config['shape_dir']}: {e}")

    def _get_format(self, path: Path) -> str:
        """Determine RDF format from file extension"""
        ext = path.suffix.lower()
        if ext not in self._FORMAT_MAPPING:
            logger.warning(f"Unknown file extension {ext}, defaulting to turtle")
        return self._FORMAT_MAPPING.get(ext, "turtle")

    def _load_shapes_from_dir(self, shape_dir: Path, shapes: List[str], graph: Graph) -> bool:
        """
        Attempt to load SHACL shape files from a directory

        Args:
            shape_dir: Directory path.
            shapes: List of expected SHACL shape files
            graph: Graph to populate with parsed shapes

        Returns:
            Boolean indicating whether all files were successfully loaded
        """
        if not shape_dir.exists():
            return False
            
        success = True
        for shape in shapes:
            path = shape_dir / shape
            try:
                if path.exists():
                    graph.parse(path, format="turtle")
                    logger.info(f"Loaded shape from cache: {path}")
                else:
                    success = False
            except Exception as e:
                logger.warning(f"Failed to parse shape file {path}: {e}")
                success = False
        return success

    def _download_shapes(self, repo_url: str, shapes: List[str], shape_dir: Path, target_graph: Graph) -> None:
        """Download shapes in parallel and add them to the target graph"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._download_shape,
                    urljoin(repo_url.rstrip('/') + "/", shape),
                    shape_dir / shape
                ): shape for shape in shapes
            }
            
            for future in as_completed(futures):
                shape = futures[future]
                try:
                    success, downloaded_graph = future.result()
                    if success and downloaded_graph:
                        target_graph += downloaded_graph
                except Exception as e:
                    logger.exception(f"Failed to download shape {shape}: {e}")

    def _download_shape(self, url: str, local_path: Optional[Path] = None) -> Tuple[bool, Optional[Graph]]:
        """
        Download and optionally cache a SHACL shape file from a URL

        Args:
            url: URL of the shape file to download
            local_path: Local destination to cache the downloaded shape file
 
        Returns:
            Tuple of (success, graph) where success is True if parsed
        """
        graph = Graph()
 
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse into graph first to validate it's valid RDF
            graph.parse(data=response.text, format="turtle")
 
            if local_path:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(response.text, encoding="utf-8")
                logger.info(f"Downloaded and cached shape: {url} -> {local_path}")
       
            return True, graph
        except requests.RequestException as e:
            logger.warning(f"Failed to download shape from {url}: {e}")
            return False, None
        except Exception as e:
            logger.warning(f"Invalid RDF in shape file {url}: {e}")
            return False, None

    def _get_shacl_graph(
        self,
        profile: SHACLProfile,
        custom_shapes: Optional[List[str]] = None
    ) -> Graph:
        """Get SHACL graph for profile, loading if not cached"""
        custom_shapes_hash = None
        if custom_shapes:
            hasher = hashlib.md5()
            for shape in sorted(custom_shapes):
                hasher.update(Path(shape).read_bytes())
            custom_shapes_hash = hasher.hexdigest()
            
        cache_key = ShapeCacheKey(profile, custom_shapes_hash)
        
        if cache_key not in self._shape_cache:
            if profile == SHACLProfile.CUSTOM and custom_shapes:
                self._shape_cache[cache_key] = self._load_custom_shapes(custom_shapes)
            else:
                self._shape_cache[cache_key] = self._load_shapes_for_profile(profile)
                
        return self._shape_cache[cache_key]

    def _load_custom_shapes(self, shape_files: List[str]) -> Graph:
        """Load custom SHACL shapes from files"""
        shacl_graph = Graph()
        
        for shape_file in shape_files:
            path = Path(shape_file)
            if not path.exists():
                raise FileNotFoundError(f"Shape file not found: {shape_file}")
                
            try:
                shacl_graph.parse(path, format=self._get_format(path))
                logger.info(f"Loaded custom shape: {shape_file}")
            except Exception as e:
                logger.exception(f"Failed to parse shape file {shape_file}: {e}")
                raise
                    
        if len(shacl_graph) == 0:
            raise ValueError("No valid SHACL shapes were loaded")
            
        return shacl_graph

    def _load_shapes_for_profile(self, profile: SHACLProfile) -> Graph:
        """Load all required shapes for a profile with parallel downloading"""
        config = self._PROFILE_CONFIG[profile]
        shacl_graph = Graph()

        # For DCAT-AP.de, first load base DCAT-AP shapes
        if profile == SHACLProfile.DCAT_AP_DE:
            base_config = self._PROFILE_CONFIG[config["base_profile"]]
            if not self._load_shapes_from_dir(base_config["shape_dir"], 
                                           base_config["required_shapes"], 
                                           shacl_graph):
                self._download_shapes(base_config["repo_url"],
                                   base_config["required_shapes"],
                                   base_config["shape_dir"],
                                   shacl_graph)
        
        # Load DCAT-AP shapes 
        if not self._load_shapes_from_dir(config["shape_dir"], 
                                       config["required_shapes"], 
                                       shacl_graph):
            self._download_shapes(config["repo_url"],
                               config["required_shapes"],
                               config["shape_dir"],
                               shacl_graph)
        
        if len(shacl_graph) == 0:
            raise ValueError(f"Could not load any SHACL shapes for profile {profile.name}")
        
        return shacl_graph

    def _validate_graph(
        self,
        data_graph: Graph,
        shacl_graph: Graph,
        inference: str = 'rdfs',
        meta_shacl: bool = False,
        abort_on_first: bool = False,
        allow_warnings: bool = True
    ) -> ValidationResult:
        """
        Core validation logic
        
        Args:
            data_graph: Graph to be validated
            shacl_graph: SHACL shapes graph
            inference: Type of inference ('none', 'rdfs', 'owlrl', 'both')
            meta_shacl: Validate SHACL shapes against SHACL-SHACL
            abort_on_first: Stop on first validation error
            allow_warnings: Treat warnings as non-conforming
            
        Returns:
            ValidationResult object
        """
        try:
            conforms, report_graph, report_text = validate(
                data_graph,
                shacl_graph=shacl_graph,
                ont_graph=None,
                inference=inference,
                abort_on_first=abort_on_first,
                allow_warnings=allow_warnings,
                meta_shacl=meta_shacl,
                advanced=True,
                js=False
            )
            
            details = self._extract_validation_details(report_graph)
            return ValidationResult(conforms, report_text, details)
            
        except Exception as e:
            logger.exception(f"Validation failed: {str(e)}")
            return ValidationResult(
                False,
                str(e),
                {"error": str(e)}
            )

    def validate_file(
        self,
        rdf_file: Union[str, Path],
        profile: SHACLProfile = SHACLProfile.DCAT_AP_DE,
        custom_shapes: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        report_formats: Iterable[ReportFormat] = (ReportFormat.JSON, ReportFormat.TXT),
        inference: str = 'rdfs',
        meta_shacl: bool = True,
        abort_on_first: bool = False,
        allow_warnings: bool = True
    ) -> ValidationResult:
        """
        Validate an RDF file against SHACL shapes
        
        Args:
            rdf_file: Path to the RDF file to be validated
            profile: SHACL profile (DCAT_AP, DCAT_AP_DE, CUSTOM)
            custom_shapes: Optional list of paths to custom SHACL shape files
            output_dir: Optional directory to save reports
            report_formats: Formats for report output (JSON, TXT, HTML).
            inference: Type of inference ('none', 'rdfs', 'owlrl', 'both')
            meta_shacl: Validate SHACL shapes against SHACL-SHACL
            abort_on_first: Stop on first validation error
            allow_warnings: Treat warnings as non-conforming
            
        Returns:
            ValidationResult with conformance info, report text, and structured issue details.

        Usage Example:
            >>> validator = SHACLValidator()
            >>> result = validator.validate_file("example.ttl")
            >>> print(result.conforms)
            >>> print(result.details)
        """
        rdf_file = Path(rdf_file)
        if not rdf_file.exists():
            raise FileNotFoundError(f"RDF file not found: {rdf_file}")
            
        if self.validation_start_hook:
            self.validation_start_hook(rdf_file)
            
        graph = Graph()
        graph.parse(rdf_file, format=self._get_format(rdf_file))
        
        shacl_graph = self._get_shacl_graph(profile, custom_shapes)
        result = self._validate_graph(
            graph,
            shacl_graph,
            inference,
            meta_shacl,
            abort_on_first,
            allow_warnings
        )
        
        if output_dir:
            self._save_reports(
                output_dir,
                rdf_file.stem,
                result,
                file_source=rdf_file,
                formats=report_formats
            )
            
        if self.validation_complete_hook:
            self.validation_complete_hook(rdf_file, result)
            
        return result

    def validate_graph(
        self,
        graph: Graph,
        profile: SHACLProfile = SHACLProfile.DCAT_AP_DE,
        custom_shapes: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        identifier: str = "graph",
        report_formats: Iterable[ReportFormat] = (ReportFormat.JSON, ReportFormat.TXT),
        inference: str = 'rdfs',
        meta_shacl: bool = True,
        abort_on_first: bool = False,
        allow_warnings: bool = True
    ) -> ValidationResult:
        """
        Validate an RDFlib Graph object
        
        Args:
            graph: RDFlib Graph to be validated
            profile: SHACL profile (DCAT_AP, DCAT_AP_DE, CUSTOM)
            custom_shapes: Optional list of paths to custom SHACL shape files
            output_dir: Optional directory to save reports
            identifier: Identifier for report files
            report_formats: Output formats (JSON, TXT, HTML)
            inference: Type of inference ('none', 'rdfs', 'owlrl', 'both')
            meta_shacl: Validate SHACL shapes against SHACL-SHACL
            abort_on_first: Stop on first validation error
            allow_warnings: Treat warnings as non-conforming
            
        Returns:
            ValidationResult with conformance info, report text, and structured issue details.

        Usage Example:
            >>> g = Graph()
            >>> g.parse("dataset.ttl", format="turtle")
            >>> validator = SHACLValidator()
            >>> result = validator.validate_graph(g)
        """
        if self.validation_start_hook:
            self.validation_start_hook(identifier)
            
        shacl_graph = self._get_shacl_graph(profile, custom_shapes)
        result = self._validate_graph(
            graph,
            shacl_graph,
            inference,
            meta_shacl,
            abort_on_first,
            allow_warnings
        )
        
        if output_dir:
            self._save_reports(
                output_dir,
                identifier,
                result,
                formats=report_formats
            )
            
        if self.validation_complete_hook:
            self.validation_complete_hook(identifier, result)
            
        return result

    def validate_string(
        self,
        rdf_content: str,
        format: str = "turtle",
        profile: SHACLProfile = SHACLProfile.DCAT_AP_DE,
        custom_shapes: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        report_formats: Iterable[ReportFormat] = (ReportFormat.JSON, ReportFormat.TXT),
        inference: str = 'rdfs',
        meta_shacl: bool = True,
        abort_on_first: bool = False,
        allow_warnings: bool = True
    ) -> ValidationResult:
        '''
        Validate RDF content provided as a string
        
        Args:
            rdf_content: RDF content as string
            format: RDF format (e.g., 'turtle', 'json-ld')
            profile: SHACL profile (DCAT_AP, DCAT_AP_DE, CUSTOM)
            custom_shapes: Optional list of paths to custom SHACL shape files
            output_dir: Optinal directory to save reports
            report_formats: Output formats (JSON, TXT, HTML)
            inference: Type of inference ('none', 'rdfs', 'owlrl', 'both')
            meta_shacl: Validate SHACL shapes against SHACL-SHACL
            abort_on_first: Stop on first validation error
            allow_warnings: Treat warnings as non-conforming
            
        Returns:
            ValidationResult with conformance info, report text, and structured issue details.

        Usage Example:
            >>> data = """@prefix dcat: <http://www.w3.org/ns/dcat#> . ..."""
            >>> validator = SHACLValidator()
            >>> result = validator.validate_string(data)
        '''

        if self.validation_start_hook:
            self.validation_start_hook("string_content")
            
        graph = Graph()
        graph.parse(data=rdf_content, format=format)
        
        shacl_graph = self._get_shacl_graph(profile, custom_shapes)
        result = self._validate_graph(
            graph,
            shacl_graph,
            inference,
            meta_shacl,
            abort_on_first,
            allow_warnings
        )
        
        if output_dir:
            self._save_reports(
                output_dir,
                "string_content",
                result,
                formats=report_formats
            )
            
        if self.validation_complete_hook:
            self.validation_complete_hook("string_content", result)
            
        return result

    def validate_batch(
        self,
        sources: List[Union[str, Path]],
        profile: SHACLProfile = SHACLProfile.DCAT_AP_DE,
        custom_shapes: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        report_formats: Iterable[ReportFormat] = (ReportFormat.JSON, ReportFormat.TXT),
        inference: str = 'rdfs',
        meta_shacl: bool = True,
        abort_on_first: bool = False,
        allow_warnings: bool = True,
        parallel: bool = True
    ) -> List[Dict]:
        """
        Validate a list of RDF files or directories in batch mode
        
        Args:
            sources: List of files/directories to validate
            profile: SHACL profile (DCAT_AP, DCAT_AP_DE, CUSTOM)
            custom_shapes: Optional list of paths to custom SHACL shape files
            output_dir: Optional directory to save reports
            report_formats: Report formats (JSON, TXT, HTML)
            inference: Type of inference ('none', 'rdfs', 'owlrl', 'both')
            meta_shacl: Validate SHACL shapes against SHACL-SHACL
            abort_on_first: Stop on first validation error
            allow_warnings: Treat warnings as non-conforming
            parallel: Whether to use parallel processing
            
        Returns:
            List of validation results with file info and conformance results.

        Usage Example:
            >>> validator = SHACLValidator()
            >>> results = validator.validate_batch(["dir/", "file.ttl"])
            >>> for r in results:
            ...     print(r["source"], r["conforms"])
        """
        files_to_validate = self._collect_rdf_files(sources)
        if not files_to_validate:
            logger.warning("No valid RDF files found in provided sources")
            return []
            
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = output_dir / "validation_summary.json"
            self._processed_hashes.update(self._load_hashes_from_summary(summary_path))
        else:
            summary_path = None
            
        results = []
        
        if parallel:
            results = self._validate_batch_parallel(
                files_to_validate,
                profile,
                custom_shapes,
                output_dir,
                report_formats,
                inference,
                meta_shacl,
                abort_on_first,
                allow_warnings
            )
        else:
            results = self._validate_batch_sequential(
                files_to_validate,
                profile,
                custom_shapes,
                output_dir,
                report_formats,
                inference,
                meta_shacl,
                abort_on_first,
                allow_warnings
            )
        
        if summary_path:
            self._update_summary_file(summary_path, results)
            
        return results

    def _validate_batch_parallel(
        self,
        files: List[Path],
        profile: SHACLProfile,
        custom_shapes: Optional[List[str]],
        output_dir: Optional[Path],
        report_formats: Iterable[ReportFormat],
        inference: str,
        meta_shacl: bool,
        abort_on_first: bool,
        allow_warnings: bool
    ) -> List[Dict]:
        """
        Perform parallel validation of multiple RDF files using threads

        Args:
            files: List of RDF file paths.
            profile: SHACL profile (DCAT_AP, DCAT_AP_DE, CUSTOM)
            custom_shapes: Optional list of paths to custom SHACL shape files
            output_dir: Optional directory to store report files
            report_formats: Report formats (JSON, TXT, HTML)
            inference: Type of inference ('none', 'rdfs', 'owlrl', 'both')
            meta_shacl: Validate SHACL shapes against SHACL-SHACL
            abort_on_first: Stop on first validation error
            allow_warnings: Treat warnings as non-conforming

        Returns:
            A list of result dictionaries from each file validated
        """
        results = []
        temp_dir = Path(tempfile.mkdtemp(prefix="shacl_val_"))
        self._temp_dirs.append(temp_dir)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_file,
                    file_path,
                    profile,
                    custom_shapes,
                    output_dir,
                    report_formats,
                    inference,
                    meta_shacl,
                    abort_on_first,
                    allow_warnings,
                    temp_dir / f"{file_path.stem}_report.json"
                ): file_path for file_path in files
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.exception(f"Failed to validate {file_path}: {e}")
                    results.append({
                        "source": str(file_path),
                        "error": str(e),
                        "conforms": False
                    })
                    
        return results

    def _validate_batch_sequential(
        self,
        files: List[Path],
        profile: SHACLProfile,
        custom_shapes: Optional[List[str]],
        output_dir: Optional[Path],
        report_formats: Iterable[ReportFormat],
        inference: str,
        meta_shacl: bool,
        abort_on_first: bool,
        allow_warnings: bool
    ) -> List[Dict]:
        """
        Perform sequential validation of RDF files

        Args:
            files: RDF file paths to validate
            profile: SHACL profile (DCAT_AP, DCAT_AP_DE, CUSTOM)
            custom_shapes: Optional list of paths to custom SHACL shape files
            output_dir: Optional directory to store report files
            report_formats: Report formats (JSON, TXT, HTML)
            inference: Type of inference ('none', 'rdfs', 'owlrl', 'both')
            meta_shacl: Validate SHACL shapes against SHACL-SHACL
            abort_on_first: Stop on first validation error
            allow_warnings: Treat warnings as non-conforming

        Returns:
            List of dictionaries containing validation outcomes
        """
        results = []
        
        for file_path in files:
            try:
                result = self._process_single_file(
                    file_path,
                    profile,
                    custom_shapes,
                    output_dir,
                    report_formats,
                    inference,
                    meta_shacl,
                    abort_on_first,
                    allow_warnings
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.exception(f"Failed to validate {file_path}: {e}")
                results.append({
                    "source": str(file_path),
                    "error": str(e),
                    "conforms": False
                })
                
        return results

    def _process_single_file(
        self,
        file_path: Path,
        profile: SHACLProfile,
        custom_shapes: Optional[List[str]],
        output_dir: Optional[Path],
        report_formats: Iterable[ReportFormat],
        inference: str,
        meta_shacl: bool,
        abort_on_first: bool,
        allow_warnings: bool,
        temp_report: Optional[Path] = None
    ) -> Optional[Dict]:
        """
        Run validation on a single RDF file and generate result metadata

        Args:
            file_path: File path to RDF data to validate
            profile: SHACL profile (DCAT_AP, DCAT_AP_DE, CUSTOM)
            custom_shapes: Optional list of paths to custom SHACL shape files
            output_dir: Optional directory to store report files
            report_formats: Report formats (JSON, TXT, HTML)
            inference: Type of inference ('none', 'rdfs', 'owlrl', 'both')
            meta_shacl: Validate SHACL shapes against SHACL-SHACL
            abort_on_first: Stop on first validation error
            allow_warnings: Treat warnings as non-conforming
            temp_report: Optional path to write temp report JSON

        Returns:
            A dictionary with validation result and metadata, or None if skipped
        """
        file_hash = self._calculate_file_hash(file_path)
        
        if file_hash in self._processed_hashes:
            logger.debug(f"Skipping duplicate file: {file_path}")
            return None
            
        result = self.validate_file(
            file_path,
            profile,
            custom_shapes,
            output_dir,
            report_formats,
            inference,
            meta_shacl,
            abort_on_first,
            allow_warnings
        )
        report_entry = self._generate_report_entry(file_path, result, file_hash)
        self._processed_hashes.add(file_hash)
        
        if temp_report:
            temp_report.write_text(json.dumps(report_entry, indent=2), encoding="utf-8")
            
        if output_dir:
            self._save_reports(
                output_dir,
                file_path.stem,
                result,
                file_source=file_path,
                formats=report_formats
            )
            
        return report_entry

    def _collect_rdf_files(self, sources: List[Union[str, Path]]) -> List[Path]:
        """Collect all RDF files from paths (including directories)"""
        files = []
        for source in sources:
            path = Path(source)
            if path.is_file() and path.suffix.lower() in self._FORMAT_MAPPING:
                files.append(path)
            elif path.is_dir():
                files.extend(
                    f for f in path.rglob("*.*") 
                    if f.suffix.lower() in self._FORMAT_MAPPING and not f.name.startswith(".")
                )
        return files

    def _extract_validation_details(self, report_graph: Graph) -> ValidationDetails:
        """Extract structured details from SHACL validation report graph, including conformance."""
        details: ValidationDetails = {
            "conforms": True,  
            "violations": [],
            "warnings": [],
            "infos": []
        }

        # SHACL predicates
        SH_CONFORMS = URIRef("http://www.w3.org/ns/shacl#conforms")
        SH_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        SH_RESULT = URIRef("http://www.w3.org/ns/shacl#ValidationResult")
        SH_SEVERITY = URIRef("http://www.w3.org/ns/shacl#resultSeverity")
        SH_VIOLATION = URIRef("http://www.w3.org/ns/shacl#Violation")
        SH_WARNING = URIRef("http://www.w3.org/ns/shacl#Warning")
        SH_INFO = URIRef("http://www.w3.org/ns/shacl#Info")
        SH_FOCUS_NODE = URIRef("http://www.w3.org/ns/shacl#focusNode")
        SH_RESULT_PATH = URIRef("http://www.w3.org/ns/shacl#resultPath")
        SH_MESSAGE = URIRef("http://www.w3.org/ns/shacl#resultMessage")
        SH_SOURCE_SHAPE = URIRef("http://www.w3.org/ns/shacl#sourceShape")
        SH_VALUE = URIRef("http://www.w3.org/ns/shacl#value")

        # Check global conformance flag
        for conforms in report_graph.objects(None, SH_CONFORMS):
            if isinstance(conforms, (bool, int)):
                details["conforms"] = bool(conforms)

        # Extract validation results
        for result in report_graph.subjects(SH_TYPE, SH_RESULT):
            severity = report_graph.value(result, SH_SEVERITY)

            source_shape_node = report_graph.value(result, SH_SOURCE_SHAPE)
            # Try to resolve a label if the node is a blank node
            label = report_graph.value(source_shape_node, RDFS.label)
            source_shape_str = str(label if label else source_shape_node)

            entry: ValidationDetail = {
                "focusNode": str(report_graph.value(result, SH_FOCUS_NODE) or ""),
                "resultPath": str(report_graph.value(result, SH_RESULT_PATH) or ""),
                "message": str(report_graph.value(result, SH_MESSAGE) or ""),
                "sourceShape": source_shape_str, 
                "valueNode": str(report_graph.value(result, SH_VALUE) or "")
            }

            if severity == SH_VIOLATION:
                details["violations"].append(entry)
            elif severity == SH_WARNING:
                details["warnings"].append(entry)
            elif severity == SH_INFO:
                details["infos"].append(entry)

        # Flag unexplained non-conformance
        if not details["conforms"] and not any(details[k] for k in ["violations", "warnings", "infos"]):
            details["warnings"].append({
                "focusNode": "",
                "resultPath": "",
                "message": "File marked as non-conforming but no validation results found. Possible parsing error.",
                "sourceShape": "",
                "valueNode": ""
            })

        return details

    def group_validation_results(
        self,
        details: ValidationDetails,
        *,
        include_warnings: bool = True,
        include_infos: bool = False,
        max_focus_nodes: int = 5,
        severity_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Group validation results by message and path
        
        Args:
            details: ValidationDetails dictionary containing violations, warnings, infos
            include_warnings: Whether to include warning messages in grouping
            include_infos: Whether to include info messages in grouping
            max_focus_nodes: Maximum number of focus nodes to include per group
            severity_filter: Optional list of severities to include ('violations', 'warnings', 'infos')
        
        Returns:
            List of grouped results with:
            - message: The validation message
            - path: The result path
            - severity: The severity level
            - focusNodes: List of focus nodes (limited by max_focus_nodes)
            - count: Total number of occurrences
            - exampleNode: First focus node as an example
        
        Usage Example:
            >>> validator = SHACLValidator()
            >>> validation_result = validator.validate_graph(g)
            >>> grouped = validator.group_validation_results(validation_result.details)
            >>> for group in grouped:
            ...     print(f"{group['severity'].upper()}: {group['message']}")
            ...     print(f"Path: {group['path']} (Occurrences: {group['count']})")
            ...     print(f"Example node: {group['exampleNode']}")
        """
        if severity_filter is None:
            severity_filter = ['violations']
            if include_warnings:
                severity_filter.append('warnings')
            if include_infos:
                severity_filter.append('infos')

        groups = defaultdict(lambda: {'focusNodes': [], 'count': 0})
        
        for severity in severity_filter:
            for result in details.get(severity, []):
                key = (result['message'], result['resultPath'], severity)
                groups[key]['focusNodes'].append(result['focusNode'])
                groups[key]['count'] += 1
        
        grouped_results = []
        for (message, path, severity), data in groups.items():
            focus_nodes = data['focusNodes']
            grouped_results.append({
                'message': message,
                'path': path,
                'severity': severity,
                'focusNodes': focus_nodes[:max_focus_nodes],
                'count': data['count'],
                'exampleNode': focus_nodes[0] if focus_nodes else None,
                'hasMoreNodes': len(focus_nodes) > max_focus_nodes
            })
        
        # Sort by severity (violations first) then count (descending)
        severity_order = {'violations': 0, 'warnings': 1, 'infos': 2}
        grouped_results.sort(key=lambda x: (severity_order[x['severity']], -x['count']))
        
        return grouped_results

    def _generate_report_entry(self, source: Union[str, Path], result: ValidationResult, file_hash: str = None) -> Dict:
        """
        Construct validation report entry with severity-organized structure
        
        Args:
            source: Input file or identifier string
            result: ValidationResult object with details
            file_hash: Optional hash for deduplication
            
        Returns:
            Dictionary with complete validation metadata and results, organized by severity
        """
        if isinstance(source, Path):
            file_hash = file_hash or self._calculate_file_hash(source)
            size = source.stat().st_size
            source_str = str(source)
            file_name = source.name
        else:
            size = len(source.encode('utf-8'))
            source_str = source
            file_name = "inline_content"
            file_hash = file_hash or hashlib.md5(source.encode('utf-8')).hexdigest()

        severity_map = {
            'violations': 'violation',
            'warnings': 'warning',
            'infos': 'info'
        }
        severity_order = ['violation', 'warning', 'info']

        groups = self.group_validation_results(result.details, include_warnings=True, include_infos=True)
        
        severity_organized = {sev: [] for sev in severity_order}
        for group in groups:
            severity = severity_map.get(group['severity'], group['severity'])
            severity_organized[severity].append({
                "message": group['message'],
                "path": group['path'],
                "count": group['count'],
                "example_node": group['exampleNode'],
                "focus_nodes": group['focusNodes'],
                "has_more_nodes": group['hasMoreNodes']
            })

        results_by_severity = []
        for severity in severity_order:
            groups = severity_organized[severity]
            if not groups:
                continue
                
            numbered_groups = []
            for i, group in enumerate(groups, 1):
                numbered_group = group.copy()
                numbered_group["group_number"] = i
                numbered_groups.append(numbered_group)
                
            results_by_severity.append({
                "severity": severity,
                "count": len(groups),
                "groups": numbered_groups
            })

        counts = {
            "total_groups": sum(len(severity_organized[sev]) for sev in severity_order),
            "violation": len(severity_organized['violation']),
            "warning": len(severity_organized['warning']),
            "info": len(severity_organized['info'])
        }

        report_data = {
            "source": source_str,
            "file_name": file_name,
            "conforms": result.conforms,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "size": size,
            "hash": file_hash,
            "results": {
                "by_severity": results_by_severity,
                # Maintain original flat structure for backward compatibility
                "all_groups": [group for sev in severity_order for group in severity_organized[sev]]
            },
            "counts": counts
        }

        return report_data

    def _calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Compute MD5 hash of a file to identify duplicates.

        Args:
            file_path: File to hash.
            chunk_size: Bytes to read per chunk.

        Returns:
            MD5 hash string.
        """
        hash_md5 = hashlib.md5()
        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _generate_html_report(self, report_entry: Dict) -> str:
        """Generate HTML report using the new severity-organized structure."""
        def format_group(group, severity):
            severity_class = severity
            severity_title = severity.capitalize()
            
            nodes_list = "\n".join(
                f"<li>{escape(node)}</li>" 
                for node in group['focus_nodes']
            )
            
            more_nodes = ""
            if group['has_more_nodes']:
                more_nodes = f"<p>+ {group['count'] - len(group['focus_nodes'])} more nodes affected</p>"
            
            return f"""
            <div class="{severity_class}">
                <h3>{severity_title} Group {group['group_number']}</h3>
                <p><strong>Message:</strong> {escape(group['message'])}</p>
                <p><strong>Occurrences:</strong> {group['count']}</p>
                <p><strong>Path:</strong> {escape(group['path'])}</p>
                <p><strong>Example Node:</strong> {escape(group['example_node'])}</p>
                <p><strong>Affected Nodes ({len(group['focus_nodes'])} shown):</strong></p>
                <ul>{nodes_list}</ul>
                {more_nodes}
            </div>
            """
            
        groups_html = []
        for severity_section in report_entry['results']['by_severity']:
            severity = severity_section['severity']
            groups = severity_section['groups']
            
            severity_title = severity.upper()
            groups_html.append(f"""
            <h2>{severity_title}S ({len(groups)} groups)</h2>
            <hr>
            """)
            
            for group in groups:
                groups_html.append(format_group(group, severity))

        file_name = escape(report_entry.get('file_name', 'Unknown'))
        source = escape(report_entry.get('source', 'Unknown'))
        timestamp = escape(report_entry.get('timestamp', 'Unknown'))
        conforms = "true" if report_entry.get('conforms', False) else "false"
        conforms_text = "Yes" if report_entry.get('conforms', False) else "No"

        return f"""<!DOCTYPE html>
            <html>
                <head>
                    <title>SHACL Validation Report - {file_name}</title>
                    <style>
                        /* ... (keep existing styles the same) ... */
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>SHACL Validation Report</h1>
                        <div class="metadata">
                            <p><strong>File:</strong> {source}</p>
                            <p><strong>Validated at:</strong> {timestamp}</p>
                        </div>
                    </div>
                    
                    <div class="summary">
                        <h2>Summary</h2>
                        <p><strong>Conforms:</strong> <span class="conforms-{conforms}">{conforms_text}</span></p>
                        <p><strong>Total Issue Groups:</strong> {report_entry['counts']['total_groups']}</p>
                        <p><strong>Violations:</strong> {report_entry['counts']['violation']}</p>
                        <p><strong>Warnings:</strong> {report_entry['counts']['warning']}</p>
                        <p><strong>Informational:</strong> {report_entry['counts']['info']}</p>
                    </div>

                    <h2>Validation Results</h2>
                    {"".join(groups_html)}
                </body>
            </html>
        """

    def _generate_text_report(self, report_entry: Dict) -> str:
        """Generate text report"""
        lines = [
            "SHACL Validation Report",
            "=" * 60,
            f"File: {report_entry.get('source', 'Unknown')}",
            f"Validated at: {report_entry.get('timestamp', 'Unknown')}",
            f"Conforms: {'Yes' if report_entry.get('conforms', False) else 'No'}",
            "",
            "Summary:",
            "-" * 60,
            f"Total Issue Groups: {report_entry['counts']['total_groups']}",
            f"Violations: {report_entry['counts']['violation']}",
            f"Warnings: {report_entry['counts']['warning']}",
            f"Informational: {report_entry['counts']['info']}",
            ""
        ]

        for severity_section in report_entry['results']['by_severity']:
            severity = severity_section['severity'].upper()
            groups = severity_section['groups']
            
            lines.extend([
                f"{severity}S ({len(groups)} groups)",
                "-" * 60
            ])

            for group in groups:
                lines.extend([
                    f"Group {group['group_number']}:",
                    f"  Message: {group['message']}",
                    f"  Occurrences: {group['count']}",
                    f"  Path: {group['path']}",
                    f"  Example Node: {group['example_node']}",
                    f"  Affected Nodes: {len(group['focus_nodes'])} shown" + 
                    (f" (+ {group['count'] - len(group['focus_nodes'])} more)" 
                    if group['has_more_nodes'] else ""),
                    ""
                ])

        return "\n".join(lines)
            
    def _load_hashes_from_summary(self, summary_path: Path) -> Set[str]:
        """Load processed file hashes from summary file"""
        hashes = set()
        if summary_path.exists():
            try:
                with summary_path.open('r', encoding='utf-8') as f:
                    existing_entries = json.load(f)
                    if isinstance(existing_entries, list):
                        hashes.update(
                            entry['hash'] for entry in existing_entries 
                            if isinstance(entry, dict) and 'hash' in entry
                        )
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Could not read summary file: {e}")
        return hashes

    def _update_summary_file(self, summary_path: Path, new_entries: List[Dict]) -> None:
        """Update summary JSON file with new entries"""
        existing_entries = []
        
        if summary_path.exists():
            try:
                existing_data = summary_path.read_text(encoding="utf-8")
                if existing_data.strip():
                    existing_entries = json.loads(existing_data)
                    if not isinstance(existing_entries, list):
                        existing_entries = []
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Could not read summary file: {e}")
                existing_entries = []
        
        entry_lookup = {entry['hash']: entry for entry in existing_entries if isinstance(entry, dict) and 'hash' in entry}
        
        for entry in new_entries:
            if isinstance(entry, dict) and 'hash' in entry:
                entry_lookup[entry['hash']] = entry
        
        final_data = json.dumps(list(entry_lookup.values()), indent=2, ensure_ascii=False)

        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=summary_path.parent, delete=False) as tmp:
                tmp.write(final_data)
                temp_name = tmp.name
            Path(temp_name).replace(summary_path)
        except Exception as e:
            logger.exception(f"Failed to safely write summary file: {e}")

    def _save_reports(
        self, 
        output_dir: Path, 
        base_name: str, 
        result: ValidationResult,
        file_source: Path = None, 
        formats: Iterable[ReportFormat] = (ReportFormat.JSON, ReportFormat.TXT)
    ) -> None:
        """Save validation reports in all requested formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        entry = self._generate_report_entry(file_source or base_name, result)

        if ReportFormat.JSON in formats:
            json_path = output_dir / f"{base_name}_report.json"
            json_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")

        if ReportFormat.TXT in formats:
            txt_path = output_dir / f"{base_name}_report.txt"
            txt_content = self._generate_text_report(entry)
            txt_path.write_text(txt_content, encoding="utf-8")

        if ReportFormat.HTML in formats:
            html_path = output_dir / f"{base_name}_report.html"
            html_content = self._generate_html_report(entry)
            html_path.write_text(html_content, encoding="utf-8")

        summary_path = output_dir / "validation_summary.json"
        self._update_summary_file(summary_path, [entry])