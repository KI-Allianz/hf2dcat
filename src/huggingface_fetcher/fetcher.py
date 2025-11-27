import hashlib
import argparse
import logging
import json
import re
import random
import time
import requests
import threading
import types
import os
import shutil
import yaml
from typing import List, Dict, Optional, Union, Callable, TypeVar, Set, Any, Tuple
from huggingface_hub import HfApi, DatasetCard, ModelCard, DatasetInfo, ModelInfo
from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError
from datasets import get_dataset_infos
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import quote, urlparse

from dotenv import load_dotenv

# env_path = Path(__file__).resolve().parents[2] / ".env"
# load_dotenv(dotenv_path=env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


T = TypeVar("T")
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
EXCLUDED_DATASET_IDS = {
    "KakologArchives/KakologArchives", # no English description
    "ACCC1380/private-model", # no English description
    "jamesqijingsong/chengyu", # no English description
    "kuroneko5943/jd21", 
    # "nvidia/Nemotron-Personas-Japan", 
    "liwu/MNBVC", # unsafe dataset files
    "Derur/all-portable-apps-and-ai-in-one-url" # unsafe dataset files
}

@dataclass
class FetcherConfig:
    default_limit: int = 100 # Default limit for fetch in BATCH FETCH MODE
    overfetch_factor: float = 3 # Fetch more than desired limit to ensure the reach of desired limit
    rate_limit: float = 0.5
    max_rate_limit: float = 5.0 
    rate_limit_backoff_factor: float = 1.5
    rate_limit_recovery_factor: float = 0.9
    max_rate_limit_delay: int = 60
    max_retries: int = 3
    timeout: int = 30
    max_workers: int = 8
    max_concurrent_requests: int = 4  
    batch_size: int = 20
    jitter_factor: float = 0.2 
    max_parquet: int = 2
    enable_progress: bool = True

    cache_dir: str = ".hf_cache"
    cache_version: str = "v1"
    cache_ttl: int = 604800 # 7 days 
    

class HuggingfaceFetcher:
    """
    HuggingfaceFetcher is a metadata harvester for datasets and models
    hosted on the Hugging Face Hub.

    Features:
    - Fetches metadata for specific datasets/models or in bulk..
    - Retrieves details such as id, likes, license, language, downloads, tags, task categories, and related resources.
    - Sorts and limits results by popularity (downloads), if requested.
    - Saves results to JSON if an output directory is provided.

    Returns:
        A dictionary containing metadata, fetch parameters, and counts of datasets/models fetched.
    """
    def __init__(self, hf_token: Optional[str] = None, config: Optional[FetcherConfig] = None):

        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.hf_token) 
        self.fetch_url = "https://huggingface.co"

        self.config = config or FetcherConfig()

        self._last_request_time = 0
        self._consecutive_errors = 0
        self._request_semaphore = threading.Semaphore(self.config.max_concurrent_requests)

        self.default_limit = self.config.default_limit
        self.overfetch_factor = self.config.overfetch_factor

        self.base_rate_limit = self.config.rate_limit
        self.current_rate_limit = self.config.rate_limit
        self.rate_limit_backoff_factor  = self.config.rate_limit_backoff_factor
        self.rate_limit_recovery_factor = self.config.rate_limit_recovery_factor
        self.max_rate_limit_delay = self.config.max_rate_limit_delay
        self.max_retries = self.config.max_retries
        self.timeout = self.config.timeout
        self._init_custom_session()

        self._cache_lock = threading.RLock() 
        self.cache_dir = Path(self.config.cache_dir or ".hf_cache") 
        self.cache_version =self.config.cache_version
        self.index_file = self.cache_dir / "index.json"
        self.payloads_dir = self.cache_dir / "payloads"
        self.cache_ttl = self.config.cache_ttl

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.payloads_dir.mkdir(exist_ok=True)

        self._cache_index = {}
        self._load_index()

    def _init_custom_session(self):
        """Initialize and configure the custom requests session."""
        self.custom_session = requests.Session()
        headers = {
            "User-Agent": "HuggingfaceFetcher/2.0",
            "Accept": "application/json",
        }
        
        if self.hf_token:
            self.custom_session.headers["Authorization"] = f"Bearer {self.hf_token}"
        
        self.custom_session.headers.update(headers)
        
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=0,
            pool_block=False
        )
        self.custom_session.mount("https://", adapter)
        self.custom_session.mount("http://", adapter)

    def close(self):
        if hasattr(self, 'custom_session'):
            self.custom_session.close()
        if hasattr(self, 'hf_token'):
            del self.hf_token
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
  
    def _rate_limiter(self):
        with self._request_semaphore:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.current_rate_limit:
                jitter = random.uniform(0, self.current_rate_limit * self.config.jitter_factor)
                sleep_time = self.current_rate_limit - elapsed + jitter
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def _handle_rate_limit_exceeded(self):
        """Adjust rate limiting when hitting API limits."""
        self._consecutive_errors += 1
        self.current_rate_limit = min(
            self.current_rate_limit * self.rate_limit_backoff_factor,
            self.config.max_rate_limit
        )
        logger.warning(
            f"Rate limit exceeded. Increasing delay between requests to {self.current_rate_limit:.2f}s"
        )
    
    def _reset_rate_limit(self):
        """Reset rate limiting after successful requests."""
        if self._consecutive_errors > 0:
            self._consecutive_errors = 0
            self.current_rate_limit = max(
                self.base_rate_limit,
                self.current_rate_limit * self.rate_limit_recovery_factor
            )

    def _load_index(self):
        """Load the cache index if it exists"""
        with self._cache_lock:
            try:
                if self.index_file.exists():
                    with open(self.index_file, "r") as f:
                        self._cache_index = json.load(f)
                    logger.debug(f"Loaded cache index with {len(self._cache_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._cache_index = {}
    
    def _save_index(self):
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.index_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self._cache_index, f, indent=2, default=self._default_serializer)
            temp_file.replace(self.index_file)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def _get_cache_key(self, repo_id: str, obj_type: str) -> str:
        """Generate consistent cache key"""
        repo_id = re.sub(r'^https?://[^/]+/', '', repo_id)
        return f"{self.cache_version}:{obj_type}:{repo_id}"

    def _slugify(self, key: str) -> str:
        """Convert repo_id to filesystem-safe name"""
        if not key:
            logger.warning("Empty key provided for slugification")
            key_hash = hashlib.sha256(b"").hexdigest()[:6]
            return f"invalid-key-{key_hash}.json"

        try:
            clean_key = re.sub(r'^https?://[^/]+/', '', key).casefold()
            clean_key = re.sub(r'[\s\\/:._]+', '-', clean_key)
            clean_key = re.sub(r'[^\w-]', '', clean_key)
            clean_key = re.sub(r'-+', '-', clean_key).strip('-')

            if not clean_key:
                logger.warning(f"Key '{key}' normalized to empty string")
                clean_key = "invalid-key"

            max_base_length = 200 - len('-xxxxxx.json')
            base = clean_key[:max_base_length]

            key_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()[:6]
            return f"{base}-{key_hash}.json"

        except Exception as e:
            logger.error(f"Slugify failed for key '{key}': {e}")
            key_hash = hashlib.sha256(key.encode('utf-8') if key else b"").hexdigest()[:6]
            return f"invalid-key-{key_hash}.json"

    def _get_payload_path(self, key: str) -> Path:
        """Generate human-readable payload filename"""
        return self.payloads_dir / self._slugify(key)
    
    def _get_cached(self, repo_id: str, obj_type: str) -> Optional[Dict]:
        """Retrieve cached metadata if valid"""
        key = self._get_cache_key(repo_id, obj_type)

        with self._cache_lock:
            try:
                entry = self._cache_index.get(key)
                if not entry or not self._validate_cache_entry(entry, repo_id, obj_type):
                    return None

                payload_path = self.cache_dir / entry["payload_path"]
                if not payload_path.exists():
                    return None

                with open(payload_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cached {key}: {e}")
                self._remove_cache_entry(key)
                return None

    def _remove_cache_entry(self, key: str):
        """Remove a cache entry and its payload file."""
        with self._cache_lock:
            try:
                entry = self._cache_index.pop(key, None)
                if entry and 'payload_path' in entry:
                    payload_path = self.cache_dir / entry['payload_path']
                    if payload_path.exists():
                        payload_path.unlink()
                self._save_index()
            except Exception as e:
                logger.warning(f"Failed to remove cache entry {key}: {e}")

    def _save_to_cache(self, repo_id: str, obj_type: str, metadata: Dict):
        """Save metadata to cache"""
        key = self._get_cache_key(repo_id, obj_type)
        temp_suffix = ".tmp"
        etag = self._get_current_etag(repo_id, obj_type)

        with self._cache_lock:
            try:
                self.payloads_dir.mkdir(parents=True, exist_ok=True)
                
                payload_filename = self._slugify(key)
                final_path = self.payloads_dir / payload_filename
                temp_path = final_path.with_suffix(temp_suffix)
                
                with open(temp_path, "w", encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=self._default_serializer)
                
                temp_path.replace(final_path)
                
                temp_index = self.index_file.with_suffix(temp_suffix)
                self._cache_index[key] = {
                    "timestamp": int(time.time()),
                    "etag": etag,
                    "last_modified":   
                        metadata["last_modified"].isoformat()
                        if isinstance(metadata.get("last_modified"), datetime)
                        else metadata.get("last_modified"), 
                    "payload_path": str(final_path.relative_to(self.cache_dir)),
                    "obj_type": obj_type,
                    # "expires_at": time.time() + self.cache_ttl,
                }
                
                with open(temp_index, "w", encoding='utf-8') as f:
                    json.dump(self._cache_index, f, indent=2)
                temp_index.replace(self.index_file)
                
            except Exception as e:
                logger.warning(f"Failed to cache {key}: {e}")
                if 'temp_path' in locals() and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                if 'temp_index' in locals() and temp_index.exists():
                    try:
                        temp_index.unlink()
                    except Exception:
                        pass
                raise

    def _clean_for_cache(self, data: Any) -> Any:
        """
        Recursively ensure data is cache-safe using our standard serializer.
        Called before putting items in the cache.
        """
        if isinstance(data, dict):
            return {k: self._clean_for_cache(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple, set)):
            return [self._clean_for_cache(v) for v in data]
        else:
            try:
                json.dumps(data, default=self._default_serializer)
                return data
            except (TypeError, ValueError) as e:
                logger.debug(f"Cleaning non-serializable data: {type(data)}")
                return str(data)

    def _is_cache_expired(self, cache_entry: Dict) -> bool:
        """Check if cache entry should be considered expired."""     
        # # Check explicit expiration first
        # if cache_entry.get('expires_at') and time.time() > cache_entry['expires_at']:
        #     return True
        
        # Fall back to default TTL
        cache_age = time.time() - cache_entry.get('timestamp', 0)
        return cache_age > self.cache_ttl

    def _get_current_etag(self, repo_id: str, obj_type: str) -> Optional[str]:
        """Get current ETag from API headers."""
        try:
            self._rate_limiter()
            url = (f"{self.fetch_url}/api/datasets/{repo_id}" 
                if obj_type == 'dataset' 
                else f"{self.fetch_url}/api/models/{repo_id}")
            
            headers = {'Accept': 'application/json'}
            if self.hf_token:
                headers['Authorization'] = f'Bearer {self.hf_token}'
            
            response = self.custom_session.head(url, headers=headers)
            response.raise_for_status()
            etag = response.headers.get('ETag')
            if not etag:
                logger.debug(f"No ETag header received for {repo_id}")
                return None

            if not re.match(r'^(W/)?"[^"]+"$', etag):
                logger.warning(f"Malformed ETag received for {repo_id}: {etag}")
                return None
                
            return etag
        except Exception as e:
            logger.debug(f"Failed to get ETag for {repo_id}: {e}")
            return None

    def _validate_cache_entry(self, cache_entry: Dict, repo_id: str, obj_type: str) -> bool:
        """Validate cached entry"""
        if not cache_entry:
            return False
        
        key = self._get_cache_key(repo_id, obj_type)
        if not key.startswith(self.cache_version):
            logger.debug(f"Invalid cache version (expected {self.cache_version})")
            return False
            
        # 1. Check if the entry expires 
        if self._is_cache_expired(cache_entry):
            return False
        
        # 2. Verify payload file exists
        payload_path = self.cache_dir / cache_entry.get('payload_path', '')
        if not payload_path.exists():
            logger.debug(f"Payload file missing at {payload_path}")
            return False
        
        # 3. Check ETag if available
        cached_etag = cache_entry.get('etag')
        if cached_etag:
            try:
                current_etag = self._get_current_etag(repo_id, obj_type)
                if current_etag:
                    return current_etag == cached_etag
            except Exception as e:
                logger.debug(f"Failed to verify ETag for {repo_id}: {e}")        

        # 4. Fallback to last_modified if available
        cached_last_modified = cache_entry.get('last_modified')
        if cached_last_modified:
            try:
                current_last_modified = self._get_current_last_modified(repo_id, obj_type)
                if current_last_modified:
                   return current_last_modified == cached_last_modified
            except Exception as e:
                logger.debug(f"Failed to verify last_modified for {repo_id}: {e}")
           
        return False

    def _get_current_last_modified(self, repo_id: str, obj_type: str) -> Optional[str]:
        """Get current last_modified timestamp from API"""
        try:
            self._rate_limiter()
            url = (f"{self.fetch_url}/api/datasets/{repo_id}" 
                  if obj_type == 'dataset' 
                  else f"{self.fetch_url}/api/models/{repo_id}")
            
            headers = {'Accept': 'application/json'}
            if self.hf_token:
                headers['Authorization'] = f'Bearer {self.hf_token}'
            
            response = self.custom_session.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            last_modified_meta = None
            if data:
                last_modified_meta = data.get("lastModified")     
            return last_modified_meta
        except Exception as e:
            logger.debug(f"Failed to get last_modified for {repo_id}: {e}")
            return None

    def _clear_cache(self):
        """
        Remove everything under self.cache_dir (index.json + payloads)
        and reset the in-memory index
        """
        with self._cache_lock:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.payloads_dir.mkdir(parents=True, exist_ok=True)
            self._cache_index = {}

    def _normalize_name_input(self, name_input: Union[str, List[str]]) -> List[str]:
        """
        Normalize a dataset or model name input into a list of non-empty, stripped strings.

        Args:
            name_input: A single string (e.g., "glue") or a list of strings (e.g., ["glue", "bert"]).

        Returns:
            A cleaned list of strings.
        """
        if isinstance(name_input, str):
            return [name_input.strip()]
        elif isinstance(name_input, list):
            return [str(n).strip() for n in name_input if str(n).strip()]
        else:
            raise TypeError("Expected a string or list of strings")
    
    def _normalize_license(self, license_field):
        """Normalize HF license field into a list of lowercase strings."""
        if not license_field:
            return []
        if isinstance(license_field, str):
            return [license_field.lower()]
        if isinstance(license_field, list):
            return [str(l).lower() for l in license_field if l]
        return []
    
    def _is_restricted(self, item: Dict) -> Tuple[bool, Optional[List[str]]]:
        """
        Check whether a dataset/model is restricted

        Args: 
            item: metadata dict of a dataset/model

        Return (is_restricted, reasons) where:
            - is_restricted = True if dataset/model is disabled, private, gated, has a closed license or 
                    has no license at all (unless in LICENSE_EXCEPTIONS)
            - reasons = list of strings explaining why
        """

        if not item:
            return False, None

        reasons = []

        if item.get("disabled"):
            reasons.append("disabled=True")
        if item.get("private"):
            reasons.append("private=True")
        if item.get("gated"):
            reasons.append("gated=True")

        item_id = item.get("id")

        if item_id in LICENSE_EXCEPTIONS:
            return False, None

        licenses = self._normalize_license(item.get("license"))
        if not licenses:
            reasons.append("no license info")
        else: 
            closed_licenses = [l for l in licenses if l in NON_OPEN_LICENSES]
            if closed_licenses:
                reasons.append(f"closed license(s): {', '.join(closed_licenses)}")

        return (bool(reasons), reasons if reasons else None)

    def _apply_filter(self, items, kind: str):
        """Filter out restricted datasets or models"""
        kept, removed = [], []
        for itm in items:
            restricted, reasons = self._is_restricted(itm)
            if restricted:
                removed.append({"id": itm.get("id", "<unknown>"), "reasons": reasons, "metadata": itm})
            else:
                kept.append(itm)
        if removed:
            for r in removed:
                logger.info(f"Filtered out {kind} {r['id']} due to: {', '.join(r['reasons'])}")
        return kept, removed
    
    def _fetch_items(self, names: List[str], fetch_func, kind: str):
        """Fetch a list of datasets or models by name, separating successes and failures."""
        successes, failures = [], []
        for name in names:
            try:
                res = fetch_func(name)
                if isinstance(res, dict) and "error" in res:
                    logger.warning(
                        f"Failed to fetch {kind} '{res.get('id', name)}': {res.get('error')}"
                    )
                    failures.append(res)
                else:
                    successes.append(res)

            except Exception as e:
                logger.warning(
                    f"Exception while fetching {kind} '{name}': {str(e)}"
                )
                failures.append({"id": name, "error": str(e)})

        logger.info(
            f"Completed fetching {kind}s: {len(successes)} succeeded, {len(failures)} failed."
        )
        return successes, failures
    
    def fetch_and_process(
        self,
        fetch_type: str = "both",
        params: Optional[Dict] = None,
        dataset_name: Optional[Union[str, List[str]]] = None,
        model_name: Optional[Union[str, List[str]]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        sort_by_downloads: bool = True,
        filter_restricted: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Fetch and process metadata from Hugging Face Hub.

        Fetching Rules:
        1. Name Fetch Mode (takes precedence):
        - Triggered when `dataset_name`or `model_name` is specified
        - Fetches only the specific item(s)
        - Ignores `fetch_type`, `params`, and `sort_by_downloads`

        2. Batch Fetch Mode:
        - Triggered when neither `dataset_name` nor `model_name` is provided 
        - Uses `fetch_type` to determine whether to fetch datasets, models, or both (default both)
        - Accepts any additional `params` for filtering and sorting via Hugging Face Hub API
        - If no `limit` is provided in `params`, a default limit is enforced to prevent excessive fetches
        - If `sort_by_downloads=True` and `params` does not already include a `sort`, results are sorted by downloads (descending)

        Args:
            fetch_type: "dataset", "model", or "both" — only used in batch mode (default: both)
            params: Hugging Face API parameters (e.g. {"search": "AI", "limit": 5}); `full` is always set to True.
            dataset_name: Hugging Face repo ID of a single dataset (full, e.g. "nyu-mll/glue", or short, e.g. "glue").
            model_name: Hugging Face repo ID of a single model (full or short form).
            output_dir: Optional directory to save fetched results as a JSON file
            sort_by_downloads: If True, auto-sort by downloads when no sort is provided (default: True)
            - When no specific name is provided:
                - Uses fetch_type to determine what to fetch
                - Uses params for filtering/sorting
                - Enforece a default limit when no limit is provided in params
                - Respects sort_by_downloads flag 
            filter_restricted: if True, filter out datasets/models that are restricted, such as those that are gated, 
                priveate, disabled or have non-open licenses. Defaults to True.

        Returns:
            A tuple containing the path to the saved metadata file and the result dictionary with the structure:
            {
                "metadata": {...},         # Information about the fetch operation
                "counts": {...},           # Number of items fetched
                "fetched_metadata": {...}  # The actual fetched metadata for dataset and/or models
            }
        """
        # Initialize tracking variables
        start_time = time.time()
        datasets, models = [], []
        failed_datasets, failed_models = [], []
        failed_metadata = {}

        success = True
        error = None
        used_params = {}
        saved_path = None

        include_datasets = False
        include_models = False

        dataset_name_list = self._normalize_name_input(dataset_name) if dataset_name else []
        model_name_list = self._normalize_name_input(model_name) if model_name else []

        try:
            # --- Named fetch mode --- 
            if dataset_name_list or model_name_list:
                if params:
                    logger.info("API parameters ignored when fetching specific dataset or model")
                
                if dataset_name_list:
                    logger.info(f"Fetching {len(dataset_name_list)} dataset(s)...")
                    include_datasets = True
                    datasets, failed_datasets = self._fetch_items(dataset_name_list, self._fetch_single_dataset, "dataset")

                if model_name_list:
                    logger.info(f"Fetching {len(model_name_list)} model(s)...")
                    include_models = True
                    models, failed_models = self._fetch_items(model_name_list, self._fetch_single_model, "model")
           
            # BATCH FETCH MODE
            else:
                used_params = params.copy() if params else {} 
              
                # Enforce default batch limit 
                if 'limit' not in used_params:
                    used_params['limit'] = self.default_limit
                    logger.warning(f"No limit provided. Defaulting to limit={self.default_limit} to avoid excessive fetch.")

                # Apply automatic sorting if enabled
                if sort_by_downloads and 'sort' not in used_params:
                    used_params.update({
                        'sort': 'downloads',
                        'direction': -1
                    })
                    logger.debug("Auto-applied sorting by downloads in descending order for limited batch fetch.")
                
                # Validate fetch_type and fall back to "both" if invalid 
                if fetch_type not in {"dataset", "model", "both"}:
                    logger.warning(f"Invalid fetch_type '{fetch_type}'. Defaulting to 'both'.")
                    fetch_type = "both"

                # Execute batch fetches based on fetch_type
                if fetch_type in ["both", "dataset"]:
                    logger.info(f"Fetching datasets with params: {used_params}")
                    include_datasets = True
                    fetched_list = self._fetch_datasets(filter_restricted=filter_restricted, params=used_params) or []
                    for d in fetched_list:
                        if isinstance(d, dict) and "error" in d:
                            failed_datasets.append(d)
                        else:
                            datasets.append(d)
                    
                if fetch_type in ["both", "model"]:
                    logger.info(f"Fetching models with params: {used_params}")
                    include_models = True
                    fetched_list = self._fetch_models(filter_restricted=filter_restricted, params=used_params) or []
                    for m in fetched_list:
                        if isinstance(m, dict) and "error" in m:
                            failed_models.append(m)
                        else:
                            models.append(m)

        except Exception as e:
            logger.exception("Fetch and process operation failed")
            success = False
            error = str(e)
            self._reset_rate_limit()
        
        failed_total = len(failed_datasets) + len(failed_models)
        if failed_datasets:
            failed_metadata["datasets"] = failed_datasets
        if failed_models:
            failed_metadata["models"] = failed_models
         
        counts = {}
        fetched = {}
        total = 0
        filtered_out_total = 0
        filtered_out = {}

        if filter_restricted:
            for label, items, included in (
                ("datasets", datasets, include_datasets),
                ("models", models, include_models)
            ):
                if included:
                    kept, removed = self._apply_filter(items, label)
                    if kept:
                        fetched[label] = kept
                        counts[label] = len(kept)
                    if removed:
                        filtered_out[label] = removed
                        filtered_out_total += len(removed)
                    total += len(kept) + len(removed)
        else:
            for label, items, included in (
                ("datasets", datasets, include_datasets),
                ("models", models, include_models)
            ):
                if included:
                    fetched[label] = items
                    counts[label] = len(items)
                    total += len(items)

        if filter_restricted and filtered_out_total:
            counts["filtered_out"] = filtered_out_total
        
        if failed_total:
            counts["failed"] = failed_total

        if total == 0:
            if filter_restricted and filtered_out_total:
                msg = "❌ No datasets or models passed the filtering rules. All fetched items were filtered out as restricted."
            else:
                msg = "❌ No datasets or models were fetched and processed. Check inputs and filters, or review the logs for possible internal errors."
            logger.error(msg)
            raise RuntimeError(msg)
        else:        
            counts["total"] = total
        
            # Prepare results structure
            results = {
                "metadata": {
                    "fetcher": "HuggingfaceFetcher",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "success" if success else "failed",
                    "parameters": {
                        "mode": "name fetch" if dataset_name or model_name else "batch fetch",
                        "fetch_type": fetch_type if not (dataset_name or model_name) else None,
                        "params": used_params,
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "sort_by_downloads_applied": sort_by_downloads if not (dataset_name or model_name) and 'limit' in used_params else None
                    }
                },
                "counts": counts, 
                "fetched_metadata": fetched
            }

            if filter_restricted and filtered_out_total:
                results["filtered_metadata"] = filtered_out

            if failed_metadata:
                results["failed_metadata"] = failed_metadata

            if error:
                results["metadata"]["error"] = error

            if output_dir:
                saved_path = self._save_to_file(Path(output_dir), results)
                logger.info(f"Fetch results are saved to the file at: {saved_path}")
                logger.info(f"Counts: {results.get('counts')}")
                results["metadata"]["output_file"] = str(saved_path)

            return (saved_path, results)
    
    def _fetch_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        max_retries: Optional[int] = None,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs: Any
    ) -> Optional[T]:
        """
        Call `func(*args, **kwargs)` with retries and exponential backoff

        - Returns the function’s result on success
        - Returns None immediately on 404 (EntryNotFoundError or HTTP 404)
        - Retries on any HTTP status in `retryable_status_codes` (default 500–599)
        - Aborts immediately on other 4xx or non-HTTP exceptions

        Args:
            func:            API function to call (e.g., DatasetCard.load)
            *args, **kwargs: Arguments to pass to `func`
            max_retries:     How many total attempts
            base_delay:      Initial backoff in seconds (default 1.0)
            backoff_factor:  Multiplier per retry (default 2.0)

        Returns:
            The return value of `func`, or `None` if the resource was not found
        """
        max_retries = max_retries or self.max_retries
        # retryable_codes = set(range(500, 600)) | {429} 
        retryable_codes = {408, 429, 500, 502, 503, 504} 

        for attempt in range(1, max_retries + 1):
            try:
                self._rate_limiter()         

                if (
                    isinstance(func, types.MethodType)
                    and getattr(func, '__self__', None) is self.custom_session
                ):
                    method = func.__name__.upper()
                    logger.debug(f"Using session.request for HTTP method: {method}")
                    result = self.custom_session.request(method, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Handle HTTP responses
                if hasattr(result, 'status_code'):
                    if result.status_code == 429:
                        retry_after = self._get_retry_after(result, base_delay)
                        logger.warning(f"Rate limited (attempt {attempt}/{max_retries}). Waiting {retry_after:.1f}s...")
                        time.sleep(retry_after)
                        continue
                    result.raise_for_status()
                
                self._reset_rate_limit()
                return result

            except requests.HTTPError as e:
                if not hasattr(e, 'response') or e.response is None:
                    logger.warning(f"HTTPError without response: {e}")
                    if attempt == max_retries:
                        return None
                    continue
                
                if e.response.status_code == 429:
                    retry_after = self._get_retry_after(e.response, base_delay)
                    logger.warning(f"Rate limited (HTTP 429, attempt {attempt}/{max_retries}). Waiting {retry_after:.1f}s...")
                    time.sleep(retry_after)
                    self._handle_rate_limit_exceeded()
                    continue
                    
                if e.response.status_code == 404:
                    logger.info(f"Resource not found (404): {args}")
                    return None

                if e.response.status_code in retryable_codes:
                    logger.warning(f"Retryable error {e.response.status_code} (attempt {attempt}/{max_retries})")
                else:
                    logger.error(f"Non-retryable error {e.response.status_code}: {e}")
                    return None

            except HfHubHTTPError as e:
                if getattr(e, 'status_code', None) == 429:
                    retry_after = self._get_retry_after(e, base_delay)
                    logger.warning(f"Hub rate limited (attempt {attempt}/{max_retries}). Waiting {retry_after:.1f}s...")
                    time.sleep(retry_after)
                    self._handle_rate_limit_exceeded()
                    continue
                logger.warning(f"HuggingFace Hub error: {str(e)}")
                if attempt == max_retries:
                    return None
            

            except EntryNotFoundError:
                logger.info(f"Resource not found (EntryNotFoundError): {args}")
                return None
            
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt}/{max_retries}: {str(e)}")
                if attempt == max_retries:
                    return None
            
            # Exponential backoff with jitter
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** (attempt - 1)) * random.uniform(0.8, 1.2)
                logger.debug(f"Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
        
        logger.error(f"Failed after {max_retries} attempts")
        return None

    def _get_retry_after(self, response_or_error, default: float = 2.0) -> float:
        """Extract Retry-After header with jitter and sanity checks."""
        try:
            if hasattr(response_or_error, 'headers'):
                retry_after = float(response_or_error.headers.get('Retry-After', default))
            else:
                retry_after = default
                
            # Apply jitter and ensure minimum/maximum bounds
            retry_after = retry_after * random.uniform(0.8, 1.2)  
            return max(1.0, min(retry_after, self.max_rate_limit_delay)) 
        except (AttributeError, ValueError):
            return default

    def _fetch_dataset_extras(self, dataset_id: str) -> Dict[str, Any]:
        """
        Fetch parquet file list and croissant schema metadata for a dataset.
        """
        extras = {}

        # Parquet list
        try:
            self._rate_limiter()
            resp = self.custom_session.get(
                f"https://datasets-server.huggingface.co/parquet?dataset={dataset_id}",
                timeout=self.timeout,
            )
            if resp.ok:
                data = resp.json()
                extras["parquet_files"] = data.get("parquet_files", [])
            else:
                # logger.info(f"[{dataset_id}] No parquet metadata (HTTP {resp.status_code})")
                extras["parquet_files"] = []
        except Exception as e:
            logger.warning(f"[{dataset_id}] Failed to fetch parquet metadata: {e}")
            extras["parquet_files"] = []

        # Croissant schema
        try:
            self._rate_limiter()
            resp = self.custom_session.get(
                f"https://huggingface.co/api/datasets/{dataset_id}/croissant",
                timeout=self.timeout,
            )
            if resp.ok:
                extras["croissant"] = resp.json()
            else:
                # logger.info(f"[{dataset_id}] No croissant metadata (HTTP {resp.status_code})")
                extras["croissant"] = {}
        except Exception as e:
            logger.warning(f"[{dataset_id}] No croissant metadata available: {e}")
            extras["croissant"] = {}

        return extras
    
    def _fetch_full_metadata_with_cache(self, repo_id: str, obj_type: str) -> Optional[Dict]:
        """Fetch with retry and cache results"""
        # # Check cache first
        cached = self._get_cached(repo_id, obj_type)
        if cached:
            return cached

        full_metadata = {}
        # Base info (dataset_info or model_info)
        base_info_func = self.api.model_info if obj_type == "model" else self.api.dataset_info
        base_info_obj = self._fetch_with_retry(
            base_info_func, repo_id=repo_id, files_metadata=True, timeout=self.timeout
        )
        if not base_info_obj:
            logger.warning(f"Failed to fetch {obj_type.capitalize()}Info for {repo_id}")
            return None
        full_metadata["base_info"] = base_info_obj

        # Parquet file list and croissant dataset only)
        if obj_type == "dataset": 
            extra_metadata = self._fetch_dataset_extras(repo_id)
            if extra_metadata:
                full_metadata.update(extra_metadata)
        
        # Process fetched HF metadata
        processed = self._process_hf_info(full_metadata, obj_type)
        if not processed:
            logger.warning(f"Failed to process metadata for {repo_id}")
            return None

        processed['timestamp'] = time.time()
        with self._cache_lock:
            try:
                logger.debug(f"Attempting to cache {repo_id}")
                self._save_to_cache(repo_id, obj_type, processed)
                logger.debug(f"Successfully cached {repo_id}")

                # # Verify cache write
                # verification = self._get_cached(repo_id, obj_type)
                # if not verification:
                #     logger.error(f"Cache verification FAILED for {repo_id}")
            except Exception as e:
                logger.error(f"Cache save failed for {repo_id}: {str(e)}")
                # Still return fresh data even if cache fails
                return processed

        return processed

    def is_valid_candidate(self, item, obj_type):
        """
        Prefiltering criteria:
        - Exclude if gated / private / disabled
        - Exclude if not in LICENSE_EXCEPTIONS AND
            • has no license
            • OR has any license in NON_OPEN_LICENSES
        """
        if obj_type == "dataset": 
            if item.id in EXCLUDED_DATASET_IDS:
                # logger.info(f"Excluded manually: {item.id}")
                return False
        if item.private or item.gated or item.disabled:
            return False

        if item.id in LICENSE_EXCEPTIONS:
            return True

        licenses = self.extract_license(item)

        if not licenses:
            return False
        
        if any(lic in NON_OPEN_LICENSES for lic in licenses):
            return False

        return True

    def extract_license(self, item):
        licenses = []
        for tag in (item.tags or []):
            if tag.startswith("license:"):
                licenses.append(tag.split("license:")[1].lower().strip())
        return licenses
    
    def _fetch_until_limit_reached(self, fetch_func, params: Dict, obj_type: str, filter_restricted: bool) -> List[Dict]:
        """
        Fetches items until the desired limit is reached
        
        Args:
            fetch_func: API function to fetch items (list_datasets/list_models)
            params: Dictionary of API parameters
            obj_type: Either "dataset" or "model"
            filter_restricted: If True, filters out restricted items (e.g., gated, private,
                disabled, or those with non-open licenses) after the initial fetch.
            
        Returns:
            List of metadata dictionaries up to the specified limit
        """
        desired_limit = params.get("limit", self.default_limit)
        overfetch_limit = max(desired_limit, int(desired_limit * self.overfetch_factor))
        batch_size = min(self.config.batch_size, 100)  

        logger.debug(f"Fetching {obj_type}s - Target: {desired_limit}, Overfetch: {overfetch_limit}")

        raw_items = []
        try:
            self._rate_limiter()
            # raw_items = list(fetch_func(**{**params, "limit": overfetch_limit, "full": True}))
            raw_items = list(fetch_func(**{**params, "limit": overfetch_limit}))
            if not raw_items:
                logger.warning("Initial fetch returned empty results")
                return []
            if filter_restricted: 
                # Prefilter restricted items 
                pre_kept = [
                    it for it in raw_items if self.is_valid_candidate(it, obj_type)
                ]
                # logger.info(f"Prefiltered items: {len(pre_kept)} / {len(raw_items)}")
                if pre_kept:
                    raw_items = pre_kept
        except Exception as e:
            logger.error(f"Initial fetch failed: {str(e)}")
            if hasattr(e, 'response') and e.response.status_code == 429:
                self._handle_rate_limit_exceeded()
            return []

        results = []
        seen_ids = set()
        lock = threading.Lock()
        rate_limit_sleep = max(0.1, self.current_rate_limit * 1.5)

        def fetch_metadata(item):
            """Thread-safe metadata fetcher with duplicate checking"""
            nonlocal results, seen_ids
            item_id = getattr(item, "id", None)
            if not item_id:
                return None

            with lock:
                if item_id in seen_ids:
                    return None
                seen_ids.add(item_id)  # Mark as seen early to prevent duplicates

            try:
                metadata = self._fetch_full_metadata_with_cache(item_id, obj_type)
                return metadata if metadata else None
            except Exception as e:
                logger.debug(f"Failed to fetch metadata for {item_id}: {str(e)}")
                return None
     
        # Process in parallel batches with progress tracking
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            batch_num = 0
            for i in range(0, len(raw_items), batch_size):
                if len(results) >= desired_limit:
                    break

                batch_num += 1
                current_batch = raw_items[i:i + batch_size]
                logger.debug(f"Processing batch {batch_num} ({len(current_batch)} items)")

                futures = {executor.submit(fetch_metadata, item): item for item in current_batch}

                for future in as_completed(futures):
                    if len(results) >= desired_limit:
                        # Cancel remaining futures in this batch
                        for f in futures:
                            f.cancel()
                        break
                    
                    try:
                        result = future.result()
                        if result:
                            with lock:
                                results.append(result)
                    except Exception as e:
                        logger.warning(f"Metadata fetch failed: {str(e)}")

                # Adaptive rate limiting
                if batch_num % 5 == 0 and len(results) < desired_limit / 2:
                    logger.debug("Adjusting rate limit due to slow progress")
                    rate_limit_sleep = min(rate_limit_sleep * 1.5, 5.0)  # Max 5 sec sleep
                
                time.sleep(rate_limit_sleep)

        # Final result processing
        success_rate = len(results) / desired_limit * 100
        if success_rate < 80:
            logger.warning(f"Low success rate: {success_rate:.1f}% ({len(results)}/{desired_limit})")
        else:
            logger.info(f"Fetched {len(results)} {obj_type}s (target: {desired_limit})")

        desired_results = results[:desired_limit]
        return desired_results

    def _camel_to_snake(self, name): 
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    def _strip_card_header(self, text: str, name: str = "") -> str:
        """
        Remove common autogenerated header lines from Hugging Face dataset/model descriptions.

        Specifically, this removes any lines that match:
        - Dataset Card for "xxx"' or 'Model Card for "xxx" (case-insensitive)
        - Dataset/Model Summary 
        - Dataset/Model Description
        - Summary or Description
        - Introduction
        - possibly "More Information Needed"
        - A line consisting of only the dataset or model name

        Args:
            text: The raw description text.
            name: The dataset or model’s name (to detect and drop exact name lines)

        Returns:
            Cleaned text with common autogenerated headings removed.
        """
        if not text:
            return ""

        lines = text.strip().splitlines()

        cleaned_lines = []
        # Patterns to drop
        patterns = [
            re.compile(r'^\s*Introduction\b\s*$'),
            re.compile(r'^\s*More Information Needed\s*$'), 
            re.compile(r'^\s*(?:Dataset|Model) Card for\b.*$', re.IGNORECASE),
            re.compile(r'^\s*(?:Dataset|Model) Card for\b.*$', re.IGNORECASE),
            re.compile(r"^\s*(Dataset|Model)?\s*Summary\s*$", re.IGNORECASE),
            re.compile(r"^\s*(Dataset|Model)?\s*Description\s*$", re.IGNORECASE),
            re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE),
            re.compile(r"^#+\s*$")  
        ]

        for line in lines:
            if any(pattern.match(line) for pattern in patterns):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _clean_description_safe(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _is_hf_autogenerated(self, text: str) -> bool:
        """Check if text looks like an autogenerated HF card"""
        patterns = [
            r"(?i)^#+\s*(dataset|model)\s+card\s+for\b",
            r"(?i)this\s+model\s+card\s+was\s+written",
            r"^-{3,}$", 
            r"^={3,}$", 
        ]
        lines = text.splitlines()
        for line in lines:
            if any(re.search(p, line) for p in patterns):
                return True
        return False
    
    def _clean_description(self, text: str, name: str = "") -> str:
        """
        Clean raw description from Hugging Face card content or backend fields.
        Removes horinzontal-rule lines, bullets, numbered lists and bod/italic markup,
            autogenerated headings and normalizes spacing.
        """
        if not text:
            return ""

        # Remove any horizontal-rule lines (---)
        text = re.sub(r'(?m)^\s*-{3,}\s*$', '', text)
        text = re.sub(r'(?m)^\s*={3,}\s*$', '', text)

        # Remove list-item bullets and numbered lists
        text = re.sub(r'(?m)^\s*[-*+]\s+', '', text)
        text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)

        # Unwrap bold/italic markup
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
        # text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text) 
        text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text) 

        # if name and self._is_hf_autogenerated(text):
        #     logger.info(f"auto generated: yes")
        #     text = self._strip_card_header(text, name)
        #     logger.info(f"stripped card header: {text}")
        
        # Strip card header regardless of name or hf autogenerated 
        text = self._strip_card_header(text, name)

        cleaned = self._clean_description_safe(text)
     
       # Fix mid-sentence periods before capital letters
        cleaned = re.sub(r'(?<=[a-z])\. (?=[A-Z])', '. ', cleaned)
        cleaned = re.sub(r'\.\s*\.', '.', cleaned)
        cleaned = re.sub(r'\s+([.?!])', r'\1', cleaned)
        
        return cleaned.strip()
    
    def _extract_backend_metadata(self, obj, declared_fields, known_aliases=None, name: str="") -> dict:
        """
        Extract backend-only metadata fields (i.e., not declared in dataclass).
        
        Args:
            obj: DatasetInfo or ModelInfo object.
            declared_fields: Set of known declared field names.
            known_aliases: Optional mapping of known camelCase → snake_case duplicates.

        Returns:
            Dictionary of backend-only attributes (normalized to snake_case).
        """
        backend_meta = {}
        declared_snake_case = {self._camel_to_snake(f) for f in declared_fields}
        known_aliases = known_aliases or {}
        raw_attrs = set(dir(obj)) - set(declared_fields)

        for attr in raw_attrs:
            # if attr.startswith("_") or attr in declared_fields or attr in ("tags", "card_data") or attr in known_aliases:
            if attr.startswith("_") or attr in ("tags", "card_data") or attr in known_aliases:
                continue

            snake_attr = self._camel_to_snake(attr)
            if snake_attr in declared_snake_case:
                continue

            try:
                value = getattr(obj, attr)
                if callable(value):
                    continue
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    if snake_attr == "description" and isinstance(value, str):
                        backend_meta[snake_attr] = self._clean_description(value, name)
                    else:
                        backend_meta[snake_attr] = value
                elif hasattr(value, "isoformat"):  
                    backend_meta[snake_attr] = value.isoformat()
                else:
                    backend_meta[snake_attr] = repr(value)
            except Exception:
                continue
        return backend_meta
 
    def _extract_base_metadata(self, obj, obj_type: str, parquet_files = [], include_backend: bool = False) -> dict:
        """
        Extract declared (and optionally backend) metadata from DatasetInfo or ModelInfo.
        
        Args:
            obj: DatasetInfo or ModelInfo object
            obj_type: "dataset" or "model"
            include_backend: whether to include backend-only fields 

        Returns:
            A metadata dictionary
        """
        meta = {}
        # Extract declared fields 
        declared_fields = set(getattr(obj, "__dataclass_fields__", {}).keys())  
       
        for field in declared_fields:
            # Excludes tags and card_data as they will be handled in _merge_card_tags
            if field in ("tags", "card_data"):
                continue
            try:
                value = getattr(obj, field)
                if callable(value):
                    continue
                elif hasattr(value, "isoformat"):
                    meta[field] = value.isoformat()
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    meta[field] = value
                else:
                    logger.debug(
                        f"Skipping unsupported type for {obj_type}.{field} "
                        f"({type(value).__name__}) on {getattr(obj, 'id', '<unknown>')}"
                    )
            except AttributeError as e:
                logger.debug(f"Field '{field}' not found in {obj_type} {getattr(obj, 'id', 'unknown')}: {str(e)}")
            except Exception as e:
                logger.warning(f"Unexpected error processing field '{field}' in {obj_type} {getattr(obj, 'id', 'unknown')}: {str(e)}", exc_info=True)

        meta["type"] = obj_type
        meta["name"] = meta.get("id", "").split("/")[-1]

        repo_id = meta.get("id", "")
        if not repo_id:
            logger.warning("Missing repo_id; skipping file metadata extraction.")
            return meta
        if repo_id in LICENSE_EXCEPTIONS:
            meta["license"] = LICENSE_EXCEPTIONS[repo_id]
        if hasattr(obj, "siblings") and isinstance(obj.siblings, list):
            file_map = {s.rfilename: s for s in obj.siblings if hasattr(s, "rfilename")}
            meta["file_count"] = len(file_map)
            meta["siblings"] = list(file_map.keys()) 
            
            # Add README.md as readme_url (applies to both models and datasets)
            readme_url = None
            for fname in file_map.keys():
                if fname.lower() in ("readme.md", "readme"):
                    if obj_type == "dataset":
                        readme_url = f"https://huggingface.co/datasets/{repo_id}/blob/main/{quote(fname)}"
                    else:  
                        readme_url = f"https://huggingface.co/{repo_id}/blob/main/{quote(fname)}"
                    meta["readme_url"] = readme_url
                    break  
            
            if obj_type == "model" and file_map:     
                model_distributions = self._build_model_distributions(repo_id, file_map, meta)
                meta["distributions"] = model_distributions
          
            if obj_type == "dataset" and (file_map or parquet_files):
                distributions = self._build_dataset_distributions(repo_id, file_map, parquet_files, meta)
                meta["distributions"] = distributions

        hub_path = f"{self.fetch_url}/datasets/{repo_id}" if obj_type == "dataset" else f"{self.fetch_url}/{repo_id}"
        meta["hub_url"] = hub_path

        if include_backend:
            KNOWN_ALIASES = {
                "modelId": "id"
            }

            backend_meta = self._extract_backend_metadata(
                obj,
                declared_fields,
                KNOWN_ALIASES, 
                meta["name"]
            )     
      
            meta.update(backend_meta)

        # Remove siblings as distribution info has already been extracted
        meta.pop("siblings", None)
        return meta

    def _build_model_distributions(self, repo_id, file_map, meta, file_count_limit=20):
        """Build model distributions with core weight files, config, tokenizer files and repo fallback."""

        CORE_WEIGHT_FILES = {
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "flax_model.msgpack",
            "onnx/model.onnx",
            "model.onnx",
            "rust_model.ot",
            "openvino_model.bin",
            "openvino_model.xml", 
            "model.gguf",
            "model.ggml"
        }
        CORE_CONFIG_FILES = {"config.json", "generation_config.json", "modules.json", "data_config.json"}
        CORE_TOKENIZER_FILES = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json", 
            "vocab.txt",
            "vocab.json", 
            "merges.txt",
            "spiece.model"
        }
        IGNORED_FILES = ["readme.md", "readme"]

        WEIGHT_VARIANTS = re.compile(r".*\.(safetensors|bin|onnx|msgpack|h5|ot|pt|pth|gguf|ggml)$")
        CONFIG_VARIANTS = re.compile(r".*config.*\.json$")
        TOKENIZER_VARIANTS = re.compile(
            r"(tokenizer.*\.json|.*tokens.*\.json|.*\.model|.*vocab.*\.(txt|json))"
        )
        WEIGHT_SHARD_PATTERN = re.compile(r".*-\d+-of-\d+\.(safetensors|bin|onnx|msgpack|h5|ot|pt|pth|gguf|ggml)$")
 
        def classify_model_file(fname: str) -> str:
            """
            Classify model repo file into categories: weight, config, tokenizer, or other.
            """
            name = Path(fname).name.lower()
            if WEIGHT_VARIANTS.match(name):
                return "weight"
            if CONFIG_VARIANTS.match(name) and "tokenizer" not in name:
                return "config"
            if TOKENIZER_VARIANTS.match(name):
                return "tokenizer"

            return "additional"

        repo_tree_url = f"https://huggingface.co/{repo_id}/tree/main"
        file_count = meta.get("file_count")
        used_storage = meta.get("used_storage")

        model_files = {
            fname: sibling
            for fname, sibling in file_map.items()
            if not Path(fname).name.startswith(".") 
            and fname.lower() not in IGNORED_FILES 
        }

        added_files = set()
        distributions = []
        model_file_count = len(model_files)
        canonical_weights = []
        shard_weights = []
        config_files = []
        tokenizer_files = []

        for fname in model_files:
            base_name = Path(fname).name
            if base_name in CORE_WEIGHT_FILES:
                canonical_weights.append(fname)
            elif WEIGHT_SHARD_PATTERN.match(base_name):
                shard_weights.append(fname)   
            elif base_name in CORE_CONFIG_FILES:
                config_files.append(fname)
            elif base_name in CORE_TOKENIZER_FILES:
                tokenizer_files.append(fname)
            
        # Add all files if repo is small 
        if model_file_count <= file_count_limit:
            for fname, sibling in model_files.items():
                # if fname.lower().startswith("checkpoint-"):
                #     continue
                ftype = classify_model_file(fname)
                size_value = getattr(sibling, "size", None)
                size = int(size_value) if size_value is not None else None
                size_info = f" (Size: ~{format_file_size(size)[0]})" if size else ""
                distributions.append({
                    "type": ftype, 
                    "name": Path(fname).name,
                    "slug": fname,
                    "description": f"{ftype.capitalize() + ' ' if ftype else ''}file: {fname}{size_info}",
                    "size": size,
                    "downloadURL": f"https://huggingface.co/{repo_id}/resolve/main/{quote(fname)}",
                    "accessURL": f"https://huggingface.co/{repo_id}/blob/main/{quote(fname)}",
                    "fileExtension": Path(fname).suffix.lstrip(".")
                })
                added_files.add(fname)
        else:
            if canonical_weights:
                for fname in canonical_weights:
                    sibling = model_files[fname]
                    size_value = getattr(sibling, "size", None)
                    size = int(size_value) if size_value is not None else None
                    size_info = f" (Size: ~{format_file_size(size)[0]})" if size else ""
                    distributions.append({
                        "type": "weight",
                        "name": Path(fname).name,
                        "slug": fname,
                        "description": f"Weight file: {fname}{size_info}",
                        "size": size,
                        "downloadURL": f"https://huggingface.co/{repo_id}/resolve/main/{quote(fname)}",
                        "accessURL": f"https://huggingface.co/{repo_id}/blob/main/{quote(fname)}",
                        "fileExtension": Path(fname).suffix.lstrip(".")
                    })
                    added_files.add(fname)
                
                if len(distributions) < file_count_limit:
                    for fname in config_files + tokenizer_files: 
                        if len(distributions) >= file_count_limit:
                                break  
                        sibling = model_files[fname]            
                        base = Path(fname).name 
                        size_value = getattr(sibling, "size", None)
                        size = int(size_value) if size_value is not None else None
                        size_info = f" (Size: ~{format_file_size(size)[0]})" if size else ""
                        distributions.append({
                            "type": "config" if base in CORE_CONFIG_FILES else "tokenizer",
                            "name": base,
                            "slug": fname,
                            "description": f"{'Config' if base in CORE_CONFIG_FILES else 'Tokenizer'} file: {fname}{size_info}",
                            "size": size,
                            "downloadURL": f"https://huggingface.co/{repo_id}/resolve/main/{quote(fname)}",
                            "accessURL": f"https://huggingface.co/{repo_id}/blob/main/{quote(fname)}",
                            "fileExtension": Path(fname).suffix.lstrip(".")
                        })
                        added_files.add(fname)

            elif shard_weights:
                preferred_ext = (".onnx", ".safetensors")
                preferred_shards = [f for f in shard_weights if Path(f).suffix.lower() in preferred_ext]
                other_shards = [f for f in shard_weights if Path(f).suffix.lower() not in preferred_ext]

                sample_shards = preferred_shards[:5]
                if len(sample_shards) < 5:
                    sample_shards += other_shards[: 5 - len(sample_shards)]

                for fname in sample_shards:
                    sibling = model_files[fname]
                    size_value = getattr(sibling, "size", None)
                    size = int(size_value) if size_value is not None else None
                    size_info = f" (Size: ~{format_file_size(size)[0]})" if size else ""
                    distributions.append({
                        "type": "weight",
                        "name": Path(fname).name,
                        "slug": fname,
                        "description": f"Shard weight file (Representative sample): {fname}{size_info}",
                        "size": size,
                        "downloadURL": f"https://huggingface.co/{repo_id}/resolve/main/{quote(fname)}",
                        "accessURL": f"https://huggingface.co/{repo_id}/blob/main/{quote(fname)}",
                        "fileExtension": Path(fname).suffix.lstrip(".")
                    })
                    added_files.add(fname)
                
                for fname in config_files + tokenizer_files: 
                    if len(distributions) >= file_count_limit:
                        break  
                    sibling = model_files[fname]            
                    base = Path(fname).name 
                    size_value = getattr(sibling, "size", None)
                    size = int(size_value) if size_value is not None else None
                    size_info = f" (Size: ~{format_file_size(size)[0]})" if size else ""
                    distributions.append({
                        "type": "config" if base in CORE_CONFIG_FILES else "tokenizer",
                        "name": base,
                        "slug": fname,
                        "description": f"{'Config' if base in CORE_CONFIG_FILES else 'Tokenizer'} file: {fname}{size_info}",
                        "size": size,
                        "downloadURL": f"https://huggingface.co/{repo_id}/resolve/main/{quote(fname)}",
                        "accessURL": f"https://huggingface.co/{repo_id}/blob/main/{quote(fname)}",
                        "fileExtension": Path(fname).suffix.lstrip(".")
                    })
                    added_files.add(fname)

        # Add repo as fallback distribution
        if not distributions or (len(added_files) < model_file_count):
            size_info = f" (Total size: ~{format_file_size(used_storage)[0]})" if used_storage else ""
            distributions.append({
                "type": "repo", 
                "name": f"All files for {repo_id}",
                "slug": "all-files",
                "description": f"Browse the repository and access "
                            f"{file_count if file_count else 'all'} files{size_info}",
                "accessURL": repo_tree_url,
                **({"fileCount": file_count} if file_count else {}),
                **({"size": used_storage} if used_storage else {})
            })

        return distributions

    def _parquet_tree_url(self, repo_id: str, example_parquet_url: str | None) -> str:
        """Derive parquet tree access url based on parquet download url"""
        if example_parquet_url and "/resolve/" in example_parquet_url:
            prefix = example_parquet_url.split("/resolve/")[0]  
            return prefix + "/tree/refs%2Fconvert%2Fparquet"
        return f"https://huggingface.co/datasets/{repo_id}/tree/refs%2Fconvert%2Fparquet"

    def _make_dist_slug(self, config=None, split=None, filename=None, extra=None):
        parts = []
        if config: parts.append(f"config={config}")
        if split: parts.append(f"split={split}")
        if filename: parts.append(f"file={filename}")
        if extra: parts.append(extra)
        slug = "-".join(parts) or "dist"
        return f"{slug}"

    def _build_dataset_distributions(self, repo_id, file_map, parquet_files, meta, file_count_limit=20):
        """
        Build dataset distributions for a Hugging Face dataset repo

        - Use HF API parquet metadata if available
        - Add non-parquet files if within file_count_limit
        - Ensure at least one repo-level distribution
        """
        data_files = {
            fname: sibling for fname, sibling in file_map.items()
            if not Path(fname).name.startswith(".") 
            and fname.lower() not in ("readme.md", "readme")
        }
        data_file_count = len(data_files)
        file_count = meta.get("file_count")
        used_storage = meta.get("used_storage")

        repo_tree_url = f"https://huggingface.co/datasets/{repo_id}/tree/main"

        distributions = []
        if parquet_files:
            parquet_file_count = len(parquet_files)
            if parquet_file_count <= file_count_limit:
                for p in parquet_files:       
                    size = int(size_str) if (size_str:=p.get("size")) is not None else None    
                    url = (p.get("url") or "").strip()
                    dataset_name = (p.get("dataset") or "").strip()
                    config = (p.get("config") or "").strip()
                    split = (p.get("split") or "").strip()
                    filename = (p.get("filename") or "").strip() or url.rsplit("/", 1)[-1]

                    parts = [(filename or "").strip()]
                    if config:
                        parts.append(f"Config: {config}")
                    if split:
                        parts.append(f"Split: {split}")                    
                    if size:
                        parts.append(f"Size: ~{format_file_size(size)[0]}")
                    
                    size_info = f" (~{format_file_size(size)[0]})" if size else ""
                    description = " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else "" 
                    description = parts[0] + description          
                    
                    access_url = url.replace("/resolve/", "/blob/") if "/resolve/" in url else None
                    slug = self._make_dist_slug(config=config, split=split, filename=filename)
                    distributions.append({
                        "type": "parquet-file", 
                        "name": filename,
                        "slug": slug, 
                        "description": description,
                        "size": size, 
                        "downloadURL": url, 
                        "accessURL": access_url,
                        "fileExtension": ".parquet"
                    })
            else:          
                # aggregate all parquet files
                total_size = sum(int(p.get("size") or 0) for p in parquet_files if p.get("size"))
                any_url = parquet_files[0].get("url")
                tree_url = self._parquet_tree_url(repo_id, any_url)
                size_info = f"; Total size: ~{format_file_size(total_size)[0]}." if total_size else ""
                distributions.append({
                    "type": "parquet-aggregate", 
                    "name": f"All Parquet files for {repo_id}",
                    "slug": "all-parquet",
                    "description": f"All Parquet files for {repo_id}. "
                                   f"({parquet_file_count} files{size_info})",
                    "size": total_size,
                    "fileCount": parquet_file_count,
                    "accessURL": tree_url,
                    "fileExtension": ".parquet"
                })

        if data_file_count <= file_count_limit:
            for fname, sibling in data_files.items():
                suffixes = Path(fname).suffixes
                ext = "".join(suffixes).lower() if suffixes else ""

                if parquet_files and ext == ".parquet":
                    continue  
         
                base_name = Path(fname).name    
                size_value = getattr(sibling, "size", None)
                size = int(size_value) if size_value is not None else None
                size_info = f" (Size: ~{format_file_size(size)[0]})" if size is not None else ""
                description = f"File: {fname}{size_info}"
                distributions.append({
                    "type": "file", 
                    "name": base_name,
                    "slug": fname, 
                    "description": description, 
                    "size": size, 
                    "downloadURL": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quote(fname)}",
                    "accessURL": f"https://huggingface.co/datasets/{repo_id}/blob/main/{quote(fname)}",
                    "fileExtension": ext
                })
     
        # Add repo distribution 
        if not distributions: 
            size_info = f" (Total size: {format_file_size(used_storage)[0]})." if used_storage is not None else ""
            distributions.append({
                "type":"repo", 
                "name": f"All files for {repo_id}",
                "slug": "all-file", 
                "description": 
                    f"Browse the repository and access "
                    f"{file_count if file_count else 'all'} files{size_info}", 
                "accessURL": repo_tree_url,
                **({"fileCount": file_count} if file_count else {}),
                **({"size": used_storage} if used_storage else {})
            })

        return distributions
    
    def _parse_base_model_tag(self, tag: str) -> Optional[Dict[str, str]]:
        """
        Parses a base_model tag of the form:
           - base_model:<model_name>
           - base_model:<type>:<model_name>
        Returns dict like {'name': model_name, 'type': type} or None
        """
        
        if not isinstance(tag, str) or not tag.startswith("base_model:"):
            return None

        parts = tag.split(":")
        if len(parts) == 2:
            return {"name": parts[1], "type": "base"}
        elif len(parts) >= 3:
            return {"name": parts[-1], "type": parts[-2]}
        return None

    def _merge_card_and_tags(self, obj_type, card_data, tags_info):
        """
        Merge tags and card_data into a unified metadata dictionary.

        Args:
            obj_type: dataset or model
            card_data: DatasetCardData or ModelCardData object
            tags_info: List of tags from DatasetInfo/ModelInfo 

        Returns:
            Merged metadata dict
        """
        meta = {}
        # Extract tags from card_data object (if present)
        raw_card_tags = getattr(card_data, "tags", []) or []
        raw_card_datasets = getattr(card_data, "datasets", []) or [] 
        all_tags = set(self._ensure_list(raw_card_tags)) 
        all_datasets = set(self._ensure_list(raw_card_datasets)) 
        # Initialize tag containers
        tags_dataset = []       # dataset attribute from ModelInfo
        structured_tags = {}    # Fields of the form "key:val" in DatasetInfo or ModelInfo tags 
        unstructured_tags = []  # Simple fields in DatasetInfo or ModelInfo tags (will be categorized into tags in final metadata)
       
        # Keys to be kept for dcat mapping to fields other than keyword
        RESERVED_KEYS = {"arxiv", "doi", "license", "language", "region", "language_creators", "annotations_creators"}
    
        # Process tags from tags_info list
        tags_info = [tags_info] if isinstance(tags_info, str) else (tags_info or [])
        for tag in tags_info:
            if not isinstance(tag, str):
                continue
            if tag in all_tags:
                continue  
            if tag.startswith("dataset:"):
                tags_dataset = tag.split(":", 1)[1]
                if tags_dataset not in all_datasets:
                    all_datasets.add(tags_dataset)
            if tag.startswith("base_model:"):
                model_info = self._parse_base_model_tag(tag)
                if model_info:
                    existing = structured_tags.setdefault("base_model", [])

                    # Check if this model name already exists
                    existing_by_name = {m["name"]: m for m in existing}
                    current_name = model_info["name"]
                    current_type = model_info.get("type")

                    if current_name not in existing_by_name:
                        # First time seeing this base model name
                        existing.append(model_info)
                    else:
                        existing_model = existing_by_name[current_name]
                        existing_type = existing_model.get("type")

                        # Prefer detailed type over generic one
                        if (existing_type in [None, "base"]) and current_type not in [None, "base"]:
                            # Replace existing less-detailed entry
                            existing.remove(existing_model)
                            existing.append(model_info)
                continue
            elif ":" in tag:
                key, value = tag.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key in RESERVED_KEYS:
                    structured_tags.setdefault(key, []).append(value)
                else:
                    # Flatten everything else as a keyword
                    all_tags.add(f"{key}:{value}")
            else:
                unstructured_tags.append(tag)
                all_tags.add(tag)       
      
        # Remove dataset:* tags (since they duplicate meta["datasets"])
        all_tags = {t for t in all_tags if not t.startswith("dataset:")}

        meta["tags"] = sorted(all_tags)
        meta["datasets"] = sorted(all_datasets)
        # Convert card_data into a dictionary 
        card_dict = {}
        if card_data:
            if hasattr(card_data, "dict"):
                card_dict = card_data.dict(exclude_none=True) 
            else:
                card_dict = {k: v for k, v in vars(card_data).items() if v is not None}
  
        # Avoid duplicating tags and datasets (already handled above)
        for key in {"tags", "datasets"}:
            card_dict.pop(key, None)
        
        # Prefer values from card_data over tags_info for structured fields 
        for key, values in structured_tags.items():
            if key not in card_dict:
                card_dict[key] = values

        meta.update(card_dict)
        if "base_model" in meta and "base_model" in structured_tags:
            meta["base_model"] = structured_tags["base_model"]
        return meta
    
    def _extract_card_content(self, card, section: Optional[List[str]] = None, name: str = "") -> str:
        """
        Retrieve the full Markdown/text of a DatasetCard or ModelCard—and
        optionally extract only a named subsection of it.

        Args:
            card:       A DatasetCard or ModelCard instance (or None).
            section:    If provided, a list of section titles to look for
                        (e.g. ["Dataset description", "Description"])—only
                        the text under the first matching heading (to the next
                        same‐level heading) will be returned.

        Returns:
            If `section` is None: the entire card.content or card.text, or ""
            If `section` is a list of titles: only the prose under that heading
                 (cleaned via `_clean_description`), or "" if not found.
        """
        if not card:
            return ""
        content = getattr(card, "content", "") or getattr(card, "text", "") or ""
        if section and not content.strip():
            return ""
        if section:
            text = self._extract_markdown_section(content, section)
            if text:
                return text
            # For description section, extract a sentence that contains "name is a ...." or the first sentence
            lower_titles = {t.strip().lower() for t in section}
            desc_keys = {"description", "dataset description", "model description", 
                "dataset summary", "model summary", "introduction"}
            if lower_titles & desc_keys:
                return self._extract_first_sentence(content, name=name)
            return ""

        return content
    
    def _extract_markdown_section(self, md: str, titles: List[str]) -> str:
        """
        Scan a Markdown blob for one of the given section titles and return just
        that section’s body.

        Look for a Markdown heading like:
            ## Dataset or Model description
            ### Description 
            ## 1. Dataset Description
        (case‐insensitive), then grab every line until the next heading of the
        same or higher level.

        Args:
            md:      The full Markdown string to scan.
            titles:  A list of heading texts (e.g. ["Model description", "Description"]).

        Returns:
            The raw text under the first matching heading (joined with newlines),
            passed through `_clean_description` to normalize whitespace/punctuation
        """
        lines = md.splitlines()
        section_lines = [] 
        capture = False
  
        # Markdown heading pattern (e.g. ## Dataset Description)
        heading_re = re.compile(r'^\s{0,3}(#+)\s*(.+?)\s*(?:#+\s*)?$')
        target = {t.lower() for t in titles}

        for line in lines:
            # Check if the line is heading
            m = heading_re.match(line)
            if m:
                # heading_text = m.group(2).strip().lower()
                heading_text = m.group(2).strip()
                heading_text = re.sub(r"^\d+\.\s*", "", heading_text).lower()
                if heading_text in target:
                    # Start capturing from the next line
                    capture = True
                    continue
                # Stop when already captuing but hitting a new heading
                if capture:
                    break
            elif capture:
                section_lines.append(line)
        
        raw = "\n".join(section_lines).strip()
        return self._clean_description(raw, name="")

    def _extract_first_sentence(self, text: str, name: str = "") -> str:
        """
        If there's a sentence like "<name> is a ...", return it. 
        Otherwise return the very first sentence of text.
        """
        # try to find "<Name> is a ..." (case‐insensitive)
        pattern = rf"\b{re.escape(name)}\b.*?[\.!?]"
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(0).strip()
        # otherwise just grab up to the first period
        m2 = re.match(r"\s*(.+?[\.!?])", text.replace("\n", " "), flags=re.DOTALL)
        return m2.group(1).strip() if m2 else text.strip()

    def _process_hf_info(self, raw_metadata: Dict, obj_type: str) -> Dict:
        """
        Normalize a Hugging Face (HF) raw metadata + README card into metadata dict.
        
        Args:
            raw_metadata: fetched HF metadata including ModelInfo or DatasetInfo, parquet files and croissant scheme metadata
            obj_type:      "dataset" or "model".
        
        Returns:
            A cleaned metadata dict, including:
            - declared + backend fields (_extract_base_metadata)
            - merged tags/card_data (_merge_card_and_tags)
            - description fallback from README
            - full README content in "card_content"
        """
        try:
            if not raw_metadata or not isinstance(raw_metadata, dict):
                logger.warning("No raw metadata provided to _process_hf_info")
                return {}

            info_obj = raw_metadata.get("base_info")
            parquet_files = raw_metadata.get("parquet_files", [])
            croissant = raw_metadata.get("croissant")

            meta = {}

            if info_obj:
                # Retrieve declared fields; include backend fields if "include_backend" is True
                meta = self._extract_base_metadata(info_obj, obj_type=obj_type, parquet_files=parquet_files, include_backend=True)
            
                # Merge info from tags and card_data
                tags = getattr(info_obj, "tags", [])
                # logger.info(f"tags from DatasetInfo or ModelInfo: {tags}")
                card_data = getattr(info_obj, "card_data", None)
                merged = self._merge_card_and_tags(obj_type, card_data, tags)
                if merged:
                    meta.update(merged)

                # If description missing, try to load it from the README card
                if not meta.get("description"):
                    if obj_type == "dataset" and croissant:
                        croi_desp = croissant.get("description")
                        if croi_desp:
                            meta["description"] = self._clean_description(croi_desp.strip())
                    
                    if not meta.get("description"): 
                        # Pick the right Card loader + section titles for description
                        CardClass = DatasetCard if obj_type == "dataset" else ModelCard
                        section_titles = (
                            ["Dataset description", "Dataset summary", "Description", "Introduction"]
                            if obj_type == "dataset"
                            else ["Model description", "Model summary", "Description", "Introduction"]
                        )

                        # Load DatasetCard (repo README.md) (if any)
                        try:
                            self._rate_limiter()
                            card = self._fetch_with_retry(CardClass.load, info_obj.id)
                            if card:
                                md = card.content or card.text or ""
                                description = self._extract_markdown_section(md, section_titles)
                                if description:
                                    meta["description"] = self._clean_description(description)
                        
                                # meta["card_content"] = self._extract_card_content(card)    
                        except yaml.YAMLError as e:
                            logger.warning(f"YAML parse error in card for {info_obj.id}: {e}. Skipping YAML front-matter.")  
                        except HfHubHTTPError as e:
                            if e.status_code == 404:
                                logger.warning(f"No {CardClass.__name__} for {info_obj.id}")
                            else:
                                logger.exception(f"Error loading {CardClass.__name__} for {info_obj.id}")
                        except Exception:
                            logger.exception(f"Unexpected error loading {CardClass.__name__} for {info_obj.id}")
                
            # Extra processing of metada results:
            # 1. Remove language codes from tag if present in language list
            if "language" in meta and isinstance(meta["language"], list) and "tags" in meta:
                language_set = set(str(lang).strip().lower() for lang in meta["language"])
                cleaned_tags = [
                    tag for tag in meta["tags"]
                    if str(tag).strip().lower() not in language_set
                ]
                meta["tags"] = sorted(set(cleaned_tags))

            # logger.info(f"meta with tags before adding datasets into the tags: {meta["tags"]}")
            # 2. Add datasets (if present) into tag if present
            # if "datasets" in meta and isinstance(meta["datasets"], list):
            #     meta["tags"] = sorted(set(meta.get("tags", []) + meta["datasets"]))
            
            if croissant:
                creator = croissant.get("creator")
                if creator: 
                    meta["croi_creator"] = creator

            # 3. Filter out meaningless values
            return {
                k: v
                for k, v in meta.items()
                if v is not None
                and not (isinstance(v, str)  and v == "")
                and not (isinstance(v, list) and len(v) == 0)
            }
        except Exception as e:
            logger.error(f"Processing failed for {obj_type} {getattr(info_obj, 'id', 'unknown')}: {e}")
            return None

    def _fetch_single_dataset(self, dataset_name: str) -> Dict:
        """
        Fetch metadata for a specific dataset by name

        Args:
            dataset_name: Hugging Face repo ID of the dataset, full or short form.

        Returns:
            A dict with at least:
            - "id"           : Hugging Face dataset ID
            - "type"         : "dataset"
            - declared fields from DatasetInfo
            - merged card_data fields
            - "description"  : from card.data or extracted from README.md
            - "card_content" : full markdown of README.md
            - "error"        : only if something irrecoverable happened
        """
        try:

            full_metadata = {}
            # Fetch DatasetInfo
            self._rate_limiter()
            ds_info = self._fetch_with_retry(self.api.dataset_info, repo_id=dataset_name, files_metadata=True,timeout=self.timeout)  
            if not isinstance(ds_info, DatasetInfo):
                msg = f"Failed to fetch dataset {dataset_name}: received no valid DatasetInfo (got {type(ds_info)})"
                logger.error(msg)
                return {"id": dataset_name, "error": msg, "type": "dataset"}
            
            full_metadata["base_info"] = ds_info
            extra_metadata = self._fetch_dataset_extras(dataset_name)
            if extra_metadata:
                full_metadata.update(extra_metadata)
        
            # Process and normalize DatasetInfo into metadata dict
            return self._process_hf_info(full_metadata, obj_type="dataset")

        except HfHubHTTPError as e:
            logger.exception(f"Failed to fetch dataset {dataset_name} after retries: {e}")
            return {
                "id": dataset_name,
                "error": f"Failed to fetch: {str(e)}",
                "type": "dataset"
            }
        except Exception as e:
            logger.exception(f"Unexpected error fetching dataset {dataset_name}: {e}")
            return {
                "id": dataset_name,
                "error": f"Unexpected error: {str(e)}",
                "type": "dataset"
            }

    def _fetch_single_model(self, model_name: str) -> Dict:
        """
        Fetch metadata for a specific model by name

        Args:
            model_name: Hugging Face repo ID of the model, full or short form.

        Returns:
            A dict with at least:
            - "id"           : Hugging Face model ID
            - "type"         : "model"
            - declared fields from ModelInfo
            - merged card_data fields
            - "description"  : from card.data or extracted from README.md
            - "card_content" : full markdown of README.md
            - "error"        : only if something irrecoverable happened
        """
        try:
            # Fetch ModelInfo
            self._rate_limiter()
            model_info = self._fetch_with_retry(self.api.model_info, repo_id=model_name, files_metadata=True, timeout=self.timeout)
            if not isinstance(model_info, ModelInfo):
                msg = f"Failed to fetch model {model_name}: received no valid ModelInfo (got {type(model_info)})"
                logger.error(msg)
                return {"id": model_name, "error": msg, "type": "model"}
            
            full_metadata = {}
            # Process and normalize ModelInfo into metadata dict
            if model_info: 
                full_metadata["base_info"] = model_info
            return self._process_hf_info(full_metadata, obj_type="model")
                   
        except HfHubHTTPError as e:
            logger.exception(f"Failed to fetch model {model_name} after retries: {e}")
            return {
                "id": model_name,
                "error": f"Failed to fetch: {str(e)}",
                "type": "model"
            }
        except Exception as e:
            logger.exception(f"Unexpected error fetching model {model_name}: {e}")
            return {
                "id": model_name,
                "error": f"Unexpected error: {str(e)}",
                "type": "model"
            }
        
    def _fetch_datasets(self, filter_restricted: bool, params: Optional[Dict] = None) -> List[Dict]:
        """
        Fetch metadata for multiple datasets with Hugging Face API parameters and exact full metadata
        
        Args:
            params: Dictionary of Hugging Face Hub API parameters. 
            filter_restricted: If True, filters out datasets that are restricted,
                such as those that are gated, private, disabled, or have non-open licenses.

        Returns:
            List of metadata dicts for datasets.
        """
        params = params or {}
        try:
            return self._fetch_until_limit_reached(
                fetch_func=self.api.list_datasets,
                params=params,
                obj_type="dataset",
                filter_restricted=filter_restricted
            )
        except Exception as e:
            logger.exception(f"Failed to fetch datasets: {e}")
            return []

    def _fetch_models(self, filter_restricted: bool, params: Optional[Dict] = None) -> List[Dict]:
        """
        Fetch metadata for multiple models with Hugging Face API parameters

        Args:
            params: Hugging Face Hub API parameters. 
            filter_restricted: If True, filters out datasets that are restricted,
                such as those that are gated, private, disabled, or have non-open licenses.
        
        Returns:
            List of metadata dicts for models.
        """
        params = params or {}
        try:
            return self._fetch_until_limit_reached(
                fetch_func=self.api.list_models,
                params=params,
                obj_type="model",
                filter_restricted=filter_restricted
            )
        except Exception as e:
            logger.exception(f"Failed to fetch models: {e}")
            return []

    def _ensure_list(self, data):
        """Normalize to list and filter for strings only"""
        if isinstance(data, List):
            normalized = data
        else:
            normalized = [data]
        return normalized

    def _default_serializer(self, obj):
        try:
            return obj.__dict__
        except AttributeError:
            return str(obj)
    
    def _sanitize_filename(self, name: str) -> str:
        """replace any non-alphanumeric or dash/underscore with underscore"""
        return re.sub(r"[^\w\-]+", "_", name).strip("_")

    def _save_to_file(self, output_dir: Path, results: Dict) -> Path:
        """Save fetched metadata results to a JSON file based on parameters.
        
        Args:
            output_dir: Directory in which to write the file.
            results:    The full dict of metadata to serialize.

        Returns:
            The Path to the file that was written.

        Raises:
            OSError:     If the directory cannot be created or the file cannot be written.
            TypeError:   If serialization fails (e.g. non-serializable objects).
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.exception(f"Could not create output directory {output_dir!r}")
            raise
    
        params = results["metadata"]["parameters"]
        mode = params.get("mode", "batch fetch")
        parts: list[str] = ["hf"] + [self._sanitize_filename(p.lower()) for p in mode.split()]
    
        if mode == "name fetch":
            dataset_name = params.get("dataset_name")
            model_name = params.get("model_name")
            if dataset_name:
                name_summary=f"{len(dataset_name)}_datasets" if len(dataset_name) > 1 else self._sanitize_filename(dataset_name[0])
                parts.append(name_summary)
            if model_name: 
                name_summary=f"{len(model_name)}_datasets" if len(model_name) > 1 else self._sanitize_filename(model_name[0])
                parts.append(name_summary)
        else:
            fetch_type = params.get("fetch_type", "both")
            parts.append(fetch_type)
            for key in ("search", "limit"):
                if key in params:
                    parts.append(f"{key}-{self._sanitize_filename(str(params[key]))}")
            if params.get("sort_by_downloads_applied"):
                parts.append("sorted")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{'_'.join(parts)}_{timestamp}.json"

        try:
            payload = json.dumps(results, ensure_ascii=False, indent=2, default=self._default_serializer)
            filename.write_text(payload, encoding="utf-8")
            return filename
        except TypeError as e:
            logger.exception("Failed to serialize results to JSON")
            return None
        except OSError as e:
            logger.exception(f"Failed to write file {filename!r}")
            return None
        except Exception as e:
            logger.exception(f"Failed to save results to file: {e}")
            return None


# Utility Functions
def format_file_size(size_bytes: Optional[int]) -> Tuple[str, Optional[float]]:
    if not size_bytes:
        return "0 bytes", 0.0
    elif size_bytes >= 1e9:
        return f"{round(size_bytes / 1e9, 2)} GB", round(size_bytes / 1e6, 2)
    elif size_bytes >= 1e6:
        return f"{round(size_bytes / 1e6, 2)} MB", round(size_bytes / 1e6, 2)
    elif size_bytes >= 1e3:
        return f"{round(size_bytes / 1e3, 2)} KB", round(size_bytes / 1e6, 2)
    else:
        return f"{size_bytes} bytes", round(size_bytes / 1e6, 2)

def run_fetcher(
    fetch_type: str,
    limit: Optional[int] = 10,
    params: Optional[Dict] = None,
    output_dir: Path = Path("output"),
    dataset_name: Optional[List[str]] = None,
    model_name: Optional[List[str]] = None,
    filter_restricted: Optional[bool] = True, 
) -> Tuple[str, Dict]:
    """
    Fetch Hugging Face datasets/models metadata, preprocess the metadata and save it to the given output directory.

    This function operates in one of two mutually exclusive modes:
    1. Name Fetch Mode (takes precedence)
    - Triggered when either dataset_name or model_name is provided.
    - Fetches only the specified dataset(s) or model(s).
    - Ignores fetch_type, limit, and params
 
    2. Batch Fetch Mode
    - Triggered when neither dataset_name nor model_name is provided.
    - Uses fetch_type to decide which resource type(s) to fetch:
         - dataset → fetch datasets only
         - model → fetch models only
         - both → fetch both datasets and models
    - Applies limit to control how many resources are retrieved.
    - Merges limit into params if params is provided and does not already contain limit.
    - Defaults: sort="downloads", direction=-1
    
    Rules
    - dataset_name or model_name triggers name fetch mode and overrides batch settings.
    - In name fetch mode, fetch_type, limit, and params are ignored.
    - In batch fetch mode, dataset_name and model_name must be `None`.
    - At least one mode must be selected; otherwise a `ValueError` is raised.

    Args: 
        fetch_type (str, optional): One of `"dataset"`, `"model"`, or `"both"` (batch mode only).
        limit (int, optional): Number of items to fetch in batch mode. Ignored in single mode.
        params (dict, optional): Additional query parameters for batch mode. Merged with {"limit": limit}.
        output_dir (Path): Directory where fetched metadata will be saved.
        dataset_name (str, optional): Name of a single dataset to fetch, e.g. "nyu-mll/glue". Activates single fetch mode.
        model_name (str, optional): Name of a single model to fetch, e.g. "google-bert/bert-base-uncased". Activates single fetch mode.
        filter_restricted (bool, optional): if True, filter out datasets/models that are restricted, 
            such as those that are gated, priveate, disabled or have non-open licenses. Defaults to True.

    Returns
        A tuple containing the path to the saved metadata file and the result dictionary.
    """
    if dataset_name or model_name:
        fetch_type = None
        limit = None
        params = None
        effective_params = None
        if dataset_name: 
            logger.info(f"Fetching metadata for dataset(s) named {dataset_name}")
        else: 
            logger.info(f"Fetching metadata for model(s) named {model_name}")
    else:
        if fetch_type not in ("dataset", "model", "both"):
            raise ValueError("In batch mode, fetch_type must be 'dataset', 'model' or 'both'.")
        effective_params = params.copy() if params else {}
        if limit is not None and "limit" not in effective_params:
            effective_params["limit"] = limit

    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = FetcherConfig(cache_dir=cache_dir)

    with HuggingfaceFetcher(config=config) as fetcher:
        try: 
            saved_path, result = fetcher.fetch_and_process(
                fetch_type=fetch_type,
                params=effective_params,
                dataset_name=dataset_name,
                model_name=model_name,
                output_dir=output_dir,
                filter_restricted=filter_restricted, 
                sort_by_downloads=True      
            )
            logger.info("✅ Fetch complete")
            return (saved_path, result)
        except Exception as e:
            logger.exception(f"Critical fetch failure: {e}")
            raise
