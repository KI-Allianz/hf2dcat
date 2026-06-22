# HF → DCAT-AP and Machine Learning Metadata Mapping

This document describes the metadata mappings implemented by **hf2dcat** for converting Hugging Face dataset and model metadata into DCAT-AP 3.0.0 compliant RDF enriched with machine-learning-specific semantics from complementary vocabularies.

hf2dcat uses DCAT-AP 3.0.0 as its primary application profile for representing datasets, models, distributions, agents, licenses, provenance, access rights, and related metadata in an interoperable manner.

Because Hugging Face resources contain machine-learning-specific metadata that is not covered by DCAT-AP, hf2dcat additionally reuses concepts and properties from complementary machine-learning metadata models and vocabularies, including:
- Machine Learning DCAT-AP (MLDCAT-AP)
- Machine Learning Schema (MLS)
- Machine Learning Sailor Ontology (MLSO)
- Linked Papers with Code (LPWC)

These vocabularies and application profiles are used to represent machine-learning-specific concepts such as task types, model architectures, modalities, implementations, software libraries, training datasets, model provenance, engagement metrics, and related machine-learning metadata while maintaining compatibility with DCAT-AP.

## Core Metadata Mappings 

| HF Field | RDF Property | Subject Class | Role |
|-----------|------------|------------|------------|
| id | `dct:identifier` | `dcat:Dataset` / `mls:Model` | Unique identifier |
| id | `dct:title` | `dcat:Dataset` / `mls:Model` | Title |
| description | `dct:description` | `dcat:Dataset` / `mls:Model` | Description |
| author | `dct:creator` | `dcat:Dataset` / `mls:Model` | Creator (`foaf:Agent`) |
| repository owner | `dct:publisher` | `dcat:Dataset` / `mls:Model` | Publisher (`foaf:Agent`) |
| tags | `dcat:keyword` | `dcat:Dataset` / `mls:Model` | Keywords |
| sha | `owl:versionInfo` | `dcat:Dataset` / `mls:Model` | Version identifier |
| createdAt | `dct:issued` | `dcat:Dataset` / `mls:Model` | Creation date |
| lastModified | `dct:modified` | `dcat:Dataset` / `mls:Model` | Last update |
| homepage | `dcat:landingPage` | `dcat:Dataset` / `mls:Model` | Landing page |
| readme url | `foaf:page` | `dcat:Dataset` / `mls:Model` | Documentation page (`foaf:Document`) |
| arxiv / doi | `dct:isReferencedBy` | `dcat:Dataset` / `mls:Model` | Linked publications (arXiv, DOI) |
| likes / downloads | `schema:interactionStatistic` | `dcat:Dataset` / `mls:Model` | User engagement metrics (schema:InteractionCounter) |

## Dataset Metadata Mappings 

| HF Field | RDF Property | Subject / Object Class | Role |
|-----------|------------|------------|------------|
| task category | `lpwc:usedForTask` / `it6:hasTaskType` | `dcat:Dataset` / `it6:Task` / `it6:TaskType` | Dataset task category |
| modality | `mlso:hasModality` | `dcat:Dataset` / `skos:Concept` | Data modality |
| size category | `dct:subject` | `dcat:Dataset` / `skos:Concept` | Dataset size category |
| library | `dct:relation` / `it6:hasLibrary` | `it6:ComputerInfrastructure` / `it6:Library` | Compatible processing library |
| language | `dct:language` | `dcat:Dataset` | Dataset language |
| region | `dct:spatial` | `dcat:Dataset` / `skos:Concept` | Regional scope |

## Distribution Metadata Mappings 

| HF Source / Field | RDF Property | Subject / Object Class | Role |
|-----------|------------|------------|------------|
| Parquet files (datasets) | `dcat:distribution` | `dcat:Dataset` / `dcat:Distribution` | Dataset data files |
| Model files (weights, config, tokenizer) | `dcat:distribution` | `mls:Model` / `dcat:Distribution` | Model artifacts |
| Croissant JSON-LD | `dcat:distribution` | `dcat:Dataset` / `dcat:Distribution` | Croissant metadata |
| File URL | `dcat:accessURL` | `dcat:Distribution` | Access location |
| License | `dct:license` | `dcat:Distribution` | Distribution license (`dct:LicenseDocument`) |
| File format | `dct:format` | `dcat:Distribution` | File format |
| Media type | `dcat:mediaType` | `dcat:Distribution` | MIME type |

## Model Metadata

### Model Semantic Metadata Mappings 

| HF Field | RDF Property | Subject / Object Class | Role |
|-----------|------------|------------|------------|
| pipeline_tag | `lpwc:usedForTask` / `it6:hasTaskType` | `it6:MachineLearningModel` / `it6:Task` / `it6:TaskType` | Model task category represented as a lightweight task resource and linked to a task-type concept and Hugging Face task page |
| transformersInfo.pipeline_tag | `lpwc:usedForTask` / `it6:hasTaskType` | `it6:MachineLearningModel` / `it6:Task` / `it6:TaskType` | Runtime inference task category when different from the primary pipeline tag |
| config.model_type | `dct:relation` | `it6:MachineLearningModel` / `skos:Concept` | Links the model to the Hugging Face model-family concept |
| config.model_type | `skos:exactMatch` | `skos:Concept` / MLSO algorithm concept | Optional alignment to MLSO concepts |
| derived architecture | `it6:modelArchitecture` | `it6:MachineLearningModel` | High-level architecture derived from model family or implementation metadata |

### Model Implementation and Infrastructure Metadata Mappings

| HF Field | RDF Property | Subject / Object Class | Role |
|-----------|------------|------------|------------|
| config.architectures | `dct:relation` | `mls:Implementation` | Framework-specific model implementation class |
| transformersInfo.auto_model | `dct:relation` | `mls:Implementation` | Standard Hugging Face loader class |
| transformersInfo.custom_class | `dct:relation` | `mls:Implementation` | Custom implementation class |
| transformersInfo.processor | `dct:relation` | `schema:SoftwareApplication` | Processor or tokenizer used for preprocessing |
| config.model_type | `dct:relation` | `mls:Implementation` / `skos:Concept` | Associates implementation classes with the model-family concept |
| library_name | `dct:relation` / `it6:hasLibrary` | `it6:ComputerInfrastructure` / `it6:Library` | Runtime software library |
| number of parameters | `it6:numberOfParameters` | `it6:MachineLearningModel` | Model size |

### Model Engagement Metadata Mapping 

| HF Field | RDF Property | Subject / Object Class | Role |
|-----------|------------|------------|------------|
| likes / downloads | `it6:hasEngagement` | `it6:MachineLearningModel` / `it6:Engagement` | Model engagement metrics represented as engagement resources |

### Model Provenance Metadata Mappings 

| HF Field | RDF Property | Subject / Object Class | Role |
|-----------|------------|------------|------------|
| training dataset | `it6:trainedOn` | `it6:MachineLearningModel` | Training dataset reference |
| base model | `prov:wasDerivedFrom` | `it6:MachineLearningModel` | Base model reference |

## Derived Metadata Mappings 

Some RDF properties are not mapped directly from Hugging Face metadata but are derived, inferred, or assigned by hf2dcat to improve interoperability and DCAT-AP compliance.

| RDF Property | Source | Description |
|--------------|--------|-------------|
| `dct:accessRights` | Derived from `private`, `gated`, and `disabled` repository flags | Indicates repository accessibility and availability |
| `dcat:theme` | Assigned by hf2dcat | Default theme category (`TECH`) |
| `dct:accrualPeriodicity` | Assigned by hf2dcat | Default update frequency (`IRREG`) |
| `dct:conformsTo` | Assigned by hf2dcat | Indicates compliance with DCAT-AP 3.0.0 |
| `dct:provenance` | Derived from repository source information | Records Hugging Face as the metadata source |

## Metadata Normalization

hf2dcat uses generated mapping resources to normalize metadata values during RDF conversion.

| Mapping Resource | Purpose |
|------------------|---------|
| `hf2dcat_license_mappings.json` | Maps Hugging Face license identifiers to standardized license resources |
| `extension2_mediatype_filetype_mappings.json` | Maps file extensions to media types and file-format information |

### License Normalization

Licenses are represented using `dct:license` and mapped to controlled vocabulary resources whenever possible.

### Format Normalization

File extensions are mapped to media types and file-format information and represented using `dct:format` and `dcat:mediaType`.

### Vocabulary Normalization

Selected metadata values such as languages, regions, task categories, modalities, and model families are normalized to controlled vocabularies, identifiers, or RDF resources when applicable.

## RDF Prefixes

| Prefix | Namespace |
|----------|----------|
| dcat | `http://www.w3.org/ns/dcat#` |
| dct | `http://purl.org/dc/terms/` |
| foaf | `http://xmlns.com/foaf/0.1/` |
| prov | `http://www.w3.org/ns/prov#` |
| schema | `https://schema.org/` |
| skos | `http://www.w3.org/2004/02/skos/core#` |
| mls | `http://www.w3.org/ns/mls#` |
| mlso | `https://w3id.org/mlso#` |
| it6 | `http://data.europa.eu/it6/` |
| lpwc | `https://linkedpaperswithcode.com/property/` |
| owl | `http://www.w3.org/2002/07/owl#` |