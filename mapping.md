### HF → DCAT-AP Mapping

The following tables show how Hugging Face metadata fields are mapped to DCAT-AP 3.0.0 classes and RDF properties in **hf2dcat**, including special handling for model-specific metadata, dataset-specific metadata, and distribution fields.

| **HF Metadata Field**                | **hf2dcat Field**               | **DCAT Class**                      | **RDF Property**                       | **Notes**                                                        |
| ------------------------------------ | ------------------------------- | ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| `id` / `modelId`                     | `id`                            | `dcat:Dataset`                      | `dct:title`, `dct:identifier`, `dct:creator`, `dct:publisher` | Identifier + title                                               |
| `description` or card                | `description`                   | `dcat:Dataset`                      | `dct:description`                      | Description from card                                            |
| `private`                            | `private`                       | `dcat:Dataset`                      | `dct:accessRights`                     | Determines availability                                          |
| `gated`                              | `gated`                         | `dcat:Dataset`                      | `dct:accessRights`                     | - |
| `disabled`                           | `disabled`                      | `dcat:Dataset`                      | `dct:accessRights`                     | - |
| `tags`                               | `tags`                          | `dcat:Dataset`                      | `dcat:keyword`                         | General tagging                                                   |
| `license` (from tags/card)           | `license`                       | `dcat:Distribution`                 | `dct:license`                          | Mapped to controlled vocabularies                                 |
| `language`                           | `language`                      | `dcat:Dataset`                      | `dct:language`                         | ISO language codes                                                |
| `region`                             | `region`                        | `dcat:Dataset`                      | `dct:spatial`                          | Region normalisation                                              |
| `arxiv`                              | `arxiv`                         | `dcat:Dataset`                      | `dct:isReferencedBy`                   | Scientific publication links                                      |
| `downloads`                          | `downloads`                     | `dcat:Dataset`                      | `schema:DownloadAction`                | schema:InteractionCounter                                         |
| `likes`                              | `likes`                         | `dcat:Dataset`                      | `schema:LikeAction`                    | schema:InteractionCounter                                         |
| `author`                             | `author`                        | `dcat:Dataset`                      | `dct:creator`                          | HF namespace used when available                                  |
| `sha`                                | `sha`                           | `dcat:Dataset`                      | `owl:versionInfo`                      | Commit hash                                                       |
| `lastModified`                       | `last_modified`                 | `dcat:Dataset`, `dcat:Distribution` | `dct:modified`                         | Timestamp                                                         |
| `createdAt`                          | `created_at`                    | `dcat:Dataset`, `dcat:Distribution` | `dct:issued`                           | Creation date                                                     |

### **Model-Specific Fields**

| **HF Field**                          | **hf2dcat Field**               | **DCAT Class**                      | **RDF Property**                       | **Notes**                                                        |
| -------------------------------------- | ------------------------------- | ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| `library_name`                        | `library_name`                  | `dcat:Dataset`                      | `schema:softwareRequirements`          | Framework: Transformers, SentenceTransformers, etc.              |
| `transformers_info`                 | `transformers_info`             | `dcat:Dataset`                      | `schema:additionalProperty`            | auto_model, custom_class, processor, pipeline_tag                |
| `config`                              | `config`                        | `dcat:Dataset`                      | `schema:additionalProperty`            | HF architectures, model_type, tokenizer_config                                            |
| `datasets`                            | `datasets`                      | `mls:Dataset`                      | `it6:trainedOn`                            | Training datasets                                                 |
| `base_model`                          | `base_model`                    | `dcat:Dataset`                      | `prov:wasDerivedFrom`                  | Model lineage                                                     |

### **Dataset-Specific Fields**

| **HF Field**                          | **hf2dcat Field**               | **DCAT Class**                      | **RDF Property**                       | **Notes**                                                        |
| -------------------------------------- | ------------------------------- | ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| `size_categories`                     | `size_categories`               | `dcat:Dataset`                      | `schema:additionalProperty`            | Small / medium / large                                           |
| `task_categories`                     | `task_categories`               | `dcat:Dataset` / `mls:Dataset`      | `dct:subject` → `mls:Task`             | Broad ML tasks                                                    |
| `task_ids`                            | `task_ids`                      | `dcat:Dataset` / `mls:Dataset`      | `dct:subject` → `mls:Task`             | Specific ML tasks                                                 |
| `modality`                            | `modality`                      | `dcat:Dataset`                      | `schema:additionalProperty`            | text / image / audio / multimodal                                 |

### **Distributions**

| **HF Field**                          | **hf2dcat Field**               | **DCAT Class**                      | **RDF Property**                       | **Notes**                                                        |
| -------------------------------------- | ------------------------------- | ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| `siblings` / `parquet_files`          | `distributions → name`          | `dcat:Distribution`                 | `dct:title`                            | File label                                                        |
| —                                      | `distributions → description`   | `dcat:Distribution`                 | `dct:description`                      | Description                                         |
| —                                      | `distributions → size`          | `dcat:Distribution`                 | `dcat:byteSize`                        | File size                                                         |
| —                                      | `distributions → fileExtension` | `dcat:Distribution`                 | `dcat:mediaType`, `dct:format`         | MIME type                                                         |
| —                                      | `distributions → accessURL`     | `dcat:Distribution`                 | `dcat:accessURL`                       | HF file page                                                       |
| —                                      | `distributions → downloadURL`   | `dcat:Distribution`                 | `dcat:downloadURL`                     | Direct URL                                                         |
| `usedStorage`                          | `used_storage`                  | `dcat:Distribution`                 | `dcat:byteSize`                        | Repo size                                                         |

### **General Metadata**

| **HF Field**                          | **hf2dcat Field**               | **DCAT Class**                      | **RDF Property**                       | **Notes**                                                        |
| -------------------------------------- | ------------------------------- | ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
|   id                                     | `hub_url`                       | `dcat:Dataset`                      | `dcat:landingPage`   `it6: hasRepository`                  | HF model/dataset page                                             |
| “Hugging Face” provenance              | provenance                      | `dcat:Dataset`                      | `dct:provenance`                       | Source                                                            |
| `README.md`                            | readme                          | `dcat:Dataset`                      | `foaf:page`                            | Documentation                                                      |
| —                                      | theme                            | `dcat:Dataset`                      | `dcat:theme`                           | Default = TECH                                                    |
| —                                      | accrualPeriodicity               | `dcat:Dataset`                      | `dct:accrualPeriodicity`               | IRREG                                                             |
| —                                      | conformsTo                       | `dcat:Dataset`                      | `dct:conformsTo`                       | DCAT-AP 3.0.0                                                     |


### RDF Prefixes Used

The tables above use the following standard RDF prefixes:

- **dct:** <http://purl.org/dc/terms/>
- **dcat:** <http://www.w3.org/ns/dcat#>
- **mls:** <http://www.w3.org/ns/mls#>
- **prov:** <http://www.w3.org/ns/prov#>
- **it6:** <http://data.europa.eu/it6/>
