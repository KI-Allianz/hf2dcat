from huggingface_hub import list_models, list_datasets
from huggingface_hub import HfApi, ModelInfo, DatasetInfo


# print(f"---- lightweight ModelInfo ------")
# # Lightweight listing
# for m in list_models(limit=2, full=False):
#     print(m)

# # Full listing
# print(f"---- full ModelInfo ------")
# for m in list_models(limit=2, full=True):
    # print(m)

# print(f"---- lightweight DatasetInfo ------")
# count = 0
# license_count = 0
# limit = 1000
# sort_criteria = ["downloads", "likes", "likes7d"]  
# all_dataset_ids = set()
# all_ids = {}
# for c in sort_criteria:
#     for d in list_datasets(limit=limit, sort=c):
#         if c in all_ids:
#             all_ids[c].append(d.id)
#         else:
#             all_ids[c] = [d.id]
#         all_dataset_ids.add(d.id)

# print(f"all dataset ids(set):{all_dataset_ids}")
# print(f"total in set: {len(all_dataset_ids)}")
# # print(f"category dataset ids: {all_ids}")       
# count = 0
# for d in list_datasets(limit=1000, sort="downloads", direction=-1):
#     if tags := d.tags:
#         # print(f"item: {d.tags}")
#         # print(f"\ndataset id: {d.id}")
#         license = []
#         for t in tags:
#             if t.startswith("license:"):
#                 license.append(t)
#         if license:
#             count += 1
#             # print(f"license: {license}")
#             if len(license) > 1:
#                 print(f"{d.id} has more than one licenses {license}")

# print(f"count: {count}")


# count = 0
# limit = 2
# for m in list_models(limit=limit, sort="downloads", direction=-1):
#     print(m)
#     # if tags := d.tags:
#     #     # print(f"item: {d.tags}")
#     #     # print(f"\ndataset id: {d.id}")
#     #     license = []
#     #     for t in tags:
#     #         if t.startswith("license:"):
#     #             license.append(t)
#     #     if license:
#     #         count += 1
#     #         # print(f"license: {license}")
#     #         if len(license) > 1:
#     #             print(f"{d.id} has more than one licenses {license}")

# print(f"count: {count}")

    
# print(f"---- full DatasetInfo ------")
# for d in list_datasets(limit=1, sort="downloads", direction=-1, full=True):
#     print(d)

# print(f"dataset info with files metadata True: \n{HfApi().dataset_info(repo_id="allenai/objaverse")}")

# api = HfApi()
# item = api.model_info("Qwen/Qwen2.5-3B")
# print(item)

api = HfApi()
item = api.dataset_info("Felix92/docTR-resource-collection")
print(item)

# datasets = list_datasets(limit=1000, sort="downloads", direction=-1)
# for d in datasets:
#     print(f"\nDataset: {d.id}")

#     # Inspect tags for language tags
#     if hasattr(d, "tags"):
#         langs = [tag.split(":", 1)[1] for tag in d.tags if isinstance(tag, str) and tag.startswith("language:")]
#         print("  language tags:", langs)


# models = list_models(limit=1, sort="downloads", direction=-1)
# for m in models:
#     print(f"\nModel: {m.id}")
#     print(f"{m}")

#     # Inspect tags for language tags
#     if hasattr(m, "tags"):
#         langs = [tag.split(":", 1)[1] for tag in m.tags if isinstance(tag, str) and tag.startswith("language:")]
#         print("  language tags:", langs)

