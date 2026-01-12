import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# 1. 创建 PersistentClient，path 参数指定本地持久化目录
client = chromadb.PersistentClient(
    path="HEMA_chroma_db_qwen",        # 本地数据库文件夹（不存在会自动创建）
    settings=Settings(                 # 可选：额外配置
        chroma_db_impl="duckdb+parquet",
        allow_reset=True,              # 如需重置数据库，可开启
        anonymized_telemetry=False,    # 禁止上报匿名统计
    ),
    tenant=DEFAULT_TENANT,             # 多租户场景下的租户 ID，默认即可
    database=DEFAULT_DATABASE,         # 数据库名称，默认即可
)

# 2. 获取或新建一个 collection
collection = client.get_or_create_collection(
    name="medical_collection",
)

# 3. 查询带有特定 pdf_name 的所有记录
target_old = "EHA_EHA Guidelines on Management of Antithrombotic Treatments in Thrombocytopenic Patients With"
target_new = "EHA_EHA Guidelines on Management of Antithrombotic Treatments in Thrombocytopenic Patients With Cancer"

all_data = collection.get(
    include=["ids", "metadatas"],
    where={"pdf_name": target_old}
)

ids       = all_data["ids"]
metadatas = all_data["metadatas"]

# 4. 打印更新前的状态
print("----- 更新前状态 -----")
for idx, md in zip(ids, metadatas):
    print(f"ID={idx}  pdf_name='{md.get('pdf_name')}'")

# 5. 构造新的 metadatas 列表，替换 pdf_name
new_metadatas = []
for md in metadatas:
    old_name = md.get("pdf_name")
    if old_name == target_old:
        md["pdf_name"] = target_new
    new_metadatas.append(md)

# 6. 批量更新
collection.update(
    ids=ids,
    metadatas=new_metadatas
)
print(f"\n已对 {len(ids)} 条记录执行更新操作。")

# 7. 再次查询，验证变更是否生效
verify_data = collection.get(
    include=["ids", "metadatas"],
    where={"pdf_name": target_new}
)

verify_ids       = set(verify_data["ids"])
verify_metadatas = verify_data["metadatas"]

print("\n----- 更新后状态（只显示 pdf_name 已改的新记录） -----")
for idx, md in zip(verify_data["ids"], verify_metadatas):
    print(f"ID={idx}  pdf_name='{md.get('pdf_name')}'")

# 8. 对比数量
if set(ids) == verify_ids:
    print(f"\n✅ 所有 {len(ids)} 条记录的 pdf_name 均已成功更新为 '{target_new}'")
else:
    missing = set(ids) - verify_ids
    print(f"\n⚠️ 以下 ID 的记录未更新：{missing}")
