# 血液肿瘤多学科会诊（MDT）Agent系统

这是一个基于AI Agent的多学科会诊（MDT, Multidisciplinary Team）系统，专门用于血液肿瘤的临床决策支持。系统集成了多个医学知识库（ESMO、NCCN、WHO、UICC等），通过RAG（Retrieval-Augmented Generation）技术和多Agent协作，模拟真实的多学科会诊流程。

## 🌟 主要特性

- **多Agent协作**：模拟真实MDT场景，包括外科医生、内科肿瘤医生、放射肿瘤医生、病理医生、放射科医生、遗传学家等角色
- **知识库集成**：整合ESMO、NCCN、WHO、UICC等权威医学指南
- **记忆系统**：支持患者历史记录追踪，实现连续性诊疗
- **文献检索**：自动查询PubMed获取最新研究证据
- **基因分析**：集成CIViC数据库进行基因变异分析
- **异步处理**：高效的并发处理能力，支持批量患者处理

## 📁 项目结构

```
hema_agent/
├── agent.py                    # 核心Agent逻辑和MDT工作流
├── run.py                      # 批量处理患者数据
├── main.py                     # 简单入口点
├── config_manager.py           # 统一配置管理器
├── config.toml                 # 配置文件
├── tools.py                    # Agent工具集（web搜索等）
├── logging_setup.py            # 日志配置
├── memory_bank_store.py        # 患者记忆存储
├── prompts.py                  # Prompt模板库
│
├── rag_guideline.py            # ESMO/NCCN/HEMA指南RAG工具
├── rag_pathology.py            # 病理学RAG工具
├── rag_staging_uicc.py         # UICC分期RAG工具
├── rag_who.py                  # WHO分类RAG工具
├── rag_common.py               # RAG公共函数
│
├── gene_search.py              # CIViC基因搜索工具
├── pubmedv4.py                 # PubMed文献检索工具
│
├── data_processing/            # 数据处理工具（离线使用）
│   ├── README.md               # 数据处理工具说明
│   ├── GROBID.py              # PDF转XML工具
│   ├── degrobid.py            # GROBID后处理
│   ├── nccn_post.py           # NCCN文档后处理
│   └── PPOCESS_GUIDELINE.py   # 指南文档整理
│
├── requirements.txt            # Python依赖
├── .gitignore                 # Git忽略规则
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 环境准备

**Python版本要求**：Python 3.10+

```bash
# 克隆项目
git clone <repository-url>
cd hema_agent

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

复制环境变量模板并填写API密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写必要的API密钥：

```bash
# 必需
OPENAI_API_KEY=your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here

# 可选（使用本地模型时）
LOCAL_API_BASE=http://localhost:8000/v1
```

编辑 `config.toml` 选择使用的模型：

```toml
# 使用OpenAI GPT-4o
[openai_gpt4o]
model_name = "gpt-4o"
max_tokens = 4096
env_prefix = "OPENAI"

# 或使用本地模型
[local_llama]
model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8"
env_prefix = "LOCAL"
```

### 3. 准备数据（首次使用）

在运行Agent之前，需要先准备医学知识库。请参考 [`data_processing/README.md`](data_processing/README.md) 了解详细步骤。

**简要流程：**
1. 准备PDF格式的医学指南文档
2. 使用GROBID转换为XML
3. 运行构建脚本创建向量数据库
4. 配置 `config.toml` 中的路径

### 4. 运行单个患者评估

```bash
python agent.py patient_12345_20250114.json
```

**输入文件格式示例（JSON或TXT）：**

```json
{
  "patient_id": "12345",
  "age": 65,
  "gender": "Male",
  "diagnosis": "Acute Myeloid Leukemia (AML)",
  "history": "...",
  "current_symptoms": "...",
  "lab_results": "...",
  "imaging": "...",
  "molecular_testing": "FLT3-ITD positive"
}
```

或纯文本格式：
```
患者ID: 12345
年龄: 65岁
性别: 男
诊断: 急性髓系白血病（AML）
...
```

### 5. 批量处理

```bash
python run.py /path/to/patient/files/
```

批量处理会：
- 自动扫描目录中的所有 `.txt` 和 `.json` 文件
- 按患者ID分组，按时间排序处理
- 跳过已处理的文件（基于日志）
- 将输出保存到 `logs/` 目录

## 📋 使用示例

### 单个患者评估

```bash
# 处理单个患者文件
python agent.py cases/patient_001_20250114.txt
```

**输出示例：**
```
📄 Processing file: patient_001_20250114.txt
✅ Initial Assessment Complete for Patient 001

Final Recommendation:
基于患者的临床特征（65岁男性，AML，FLT3-ITD阳性），
多学科团队建议：

1. 诱导化疗：7+3方案（柔红霉素+阿糖胞苷）
2. 靶向治疗：加用吉瑞替尼（Gilteritinib）
3. 巩固治疗：视缓解情况考虑异基因造血干细胞移植
4. 支持治疗：...
5. 随访计划：...

工具调用统计报告:
✅ rag_guideline: 总计 15 | 成功 15 | 失败 0
✅ pubmed_search: 总计 3 | 成功 3 | 失败 0
✅ gene_search: 总计 2 | 成功 2 | 失败 0
```

### 批量处理

```bash
# 批量处理目录中的所有患者
python run.py data/patients/

# 输出
📄 [001 @ 20250101] Processing file: patient_001_20250101.txt
✅ Done patient_001_20250101.txt in 45.23s

⏭️ Skipping patient_001_20250110.txt, already done.

📄 [002 @ 20250105] Processing file: patient_002_20250105.txt
✅ Done patient_002_20250105.txt in 38.67s

🏁 All files processed in 125.45 seconds
```

## 🔧 配置说明

### config.toml

主要配置项：

```toml
# 模型配置
[openai_gpt4o]
model_name = "gpt-4o"
max_tokens = 4096
reasoning_effort = "minimal"
reasoning_verbosity = "low"

# Embedding模型
[embedding]
model_name = "Qwen3-Embedding-8B"
env_prefix = "LOCAL"

# 路径配置
[paths]
esmo_db_storage = "ESMO_chroma_db_qwen"
nccn_db_storage = "NCCN_chroma_db_qwen"
hema_db_storage = "HEMA_chroma_db_qwen"
# ... 更多路径配置
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `AGENT_MAX_CONCURRENCY` | Agent并发数 | 5 |
| `CONTEXT_MAX_CHARS` | 最大上下文字符数 | 12000 |
| `MEMORY_MAX_ENTRIES` | 最大记忆条目数 | 3 |
| `LOG_LEVEL` | 日志级别 | INFO |
| `CHROMA_TELEMETRY` | ChromaDB遥测 | 0 (关闭) |

## 🏗️ 系统架构

### MDT工作流程

```
                    ┌─────────────────┐
                    │   患者病历输入    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   记忆银行加载   │
                    │ (历史会诊记录)   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
    │ 放射科预检   │  │ 病理科预检 │  │ 分期评估    │
    │ (影像评估)   │  │ (病理诊断) │  │ (UICC/TNM)  │
    └───────┬──────┘  └─────┬─────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │   多学科会诊     │
                    │  (并发执行)      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │         │          │          │         │
   ┌────▼───┐ ┌──▼──┐ ┌────▼────┐ ┌──▼──┐ ┌────▼────┐
   │ 外科医生│ │内科  │ │放疗科  │ │全科 │ │遗传学家 │
   └────┬───┘ └──┬──┘ └────┬────┘ └──┬──┘ └────┬────┘
        │         │          │          │         │
        └─────────┼──────────┼──────────┼─────────┘
                  │          │          │
             ┌────▼──────────▼──────────▼────┐
             │     PubMed文献检索             │
             │  (基于各科建议生成查询)        │
             └────────────┬──────────────────┘
                          │
                  ┌───────▼────────┐
                  │  补充肿瘤科医生 │
                  │ (整合最新证据)  │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │   主席汇总      │
                  │ (协调意见冲突)  │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │  最终治疗方案   │
                  │  (保存到记忆)   │
                  └────────────────┘
```

### Agent角色

1. **放射科医生（Radiologist）**：评估影像资料，确定分期
2. **病理科医生（Pathologist）**：解读病理报告，建议额外检查
3. **外科医生（Surgeon）**：评估手术可行性和方案
4. **内科肿瘤医生（Medical Oncologist）**：制定化疗和靶向治疗方案
5. **放射肿瘤医生（Radiation Oncologist）**：评估放疗指征
6. **遗传学家（Geneticist）**：分析基因变异，推荐靶向药物
7. **全科医生（GP）**：关注全身状况和支持治疗
8. **补充肿瘤医生（Additional Oncologist）**：整合PubMed文献
9. **主席（Chairman）**：协调各科意见，形成统一方案

## 📊 输出格式

系统输出包括：

1. **各科建议**：每位专科医生的独立评估
2. **文献支持**：相关的PubMed文章摘要
3. **统一方案**：主席汇总的最终治疗建议
4. **工具统计**：RAG检索、文献查询等工具的使用情况
5. **记忆更新**：保存到患者记忆银行供后续会诊参考

输出文件保存位置：
- 控制台输出：实时显示
- 日志文件：`logs/<patient_id>_<timestamp>.log`
- 记忆文件：`memory_bank/memory_bank_<patient_id>.json`

## 🧪 测试

```bash
# 运行测试（如果已配置）
pytest tests/

# 运行特定测试
pytest tests/test_rag_common.py -v
```

## 📝 日志

日志文件位置：
- **主日志**：`logs/hema_agent.log`
- **工具错误日志**：`tool_errors.log`
- **批量处理日志**：`logs/<patient_file>.log`

日志级别可通过环境变量 `LOG_LEVEL` 设置（DEBUG/INFO/WARNING/ERROR）。

## 🔐 安全注意事项

1. **API密钥保护**：
   - 不要提交 `.env` 文件到版本控制
   - 使用环境变量存储敏感信息
   - 定期轮换API密钥

2. **患者数据**：
   - 遵守HIPAA/GDPR等隐私法规
   - 不要在日志中记录敏感患者信息
   - 加密存储患者数据

3. **模型输出**：
   - AI输出仅供参考，不能替代医生判断
   - 所有临床决策需由持证医生审核
   - 定期审查和验证系统输出

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

[在此添加许可证信息]

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- Issue Tracker: [GitHub Issues]
- Email: [your-email@example.com]

## 🙏 致谢

- ESMO、NCCN、WHO、UICC等组织提供的医学指南
- OpenAI提供的语言模型
- CIViC数据库提供的基因变异数据
- PubMed/NCBI提供的文献数据

---

**免责声明**：本系统仅用于临床决策支持，不能替代专业医疗建议、诊断或治疗。所有医疗决策应由合格的医疗专业人员根据患者的具体情况做出。
