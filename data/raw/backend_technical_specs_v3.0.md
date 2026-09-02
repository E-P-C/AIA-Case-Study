AKP Backend Technical Specifications v3.0
Document Status: Final
Date: 2025‑06‑18

# Section 7 RAG‑QA Service Technical Constraints
7.1 Non‑Functional Baseline Requirements
1. End‑to‑end latency: 90% percent of user queries must complete within 10 seconds end‑to‑end (HTTP request received to full response returned to client).
2. Supported corpus input formats: native PDF, DOCX, plain text, scanned‑image‑based PDF (requires dedicated OCR pre‑processing pipeline).
3. Corpus language: mixed bilingual Chinese‑English content. System shall handle code‑switching within single document chunks.
4. Retrieval baseline configuration: default top_k = 4 candidate chunks; reranker module toggle‑able ON / OFF; generation temperature configurable within range 0.1 ~ 0.9.
5. Maximum input context window for LLM calls shall not exceed model context‑window limit; implement truncation strategy when retrieved context overflows.

90%请求端到端响应时间 ≤10秒；语料支持原生PDF、DOCX、文本、扫描PDF；支持中英混合文档；top_k默认4；重排器可开关；temperature取值0.1‑0.9；检索上下文溢出时执行截断策略。

7.2 Error Handling Specification
‑ Low‑similarity retrieval condition: similarity score below threshold = 0.45. System returns explicit refusal message, must not hallucinate plausible‑sounding answers.
‑ Zero retrieved chunks: return capability boundary notice to end‑user.
‑ Multi‑turn dialogue: conversation context state maintained per session‑ID. Users do not need to repeat historical question context for follow‑up questions. Session idle timeout set to 30 minutes. After timeout session context is discarded.
‑ API error codes: HTTP 400 for bad user input; HTTP 429 for rate‑limiting; HTTP 500 for internal service failure.

相似度低于0.45阈值拒绝回答；无检索结果告知能力边界；多轮会话依靠session‑id保存上下文，空闲30分钟销毁会话；定义API错误码：400错误输入，429限流，500内部异常。

7.3 Observability & Logging Mandatory Fields
Every single user request must log:
‑ session_id, timestamp, raw user query string
‑ list of retrieved chunks: source filename, file_type(native‑pdf / scanned‑pdf), page number, chunk snippet, similarity score
‑ final generated answer text
‑ token_in (prompt tokens), token_out (completion tokens)
‑ end‑to‑end latency in milliseconds
‑ runtime parameters: top_k value, reranker enabled flag, temperature setting
‑ detection flags: prompt‑injection‑detected flag, pii_detected flag

所有请求日志字段包含会话ID、时间戳、用户query、检索片段元数据、输入输出token、延迟毫秒、运行参数、提示注入标记、PII检测标记。

7.4 Cost Estimation & Sensitivity Analysis Requirement
Deliver token‑cost estimation per 1000 API invocation.
Sensitivity analysis must cover three variable dimensions:
1) top_k variations (e.g. 2,4,6)
2) reranker mode: ON versus OFF
3) temperature values (e.g. 0.1,0.4,0.7)

Report how token consumption, latency and expected answer quality shift across different configurations.

需要输出每1000次API调用token成本估计。敏感性分析覆盖top_k(2,4,6)、reranker开关、temperature(0.1,0.4,0.7)，分析不同配置下token消耗、延迟、回答质量变化。

7.5 Evolvability Requirements
RAG components shall be loosely coupled: embedding model, chunking logic, reranker module, LLM generator should be swap‑able without large‑scale code refactoring.
New knowledge documents shall be ingested without full vector‑database rebuild.

系统组件解耦：Embedding、分块、重排器、生成模型可替换；新增知识库文档不需要重建全部向量库。