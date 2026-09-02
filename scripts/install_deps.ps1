$ErrorActionPreference = "Continue"
$py = "d:\AIA Case\RAGagent\.venv\Scripts\python.exe"
$log = "d:\AIA Case\RAGagent\logs\pip_install.log"
$mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
"START $(Get-Date -Format o)" | Out-File $log -Encoding utf8

& $py -m pip install -i $mirror --trusted-host pypi.tuna.tsinghua.edu.cn `
  "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" "pydantic>=2.9.0" "pydantic-settings>=2.6.0" `
  "python-dotenv>=1.0.1" "orjson>=3.10.0" "huggingface-hub>=0.26.0" "numpy>=1.26.0" `
  "scikit-learn>=1.5.0" "rank-bm25>=0.2.2" "pypdf>=5.0.0" "httpx>=0.27.0" `
  "sentence-transformers>=3.3.0" *>> $log
"CORE_EXIT=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8

& $py -m pip install -i $mirror --trusted-host pypi.tuna.tsinghua.edu.cn llama-cpp-python *>> $log
if ($LASTEXITCODE -ne 0) {
  "fallback abetlen wheel" | Out-File $log -Append -Encoding utf8
  & $py -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu *>> $log
}
"LLAMA_EXIT=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8
"ALL_DONE $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
