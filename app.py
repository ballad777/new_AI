# app.py
# -*- coding: utf-8 -*-
"""
ChatGPT-like BI Assistant (API Key login)
- Sidebar: upload Excel, show tables, clear chat
- Main: chat UI (st.chat_message + st.chat_input)
- Multi-file, multi-sheet ingestion
- Chinese-first semantic understanding
- Robustness: JSON extraction + retry + auto-fix code + fallback outputs
- Sandboxed execution with whitelist import (pandas/numpy/plotly)
"""

from __future__ import annotations

import io
import os
import re
import json
import sys
import time
import pickle
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="AI 資料分析助理", layout="wide")

APP_TITLE = "AI 資料分析助理"
DEFAULT_MODEL = "gpt-4.1-mini"     # 你日後想換更強模型再改這裡
TOPK_TABLES = 6                    # 固定：不給你調，系統自動優化
SANDBOX_TIMEOUT = 18               # 固定：不給你調
HEAD_ROWS = 12
HEAD_COLS = 40


# -----------------------------
# API Key Login (per-session)
# -----------------------------
def require_api_key() -> str:
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    st.title(APP_TITLE)
    st.caption("請輸入你的 OpenAI API Key。Key 只存在此瀏覽器 Session，關閉頁面就消失。")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx")
    if st.button("✅ 開始使用", use_container_width=True):
        if not api_key or not api_key.startswith("sk-"):
            st.error("API Key 格式不正確（通常以 sk- 開頭）。")
            st.stop()
        try:
            client = OpenAI(api_key=api_key)
            _ = client.models.list()
        except RateLimitError:
            st.error("API Key 可用，但目前額度不足/未開通 Billing（429）。請先儲值後再試。")
            st.stop()
        except Exception:
            st.error("API Key 驗證失敗：請確認 Key 有效、已啟用 Billing 且有可用額度。")
            st.stop()

        st.session_state.openai_api_key = api_key
        st.rerun()

    if not st.session_state.openai_api_key:
        st.stop()

    return st.session_state.openai_api_key


# -----------------------------
# Data ingestion
# -----------------------------
@dataclass
class TableProfile:
    key: str
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    sample_head: List[Dict[str, Any]]


def read_excel_all_sheets(uploaded_file) -> Dict[str, pd.DataFrame]:
    data = uploaded_file.read()
    bio = io.BytesIO(data)
    xls = pd.ExcelFile(bio)
    out: Dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(bio, sheet_name=sheet)
        bio.seek(0)
        out[f"{uploaded_file.name} | {sheet}"] = df
    return out


def try_parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        name = str(c)
        if any(k in name for k in ["日期", "時間", "date", "time", "年月", "月份"]):
            try:
                df2[c] = pd.to_datetime(df2[c], errors="coerce")
            except Exception:
                pass
    return df2


def build_profile(key: str, df: pd.DataFrame) -> TableProfile:
    head = df.head(HEAD_ROWS)
    if head.shape[1] > HEAD_COLS:
        head = head.iloc[:, :HEAD_COLS]
    return TableProfile(
        key=key,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns.tolist()],
        dtypes={str(c): str(df[c].dtype) for c in df.columns},
        sample_head=head.fillna("").astype(str).to_dict(orient="records"),
    )


# -----------------------------
# Retrieval (Chinese-friendly, no deps)
# -----------------------------
def normalize(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def ngrams(s: str, n: int) -> List[str]:
    s = normalize(s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    if len(s) <= n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s) - n + 1)]


def score_table(question: str, p: TableProfile) -> float:
    qg = set(ngrams(question, 2) + ngrams(question, 3))
    if not qg:
        return 0.0
    meta = " ".join([p.key] + p.columns + list(p.dtypes.keys()))
    mg = set(ngrams(meta, 2) + ngrams(meta, 3))
    if not mg:
        return 0.0

    inter = len(qg & mg)
    union = len(qg | mg)
    jacc = inter / union if union else 0.0

    boost = 0.0
    q = question
    kl = p.key.lower()
    if any(k in q for k in ["採購", "進貨", "供應商"]) and any(k in kl for k in ["purchase", "採購", "進貨"]):
        boost += 0.10
    if any(k in q for k in ["銷售", "銷貨", "營收"]) and any(k in kl for k in ["sales", "銷售", "銷貨"]):
        boost += 0.10
    return float(jacc + boost)


def pick_tables(question: str, profiles: Dict[str, TableProfile], topk: int) -> List[str]:
    scored = [(score_table(question, p), k) for k, p in profiles.items()]
    scored.sort(key=lambda x: x[0], reverse=True)
    keys = [k for s, k in scored[:topk] if s > 0]
    if not keys and scored:
        keys = [scored[0][1]]
    return keys


def tables_context_json(keys: List[str], profiles: Dict[str, TableProfile]) -> str:
    blocks = []
    for k in keys:
        p = profiles[k]
        blocks.append({
            "table_key": p.key,
            "rows": p.rows,
            "cols": p.cols,
            "columns": p.columns,
            "dtypes": p.dtypes,
            "sample_head": p.sample_head,
        })
    return json.dumps(blocks, ensure_ascii=False, indent=2)


# -----------------------------
# LLM prompting (robust JSON output)
# -----------------------------
SYSTEM_PROMPT = """你是一個企業級資料分析助理，擅長用 pandas/numpy/plotly 做資料分析與圖表。
使用者用繁體中文口語提問，你必須用語意理解決定要用哪些表、哪些欄位、怎麼分群、怎麼計算。

你會取得 dfs: Dict[str, pandas.DataFrame]，key 為 table_key（例如 "sales_2023_2025.xlsx | 2023總表"）。

你必須產生「可執行 Python 程式碼」來完成分析，並且程式碼一定要設定：
- final_answer: str（繁體中文結論）
- result_tables: Dict[str, pandas.DataFrame]（至少放 1 張表）
- result_plotly_json: Optional[str]（若有圖，用 fig.to_json()；沒有則 None）

規則：
1) 日期欄位請用 pd.to_datetime(errors="coerce")；每月彙總要產生「YYYY-MM」字串欄位（例如欄名叫 '年月'）。
2) 對子集賦值前先 .copy()，並用 .loc。
3) 同時畫「數量」與「金額」時必須雙 y 軸（數量左軸、金額右軸），避免數量看起來像 0。
4) 欄位名盡量保留中文，不要無故改成 Month 這種英文。
限制：只能 import pandas, numpy, plotly（graph_objects/express）。禁止網路、檔案 IO、系統指令、其它第三方套件。

輸出格式（非常重要）：
請只輸出 JSON（不要 markdown），格式：
{
  "python_code": "<你的Python程式碼字串>"
}
"""


def extract_json_object(text: str) -> Optional[dict]:
    """Try to parse JSON from model output even if extra text exists."""
    text = (text or "").strip()
    if not text:
        return None
    # direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # find first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    chunk = m.group(0)
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def extract_text(resp) -> str:
    parts = []
    for o in getattr(resp, "output", []) or []:
        if getattr(o, "type", None) == "message":
            for c in getattr(o, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", "") or "")
    return "\n".join(parts).strip()


def llm_get_code_json(client: OpenAI, model: str, question: str, tables_json: str, extra_feedback: str = "") -> str:
    user_prompt = f"""使用者問題：
{question}

可用資料表資訊（JSON）：
{tables_json}

{extra_feedback}
請依照系統要求輸出 JSON，並在 python_code 內提供完整可執行程式碼。
"""
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return extract_text(resp)


def get_python_code_with_retries(client: OpenAI, model: str, question: str, tables_json: str) -> Tuple[str, List[str]]:
    """Return python_code and debug logs."""
    logs: List[str] = []
    feedback = ""
    for attempt in range(1, 4):
        raw = llm_get_code_json(client, model, question, tables_json, extra_feedback=feedback)
        obj = extract_json_object(raw)
        if obj and isinstance(obj.get("python_code"), str) and obj["python_code"].strip():
            code = obj["python_code"].strip()
            # quick sanity: required variables
            if "final_answer" in code and "result_tables" in code and "result_plotly_json" in code:
                logs.append(f"Attempt {attempt}: OK")
                return code, logs
            else:
                logs.append(f"Attempt {attempt}: Missing required vars, retry")
                feedback = "⚠️ 你上一輪的 python_code 沒有正確設定 final_answer / result_tables / result_plotly_json，請修正並重新輸出 JSON。"
                continue

        logs.append(f"Attempt {attempt}: JSON parse failed / empty, retry")
        feedback = "⚠️ 你上一輪沒有輸出正確 JSON（或 python_code 為空）。請只輸出 JSON，且 python_code 必須是完整程式碼。"

    # last resort: return empty code (caller will fallback)
    return "", logs


# -----------------------------
# Sandbox execution (subprocess)
# - NO template injection -> no SyntaxError
# - Whitelist import roots: pandas/numpy/plotly
# - Convert datetime columns to ISO strings in returned tables
# -----------------------------
RUNNER = r"""
import json, pickle, traceback, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in {"pandas", "numpy", "plotly"}:
        raise ImportError(f"Import of '{name}' is not allowed")
    return __import__(name, globals, locals, fromlist, level)

SAFE_BUILTINS = {
    "__import__": safe_import,
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "float": float, "int": int, "str": str, "bool": bool, "print": print,
}

with open("dfs.pkl", "rb") as f:
    dfs = pickle.load(f)

with open("user_code.py", "r", encoding="utf-8") as f:
    code = f.read()

local_env = {
    "pd": pd,
    "np": np,
    "dfs": dfs,
    "final_answer": "",
    "result_tables": {},
    "result_plotly_json": None,
}

result = {"ok": True, "final_answer": "", "tables": {}, "plotly_json": None, "stderr": ""}

def df_to_records_json(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_json(orient="records", force_ascii=False)

try:
    exec(compile(code, "<user_code>", "exec"), {"__builtins__": SAFE_BUILTINS}, local_env)

    final_answer = str(local_env.get("final_answer", "") or "")
    result_tables = local_env.get("result_tables", {}) or {}
    plotly_json = local_env.get("result_plotly_json", None)

    tables_out = {}
    for name, df in result_tables.items():
        if isinstance(df, pd.DataFrame):
            df2 = df.copy()
            if len(df2) > 2000:
                df2 = df2.head(2000)
            tables_out[str(name)] = df_to_records_json(df2)

    result["final_answer"] = final_answer
    result["tables"] = tables_out
    result["plotly_json"] = plotly_json if isinstance(plotly_json, str) else None

except Exception:
    result["ok"] = False
    result["stderr"] = traceback.format_exc()

with open("out.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
"""


def run_sandbox(user_code: str, dfs: Dict[str, pd.DataFrame], timeout_sec: int) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "dfs.pkl"), "wb") as f:
            pickle.dump(dfs, f)
        with open(os.path.join(td, "user_code.py"), "w", encoding="utf-8") as f:
            f.write(user_code)
        with open(os.path.join(td, "runner.py"), "w", encoding="utf-8") as f:
            f.write(RUNNER)

        try:
            proc = subprocess.run(
                [sys.executable, "runner.py"],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stderr": f"Execution timeout after {timeout_sec}s.", "final_answer": "", "tables": {}, "plotly_json": None}

        out_path = os.path.join(td, "out.json")
        if not os.path.exists(out_path):
            return {"ok": False, "stderr": "Sandbox did not produce out.json.", "final_answer": "", "tables": {}, "plotly_json": None}

        with open(out_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        res["_runner_stderr"] = proc.stderr
        return res


# -----------------------------
# Fallback (never empty)
# -----------------------------
def fallback_answer(dfs: Dict[str, pd.DataFrame], selected_keys: List[str]) -> Tuple[str, Dict[str, pd.DataFrame], Optional[str]]:
    lines = ["模型本次沒有產出可用分析程式碼，我先給你保底資訊（確保永遠不會空白）：", ""]
    tables: Dict[str, pd.DataFrame] = {}
    for k in selected_keys[:3]:
        df = dfs[k]
        lines.append(f"- 使用表：{k}（{df.shape[0]} rows × {df.shape[1]} cols）")
        lines.append(f"  欄位：{', '.join([str(c) for c in df.columns[:30]])}{'...' if df.shape[1] > 30 else ''}")
        tables[f"HEAD｜{k}"] = df.head(20)
    return "\n".join(lines), tables, None


# -----------------------------
# App start
# -----------------------------
api_key = require_api_key()
client = OpenAI(api_key=api_key)

# Session init
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "profiles" not in st.session_state:
    st.session_state.profiles = {}
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant", "content":..., "tables":..., "plotly_json":...}]

# -----------------------------
# Sidebar (GPT-like)
# -----------------------------
with st.sidebar:
    st.header("📁 資料")
    uploads = st.file_uploader("上傳 Excel（可多選）", type=["xlsx"], accept_multiple_files=True)

    if uploads:
        dfs: Dict[str, pd.DataFrame] = {}
        profiles: Dict[str, TableProfile] = {}
        for uf in uploads:
            try:
                temp = read_excel_all_sheets(uf)
            except Exception as e:
                st.error(f"讀取 {uf.name} 失敗：{e}")
                continue
            for k, df in temp.items():
                df2 = try_parse_datetime(df)
                dfs[k] = df2
                profiles[k] = build_profile(k, df2)

        st.session_state.dfs = dfs
        st.session_state.profiles = profiles

    if st.session_state.dfs:
        st.success(f"已載入 {len(st.session_state.dfs)} 張表")
        with st.expander("查看表清單", expanded=False):
            for k, p in st.session_state.profiles.items():
                st.write(f"- {k}（{p.rows}×{p.cols}）")
    else:
        st.info("先上傳 Excel 才能分析")

    st.divider()
    if st.button("🧹 清除對話", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("提示：此介面像 GPT：左邊管資料，右邊直接聊天問分析。")


# -----------------------------
# Main: Chat UI
# -----------------------------
st.title("💬 直接問（中文語意理解 + 自動找表 + 分析圖表）")

if not st.session_state.dfs:
    st.stop()

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # tables
        if m.get("tables"):
            for name, df in m["tables"].items():
                st.write(f"**{name}**")
                st.dataframe(df, use_container_width=True)
        # plotly
        if m.get("plotly_json"):
            try:
                import plotly.io as pio
                fig = pio.from_json(m["plotly_json"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("圖表解析失敗（JSON 仍在）")

prompt = st.chat_input("例如：分析 2023 銷貨總量與總未稅金額，做每月趨勢（雙軸），列產品TOP10與業務TOP10")

if prompt:
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant processing
    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            profiles = st.session_state.profiles
            dfs_all = st.session_state.dfs

            selected_keys = pick_tables(prompt, profiles, topk=TOPK_TABLES)
            tables_json = tables_context_json(selected_keys, profiles)
            dfs_subset = {k: dfs_all[k] for k in selected_keys if k in dfs_all}

            # 1) Get code with retries
            code, logs = get_python_code_with_retries(client, DEFAULT_MODEL, prompt, tables_json)

            # 2) If still empty => fallback
            if not code.strip():
                final_answer, result_tables, plotly_json = fallback_answer(dfs_all, selected_keys)
                st.markdown(final_answer)
                for name, df in result_tables.items():
                    st.write(f"**{name}**")
                    st.dataframe(df, use_container_width=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "tables": result_tables,
                    "plotly_json": plotly_json,
                })
                st.stop()

            # 3) Run sandbox
            res = run_sandbox(code, dfs_subset, timeout_sec=SANDBOX_TIMEOUT)

            # 4) Auto-fix if sandbox fails OR empty outputs
            if (not res.get("ok", False)) or (not (res.get("final_answer") or "").strip() and (res.get("tables") or {})):
                err = res.get("stderr", "")
                feedback = f"""
⚠️ 你上一輪的程式碼執行失敗或輸出為空。
錯誤資訊/狀況如下：
{err[:1500]}

請你修正程式碼，確保一定設定 final_answer（非空字串）與 result_tables（至少 1 張 DataFrame），必要時可不畫圖（result_plotly_json=None）。
仍請只輸出 JSON：{{"python_code": "..."}}
"""
                # one repair attempt
                raw2 = llm_get_code_json(client, DEFAULT_MODEL, prompt, tables_json, extra_feedback=feedback)
                obj2 = extract_json_object(raw2) or {}
                code2 = (obj2.get("python_code") or "").strip()

                if code2:
                    res2 = run_sandbox(code2, dfs_subset, timeout_sec=SANDBOX_TIMEOUT)
                    if res2.get("ok", False) and ((res2.get("final_answer") or "").strip() or (res2.get("tables") or {})):
                        res = res2

            # 5) Final render (never empty)
            final_answer = (res.get("final_answer") or "").strip()
            tables_json_map = res.get("tables") or {}
            plotly_json = res.get("plotly_json")

            result_tables: Dict[str, pd.DataFrame] = {}

            # decode tables
            for name, df_json in tables_json_map.items():
                try:
                    df_out = pd.read_json(io.StringIO(df_json), orient="records", dtype=False)
                    # if Month exists, rename to 年月
                    if "Month" in df_out.columns and "年月" not in df_out.columns:
                        df_out = df_out.rename(columns={"Month": "年月"})
                    # if "年月" looks like timestamp digits, keep as string anyway
                    result_tables[str(name)] = df_out
                except Exception:
                    pass

            if not final_answer and not result_tables:
                final_answer, result_tables, plotly_json = fallback_answer(dfs_all, selected_keys)

            # show selected tables hint (small, like GPT tool context)
            st.caption("已自動使用資料表：\n" + "\n".join([f"- {k}" for k in selected_keys]))

            st.markdown(final_answer if final_answer else "（本次沒有文字結論，但我已輸出表格。）")

            for name, df in result_tables.items():
                st.write(f"**{name}**")
                st.dataframe(df, use_container_width=True)

            if plotly_json:
                try:
                    import plotly.io as pio
                    fig = pio.from_json(plotly_json)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("圖表解析失敗（JSON 仍在）")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "tables": result_tables,
                "plotly_json": plotly_json,
            })
