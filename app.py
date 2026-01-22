# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
from openai import OpenAI
from openai import RateLimitError


# =========================
# Config
# =========================
st.set_page_config(page_title="AI 資料分析助理", layout="wide")

APP_TITLE = "AI 資料分析助理（GPT 風格｜上下文記憶版）"
DEFAULT_MODEL = "gpt-4.1-mini"
TOPK_TABLES = 8
HEAD_ROWS = 12
HEAD_COLS = 50
CONTEXT_TURNS = 8  # planner uses last N turns
TOPN_DEFAULT = 10


# =========================
# Custom CSS - Pure White Theme with Dark/Light Mode Support
# =========================
CUSTOM_CSS = """
<style>
    /* Light Mode (Default) */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #f1f3f4;
        --text-primary: #1a1a1a;
        --text-secondary: #4a4a4a;
        --text-muted: #6b7280;
        --border-color: #e5e7eb;
        --accent-color: #2563eb;
        --accent-hover: #1d4ed8;
    }
    
    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #242424;
            --bg-tertiary: #2d2d2d;
            --text-primary: #f5f5f5;
            --text-secondary: #d4d4d4;
            --text-muted: #9ca3af;
            --border-color: #404040;
            --accent-color: #3b82f6;
            --accent-hover: #60a5fa;
        }
    }
    
    .stApp {
        background-color: var(--bg-primary) !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: var(--bg-primary);
    }
    
    section[data-testid="stSidebar"] {
        background-color: var(--bg-primary) !important;
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: var(--bg-primary) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: var(--text-primary) !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: var(--text-primary);
    }
    
    .stMarkdown, .stMarkdown p {
        color: var(--text-primary) !important;
    }
    
    .stChatMessage {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
    }
    
    .stButton > button:hover {
        background-color: var(--bg-tertiary) !important;
        border-color: var(--accent-color) !important;
    }
    
    .stTextInput input {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    [data-testid="stFileUploader"] {
        background-color: var(--bg-secondary) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 12px;
    }
    
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
    }
    
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
    }
    
    hr {
        border-color: var(--border-color) !important;
    }
    
    .js-plotly-plot {
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def normalize(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def safe_json_extract(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def detect_compare_intent(q: str) -> bool:
    qn = normalize(q)
    keywords = ["比較", "對比", "vs", "v.s", "yoy", "年增", "年增率", "年成長", "同期"]
    if any(k in qn for k in keywords):
        return True
    years = re.findall(r"(20\d{2})", qn)
    return len(set(years)) >= 2


def detect_viz_followup_intent(q: str) -> Optional[str]:
    """
    Follow-up intent like:
    - 改成圖表 / 畫圖 / 做成圖
    - 換成折線 / 換成長條 / 柱狀
    - 只畫成長率
    """
    qn = normalize(q)

    # explicit chart change
    if any(k in qn for k in ["改成圖表", "畫成圖", "改成圖", "做成圖", "圖表", "plot", "chart"]):
        if any(k in qn for k in ["折線", "line"]):
            return "line"
        if any(k in qn for k in ["長條", "柱狀", "bar"]):
            return "bar"
        if any(k in qn for k in ["只畫成長率", "只要成長率", "成長率線", "yoy線"]):
            return "yoy_only"
        return "auto"

    # implicit chart hints
    if any(k in qn for k in ["折線", "line"]):
        return "line"
    if any(k in qn for k in ["長條", "柱狀", "bar"]):
        return "bar"
    if any(k in qn for k in ["只畫成長率", "只要成長率"]):
        return "yoy_only"

    return None


def pretty_md(sections: Dict[str, Any]) -> str:
    title = sections.get("title") or "分析結果"
    bullets = sections.get("bullets") or []
    obs = sections.get("observations") or []
    sug = sections.get("suggestions") or []
    notes = sections.get("notes") or []

    lines = [f"## {title}\n"]
    if bullets:
        lines.append("### 摘要")
        for b in bullets:
            lines.append(f"- **{b}**")
        lines.append("")
    if obs:
        lines.append("### 觀察")
        for i, o in enumerate(obs, 1):
            lines.append(f"{i}. {o}")
        lines.append("")
    if sug:
        lines.append("### 建議")
        for s in sug:
            lines.append(f"- {s}")
        lines.append("")
    if notes:
        lines.append("### 備註")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines).strip()


def df_safe_preview(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    out = df.head(n).copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


# =========================
# Data ingestion
# =========================
@dataclass
class TableProfile:
    key: str
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    sample_head: List[Dict[str, Any]]


def head_profile(df: pd.DataFrame) -> List[Dict[str, Any]]:
    head = df.head(HEAD_ROWS)
    if head.shape[1] > HEAD_COLS:
        head = head.iloc[:, :HEAD_COLS]
    return head.fillna("").astype(str).to_dict(orient="records")


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


def light_datetime_parse(df: pd.DataFrame) -> pd.DataFrame:
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
    return TableProfile(
        key=key,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns.tolist()],
        dtypes={str(c): str(df[c].dtype) for c in df.columns},
        sample_head=head_profile(df),
    )


def score_table(question: str, profile: TableProfile) -> float:
    q = normalize(question)
    meta = normalize(profile.key + " " + " ".join(profile.columns))

    def grams(s: str, n: int) -> set:
        s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
        if len(s) <= n:
            return {s} if s else set()
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    qg = grams(q, 2) | grams(q, 3)
    mg = grams(meta, 2) | grams(meta, 3)
    if not qg or not mg:
        return 0.0
    base = len(qg & mg) / len(qg | mg)

    boost = 0.0
    kl = normalize(profile.key)
    if any(k in q for k in ["銷售", "銷貨", "營收"]) and any(k in kl for k in ["sales", "銷"]):
        boost += 0.10
    if any(k in q for k in ["採購", "進貨", "進銷", "供應商"]) and any(k in kl for k in ["purchase", "進"]):
        boost += 0.10
    if detect_compare_intent(q):
        boost += 0.05
    return float(base + boost)


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


# =========================
# API Key Login
# =========================
def require_api_key() -> str:
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    if st.session_state.openai_api_key:
        return st.session_state.openai_api_key

    st.title(APP_TITLE)
    st.caption("輸入你的 OpenAI API Key 才能使用（只存在此瀏覽器 Session）。")

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

    st.stop()


# =========================
# LLM planner with CONTEXT MEMORY
# =========================
SCHEMA_SYSTEM = """你是資料分析規劃器。你只做一件事：從使用者問題、對話上下文、以及資料表欄位中，選出正確的表與欄位，並回傳結構化 JSON。
你不寫 Python 程式碼。

你必須理解繁體中文語意，並且要能接續上下文：
- 如果使用者說「改成圖表 / 換成折線 / 把剛剛那個改成...」，你要知道他指的是上一輪的分析結果。
- 如果上一輪已經選定 table_key/欄位/年份，除非使用者明確改需求，否則沿用。

輸出格式：只輸出 JSON。
JSON schema:
{
  "table_key": "要用的 table_key（若是跟上次同一個分析就沿用）",
  "task_type": "trend_monthly | compare_yoy_monthly | topn | generic_summary",
  "date_col": "日期欄(可為空字串)",
  "year_col": "年欄(可為空字串)",
  "month_col": "月欄(可為空字串)",
  "filters": [{"col":"欄位","op":"==|!=|contains|in","value":"值或list"}],
  "metrics": {"quantity_col": "數量欄(可空)", "amount_col": "金額欄(可空)"},
  "dimensions": {"product_col": "產品欄(可空)", "salesperson_col": "業務員欄(可空)", "vendor_col": "供應商欄(可空)"},
  "years": [2023, 2024],
  "topn": 10,
  "notes": "如果欄位不確定，說明你需要哪個欄位/為什麼"
}

重要規則：
- 只要使用者有比較/對比/VS/年增，task_type 一律用 compare_yoy_monthly
- compare_yoy_monthly：同月份對齊 01~12，比較兩年同月的數量或金額（不要把兩年接在一條時間軸）
"""


def build_chat_context_for_planner(messages: List[dict], last_state: dict) -> str:
    recent = messages[-CONTEXT_TURNS:] if messages else []
    lines = ["【最近對話】"]
    for m in recent:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        content = re.sub(r"\n{3,}", "\n\n", content)
        if len(content) > 600:
            content = content[:600] + "…"
        lines.append(f"- {role}: {content}")

    lines.append("\n【上一輪分析狀態】")
    if last_state:
        keep = {k: last_state.get(k) for k in [
            "table_key", "task_type", "years", "metric_col", "metric_kind",
            "filters", "dim_col", "last_table_name", "last_result_table_name"
        ]}
        lines.append(json.dumps(keep, ensure_ascii=False))
    else:
        lines.append("（無）")
    return "\n".join(lines)


def llm_plan(
    client: OpenAI,
    question: str,
    tables_json: str,
    model: str,
    messages: List[dict],
    last_state: dict
) -> dict:
    ctx = build_chat_context_for_planner(messages, last_state)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SCHEMA_SYSTEM},
            {"role": "user", "content": f"{ctx}\n\n【本次使用者新問題】\n{question}\n\n【可用資料表資訊（JSON）】\n{tables_json}\n"},
        ],
    )

    text = ""
    for o in getattr(resp, "output", []) or []:
        if getattr(o, "type", None) == "message":
            for c in getattr(o, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    text += (getattr(c, "text", "") or "")
    obj = safe_json_extract(text)
    return obj or {}


# =========================
# Deterministic analytics (stable)
# =========================
def apply_filters(df: pd.DataFrame, filters: List[dict]) -> pd.DataFrame:
    out = df.copy()
    for f in filters or []:
        col = f.get("col", "")
        op = f.get("op", "")
        val = f.get("value", None)
        if not col or col not in out.columns:
            continue
        s = out[col]
        try:
            if op == "==":
                out = out[s == val]
            elif op == "!=":
                out = out[s != val]
            elif op == "contains":
                out = out[s.astype(str).str.contains(str(val), na=False)]
            elif op == "in":
                if isinstance(val, list):
                    out = out[s.isin(val)]
        except Exception:
            continue
    return out


def ensure_year_month(df: pd.DataFrame, date_col: str, year_col: str, month_col: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Return (df2, ycol, mcol) where ycol/mcol exist in df2.
    """
    out = df.copy()

    if year_col and year_col in out.columns and month_col and month_col in out.columns:
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
        out[month_col] = pd.to_numeric(out[month_col], errors="coerce")
        return out, year_col, month_col

    if date_col and date_col in out.columns:
        dt = to_datetime_series(out[date_col])
        out["_year_"] = dt.dt.year
        out["_month_"] = dt.dt.month
        return out, "_year_", "_month_"

    # guess a datetime column
    for c in out.columns:
        if any(k in str(c) for k in ["日期", "date", "時間", "time"]):
            dt = to_datetime_series(out[c])
            if dt.notna().sum() > 0:
                out["_year_"] = dt.dt.year
                out["_month_"] = dt.dt.month
                return out, "_year_", "_month_"

    # nothing
    out["_year_"] = np.nan
    out["_month_"] = np.nan
    return out, "_year_", "_month_"


def choose_metric_col(df: pd.DataFrame, question: str, plan: dict) -> Tuple[str, str]:
    """
    return (metric_kind, metric_col)
    metric_kind: "quantity" or "amount"
    """
    q = normalize(question)
    want_amount = any(k in q for k in ["金額", "未稅", "含稅", "營收", "成本", "費用", "amount", "revenue"])
    metrics = plan.get("metrics", {}) or {}

    qcol = (metrics.get("quantity_col") or "").strip()
    acol = (metrics.get("amount_col") or "").strip()

    # explicit provided
    if want_amount and acol and acol in df.columns:
        return "amount", acol
    if (not want_amount) and qcol and qcol in df.columns:
        return "quantity", qcol

    # if only one exists
    if acol and acol in df.columns and not qcol:
        return "amount", acol
    if qcol and qcol in df.columns and not acol:
        return "quantity", qcol

    # guess by column names
    # 1) amount
    if want_amount:
        for c in df.columns:
            cn = str(c)
            if any(k in cn for k in ["金額", "未稅", "含稅", "營收", "amount"]):
                return "amount", c

    # 2) quantity
    for c in df.columns:
        cn = str(c)
        if any(k in cn for k in ["數量", "qty", "quantity", "件數"]):
            return "quantity", c

    # fallback numeric column
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return ("amount" if want_amount else "quantity"), numeric_cols[0]

    # worst fallback: first col
    return ("amount" if want_amount else "quantity"), str(df.columns[0])


def build_yoy_table(df: pd.DataFrame, question: str, plan: dict) -> Tuple[pd.DataFrame, dict]:
    years = plan.get("years") or []
    years = [int(y) for y in years if str(y).isdigit()]
    years = sorted(list(dict.fromkeys(years)))

    date_col = (plan.get("date_col") or "").strip()
    year_col = (plan.get("year_col") or "").strip()
    month_col = (plan.get("month_col") or "").strip()

    d2, ycol, mcol = ensure_year_month(df, date_col, year_col, month_col)

    metric_kind, metric_col = choose_metric_col(d2, question, plan)
    if metric_col not in d2.columns:
        metric_col = d2.columns[0]

    d2 = d2.copy()
    d2[ycol] = pd.to_numeric(d2[ycol], errors="coerce")
    d2[mcol] = pd.to_numeric(d2[mcol], errors="coerce")
    d2[metric_col] = pd.to_numeric(d2[metric_col], errors="coerce").fillna(0)

    # determine years if missing
    if len(years) < 2:
        ys = d2[ycol].dropna()
        if len(ys) > 0:
            common = ys.astype(int).value_counts().index.tolist()
            years = [int(x) for x in common[:2]] if len(common) >= 2 else [2023, 2024]
        else:
            years = [2023, 2024]
    y1, y2 = years[0], years[1]

    g = d2.groupby([ycol, mcol])[metric_col].sum().reset_index()

    base = pd.DataFrame({mcol: list(range(1, 13))})
    y1s = g[g[ycol] == y1][[mcol, metric_col]].rename(columns={metric_col: f"{y1}"})
    y2s = g[g[ycol] == y2][[mcol, metric_col]].rename(columns={metric_col: f"{y2}"})

    out = base.merge(y1s, on=mcol, how="left").merge(y2s, on=mcol, how="left")
    out[f"{y1}"] = out[f"{y1}"].fillna(0)
    out[f"{y2}"] = out[f"{y2}"].fillna(0)

    denom = out[f"{y1}"].replace(0, np.nan)
    out["成長率(%)"] = (out[f"{y2}"] - out[f"{y1}"]) / denom * 100
    out["月份"] = out[mcol].astype(int).apply(lambda x: f"{x:02d}")

    meta = {
        "y1": y1,
        "y2": y2,
        "metric_col": metric_col,
        "metric_kind": metric_kind,
        "month_col": mcol,
        "year_col": ycol,
    }
    return out[["月份", f"{y1}", f"{y2}", "成長率(%)"]], meta


def build_trend_monthly(df: pd.DataFrame, question: str, plan: dict) -> Tuple[pd.DataFrame, dict]:
    date_col = (plan.get("date_col") or "").strip()
    year_col = (plan.get("year_col") or "").strip()
    month_col = (plan.get("month_col") or "").strip()
    d2, ycol, mcol = ensure_year_month(df, date_col, year_col, month_col)

    metric_kind, metric_col = choose_metric_col(d2, question, plan)
    d2 = d2.copy()
    d2[ycol] = pd.to_numeric(d2[ycol], errors="coerce")
    d2[mcol] = pd.to_numeric(d2[mcol], errors="coerce")
    d2[metric_col] = pd.to_numeric(d2[metric_col], errors="coerce").fillna(0)

    g = d2.groupby([ycol, mcol])[metric_col].sum().reset_index()
    g = g.dropna(subset=[ycol, mcol])
    g[ycol] = g[ycol].astype(int)
    g[mcol] = g[mcol].astype(int)
    g["年月"] = g[ycol].astype(str) + "-" + g[mcol].apply(lambda x: f"{x:02d}")
    g = g.sort_values(["年月"]).reset_index(drop=True)

    meta = {
        "metric_col": metric_col,
        "metric_kind": metric_kind,
        "year_col": ycol,
        "month_col": mcol,
    }
    return g[["年月", metric_col]].rename(columns={metric_col: "數值"}), meta


def guess_dimension_col(df: pd.DataFrame, plan: dict) -> str:
    dims = plan.get("dimensions", {}) or {}
    candidates = [
        (dims.get("product_col") or "").strip(),
        (dims.get("salesperson_col") or "").strip(),
        (dims.get("vendor_col") or "").strip(),
    ]
    for c in candidates:
        if c and c in df.columns:
            return c

    # guess by name
    for c in df.columns:
        cn = str(c)
        if any(k in cn for k in ["產品", "品名", "料號", "產品代號"]):
            return c
    for c in df.columns:
        cn = str(c)
        if any(k in cn for k in ["業務", "業務員"]):
            return c
    for c in df.columns:
        cn = str(c)
        if any(k in cn for k in ["供應商", "廠商", "vendor"]):
            return c
    # fallback
    return str(df.columns[0])


def build_topn(df: pd.DataFrame, question: str, plan: dict) -> Tuple[pd.DataFrame, dict]:
    topn = int(plan.get("topn") or TOPN_DEFAULT)
    metric_kind, metric_col = choose_metric_col(df, question, plan)
    dim_col = guess_dimension_col(df, plan)

    d2 = df.copy()
    d2[metric_col] = pd.to_numeric(d2[metric_col], errors="coerce").fillna(0)

    g = d2.groupby(dim_col)[metric_col].sum().reset_index()
    g = g.sort_values(metric_col, ascending=False).head(topn).reset_index(drop=True)
    g = g.rename(columns={dim_col: "項目", metric_col: "數值"})

    meta = {
        "metric_col": metric_col,
        "metric_kind": metric_kind,
        "dim_col": dim_col,
        "topn": topn,
    }
    return g, meta


def build_generic_summary(df: pd.DataFrame, question: str, plan: dict) -> Tuple[pd.DataFrame, dict]:
    """Generic summary when no specific task type is detected."""
    metric_kind, metric_col = choose_metric_col(df, question, plan)
    
    # Try to provide a useful summary
    summary_data = {
        "指標": ["總筆數", "總計", "平均", "最大值", "最小值"],
        "數值": [
            len(df),
            df[metric_col].sum() if metric_col in df.columns and pd.api.types.is_numeric_dtype(df[metric_col]) else "N/A",
            df[metric_col].mean() if metric_col in df.columns and pd.api.types.is_numeric_dtype(df[metric_col]) else "N/A",
            df[metric_col].max() if metric_col in df.columns and pd.api.types.is_numeric_dtype(df[metric_col]) else "N/A",
            df[metric_col].min() if metric_col in df.columns and pd.api.types.is_numeric_dtype(df[metric_col]) else "N/A",
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    meta = {
        "metric_col": metric_col,
        "metric_kind": metric_kind,
    }
    return summary_df, meta


# =========================
# Plot templates (stable)
# =========================
def plot_yoy(yoy_df: pd.DataFrame, meta: dict, chart_type: str = "bar") -> go.Figure:
    y1 = meta["y1"]
    y2 = meta["y2"]
    title = f"{y1} vs {y2} 月度比較（同月份對齊）"

    # Professional color palette
    color_y1 = "#3b82f6"  # Blue
    color_y2 = "#f59e0b"  # Orange
    color_yoy = "#10b981"  # Green

    if chart_type == "line":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yoy_df["月份"], y=yoy_df[str(y1)], 
            mode="lines+markers", name=f"{y1}",
            line=dict(color=color_y1, width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=yoy_df["月份"], y=yoy_df[str(y2)], 
            mode="lines+markers", name=f"{y2}",
            line=dict(color=color_y2, width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=yoy_df["月份"],
            y=yoy_df["成長率(%)"],
            mode="lines+markers",
            name="成長率(%)",
            yaxis="y2",
            line=dict(color=color_yoy, width=2, dash="dot"),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="月份",
            yaxis=dict(title="數值", gridcolor="#e5e7eb"),
            yaxis2=dict(title="成長率(%)", overlaying="y", side="right", gridcolor="#e5e7eb"),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=60, r=60, t=80, b=60),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    if chart_type == "yoy_only":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yoy_df["月份"],
            y=yoy_df["成長率(%)"],
            mode="lines+markers",
            name="成長率(%)",
            line=dict(color=color_yoy, width=3),
            marker=dict(size=10),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.1)"
        ))
        fig.update_layout(
            title=dict(text=title + "｜只顯示成長率", font=dict(size=16)),
            xaxis_title="月份",
            yaxis_title="成長率(%)",
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=60, r=60, t=80, b=60),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#ef4444", zerolinewidth=2),
        )
        return fig

    # default bar + yoy line
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yoy_df["月份"], y=yoy_df[str(y1)], name=f"{y1}",
        marker_color=color_y1, opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=yoy_df["月份"], y=yoy_df[str(y2)], name=f"{y2}",
        marker_color=color_y2, opacity=0.85
    ))
    fig.add_trace(go.Scatter(
        x=yoy_df["月份"],
        y=yoy_df["成長率(%)"],
        name="成長率(%)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=color_yoy, width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="月份",
        yaxis=dict(title="數值", gridcolor="#e5e7eb"),
        yaxis2=dict(title="成長率(%)", overlaying="y", side="right"),
        barmode="group",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_trend(trend_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["年月"], y=trend_df["數值"], 
        mode="lines+markers", name="數值",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)"
    ))
    fig.update_layout(
        title=dict(text="月度趨勢", font=dict(size=16)),
        xaxis_title="年月",
        yaxis_title="數值",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e5e7eb"),
        yaxis=dict(gridcolor="#e5e7eb"),
    )
    return fig


def plot_topn(top_df: pd.DataFrame, topn: int = 10) -> go.Figure:
    # Reverse for horizontal bar chart (highest on top)
    df_plot = top_df.head(topn).iloc[::-1]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["數值"], 
        y=df_plot["項目"].astype(str), 
        orientation='h',
        name="數值",
        marker_color="#3b82f6",
        text=df_plot["數值"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside"
    ))
    fig.update_layout(
        title=dict(text=f"TOP{topn} 排名", font=dict(size=16)),
        xaxis_title="數值",
        yaxis_title="項目",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=150, r=80, t=80, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e5e7eb"),
        height=max(400, topn * 35),
    )
    return fig


# =========================
# "Memory" state management
# =========================
def init_state():
    if "dfs" not in st.session_state:
        st.session_state.dfs = {}
    if "profiles" not in st.session_state:
        st.session_state.profiles = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []  # chat history
    if "analysis_state" not in st.session_state:
        # last analysis context
        st.session_state.analysis_state = {
            "table_key": "",
            "task_type": "",
            "years": [],
            "metric_col": "",
            "metric_kind": "",
            "filters": [],
            "dim_col": "",
            "last_table_name": "",
            "last_result_table_name": "",
        }
    if "last_artifacts" not in st.session_state:
        # last produced result tables to support follow-ups
        st.session_state.last_artifacts = {
            "tables": {},   # name -> df
            "fig": None,    # plotly fig
            "meta": {},     # meta info (yoy / trend / topn)
            "kind": "",     # "yoy" | "trend" | "topn" | "preview"
        }


# =========================
# App start
# =========================
api_key = require_api_key()
client = OpenAI(api_key=api_key)
init_state()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="margin: 0;">🧠 AI 分析助理</h2>
        <p style="opacity: 0.6; font-size: 0.75rem;">上下文記憶版</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("📁 上傳資料")
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
                df2 = light_datetime_parse(df)
                dfs[k] = df2
                profiles[k] = build_profile(k, df2)

        st.session_state.dfs = dfs
        st.session_state.profiles = profiles

    if st.session_state.dfs:
        st.success(f"✅ 已載入 {len(st.session_state.dfs)} 張表")
        with st.expander("查看表清單", expanded=False):
            for k, p in st.session_state.profiles.items():
                st.write(f"- {k}（{p.rows}×{p.cols}）")
    else:
        st.info("📤 先上傳 Excel 才能開始問")

    st.divider()
    if st.button("🧹 清除對話", use_container_width=True):
        st.session_state.messages = []
        st.session_state.analysis_state = {
            "table_key": "", "task_type": "", "years": [], "metric_col": "",
            "metric_kind": "", "filters": [], "dim_col": "",
            "last_table_name": "", "last_result_table_name": "",
        }
        st.session_state.last_artifacts = {"tables": {}, "fig": None, "meta": {}, "kind": ""}
        st.rerun()
    
    st.divider()
    
    with st.expander("💡 使用技巧", expanded=False):
        st.markdown("""
        **比較分析（同月份對齊）**
        - 「比較 2023 vs 2024 每月銷售數量」
        - 「對比去年今年的採購金額」
        
        **圖表切換**
        - 「改成折線圖」
        - 「只畫成長率」
        - 「換成長條圖」
        
        **趨勢分析**
        - 「做每月營收趨勢圖」
        
        **排名分析**
        - 「TOP 10 產品銷售」
        """)

st.title("💬 直接問（中文語意理解｜上下文記憶｜穩定比較圖）")

if not st.session_state.dfs:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 16px; margin: 2rem 0;">
        <h2>👋 歡迎使用 AI 資料分析助理</h2>
        <p style="opacity: 0.7;">請先在左側上傳 Excel 資料檔案，即可開始智能分析</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for name, df in (m.get("tables") or {}).items():
            st.write(f"**{name}**")
            st.dataframe(df, use_container_width=True)
        if m.get("fig") is not None:
            st.plotly_chart(m["fig"], use_container_width=True)

prompt = st.chat_input("例：比較 2023 vs 2024 每月銷售數量（同月份對齊），然後幫我改成折線圖")

if prompt:
    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            dfs_all: Dict[str, pd.DataFrame] = st.session_state.dfs
            profiles: Dict[str, TableProfile] = st.session_state.profiles

            # 0) Detect follow-up viz intent
            viz_intent = detect_viz_followup_intent(prompt)

            # If user asks "改成圖表" and we have last_artifacts, do it WITHOUT calling LLM
            if viz_intent is not None and st.session_state.last_artifacts.get("kind"):
                kind = st.session_state.last_artifacts["kind"]
                tables = st.session_state.last_artifacts["tables"]
                meta = st.session_state.last_artifacts.get("meta") or {}

                fig = None
                final_answer = ""
                result_tables = tables or {}

                if kind == "yoy":
                    # find yoy table
                    yoy_df = None
                    for kname, kdf in tables.items():
                        if {"月份", "成長率(%)"}.issubset(set(kdf.columns)):
                            yoy_df = kdf
                            break
                    if yoy_df is not None and meta:
                        chart_type = viz_intent if viz_intent != "auto" else "bar"
                        fig = plot_yoy(yoy_df, meta, chart_type=chart_type)
                        final_answer = pretty_md({
                            "title": "已依照你的要求更新圖表",
                            "bullets": [
                                f"圖表類型：{chart_type}",
                                "沿用上一輪的資料與欄位（已保留同月份對齊）",
                            ],
                            "observations": [
                                "這次不重新跑分析，只是把上一輪結果換成你指定的圖表呈現。",
                            ],
                            "suggestions": [
                                "你也可以說：『只畫成長率』或『換成長條圖』。",
                            ]
                        })
                    else:
                        final_answer = pretty_md({
                            "title": "目前沒有可直接轉圖的 YoY 結果",
                            "bullets": ["我找不到上一輪的 YoY 表格欄位（月份/成長率）。"],
                            "suggestions": ["你可以再問一次：『比較 2023 vs 2024 每月 XXX（同月份對齊）』我會重算並畫圖。"]
                        })

                elif kind == "trend":
                    trend_df = None
                    for kname, kdf in tables.items():
                        if {"年月", "數值"}.issubset(set(kdf.columns)):
                            trend_df = kdf
                            break
                    if trend_df is not None:
                        fig = plot_trend(trend_df)
                        final_answer = pretty_md({
                            "title": "已把上一輪結果做成圖表",
                            "bullets": ["圖表：月度趨勢折線圖", "沿用上一輪的數值欄位與期間"],
                            "observations": ["這次只做視覺化，不重新計算。"],
                        })
                    else:
                        final_answer = pretty_md({
                            "title": "目前沒有可直接轉圖的趨勢結果",
                            "bullets": ["找不到上一輪的『年月/數值』欄位。"],
                            "suggestions": ["你可以直接問：『做成月度趨勢圖（用日期欄 XXX）』。"]
                        })

                elif kind == "topn":
                    top_df = None
                    for kname, kdf in tables.items():
                        if {"項目", "數值"}.issubset(set(kdf.columns)):
                            top_df = kdf
                            break
                    if top_df is not None:
                        fig = plot_topn(top_df, len(top_df))
                        final_answer = pretty_md({
                            "title": "已把 TOPN 結果畫成圖表",
                            "bullets": [f"圖表：TOP{len(top_df)} 水平長條圖", "沿用上一輪結果"],
                        })
                    else:
                        final_answer = pretty_md({
                            "title": "目前沒有可直接轉圖的 TOPN 結果",
                            "bullets": ["找不到上一輪的『項目/數值』欄位。"],
                        })

                else:
                    final_answer = pretty_md({
                        "title": "我知道你要改成圖表，但上一輪結果類型不明",
                        "bullets": ["我已保留上一輪表格輸出，你可以再說一次要畫哪個欄位/哪種圖。"],
                    })

                # render
                st.markdown(final_answer)
                for name, df in result_tables.items():
                    st.write(f"**{name}**")
                    st.dataframe(df, use_container_width=True)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                # save history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "tables": result_tables,
                    "fig": fig,
                })

                # update last artifacts fig
                st.session_state.last_artifacts["fig"] = fig
                st.stop()

            # 1) Normal flow: choose candidate tables
            selected_keys = pick_tables(prompt, profiles, TOPK_TABLES)
            tables_json = tables_context_json(selected_keys, profiles)

            # 2) Planner (with memory)
            plan = llm_plan(
                client=client,
                question=prompt,
                tables_json=tables_json,
                model=DEFAULT_MODEL,
                messages=st.session_state.messages,
                last_state=st.session_state.analysis_state,
            )

            # 3) Determine table_key with fallback (and memory)
            table_key = (plan.get("table_key") or "").strip()
            if not table_key:
                # if user follow-up but no table_key, use last state
                table_key = st.session_state.analysis_state.get("table_key") or ""
            if table_key not in dfs_all:
                table_key = selected_keys[0] if selected_keys else ""

            if not table_key or table_key not in dfs_all:
                st.error("找不到可用的資料表，請確認已上傳正確的 Excel 檔案。")
                st.stop()

            df = dfs_all[table_key].copy()

            # 4) Apply filters
            filters = plan.get("filters") or []
            df_f = apply_filters(df, filters)

            # 5) Determine task_type
            compare_intent = detect_compare_intent(prompt)
            task_type = (plan.get("task_type") or "").strip() or "generic_summary"
            if compare_intent:
                task_type = "compare_yoy_monthly"

            result_tables: Dict[str, pd.DataFrame] = {}
            fig = None
            meta_out: dict = {}
            kind = ""
            final_answer = ""

            try:
                if task_type == "compare_yoy_monthly":
                    yoy_df, meta = build_yoy_table(df_f, prompt, plan)
                    meta_out = meta
                    result_tables["同月份對齊比較（YoY）"] = yoy_df

                    # default chart type (bar)
                    fig = plot_yoy(yoy_df, meta, chart_type="bar")
                    kind = "yoy"

                    valid = yoy_df.dropna(subset=["成長率(%)"])
                    if len(valid) > 0:
                        max_row = valid.loc[valid["成長率(%)"].idxmax()]
                        min_row = valid.loc[valid["成長率(%)"].idxmin()]
                        bullets = [
                            f"使用表：{table_key}",
                            f"比較方式：**同月份對齊 01~12**（不把兩年接成時間軸）",
                            f"指標欄位：{meta['metric_col']}（{meta['metric_kind']}）",
                            f"最高成長月份：{max_row['月份']} 月（{max_row['成長率(%)']:.1f}%）",
                            f"最低成長月份：{min_row['月份']} 月（{min_row['成長率(%)']:.1f}%）",
                        ]
                    else:
                        bullets = [
                            f"使用表：{table_key}",
                            "比較方式：**同月份對齊 01~12**",
                            f"指標欄位：{meta['metric_col']}（{meta['metric_kind']}）",
                            "部分月份基準年為 0，成長率以 NaN 處理。",
                        ]

                    final_answer = pretty_md({
                        "title": "比較結果",
                        "bullets": bullets,
                        "observations": [
                            f"圖表：{meta['y1']}/{meta['y2']} 以同月份並排方式呈現，右軸為成長率(%)。",
                            "你可以直接接一句：『改成折線圖』或『只畫成長率』，我會沿用這份結果快速換圖。",
                        ],
                        "suggestions": [
                            "如果你要同時比較『數量』與『金額』，請明確說：『再做一張金額的 YoY』，我會分開輸出兩張圖。",
                            "如果你要看『差異最大的產品/業務/客戶』，你可以再補：『再列 TOP10 差異』。",
                        ],
                        "notes": [
                            (f"規劃備註：{plan.get('notes','')}".strip() if plan.get("notes") else "規劃備註：（無）"),
                        ],
                    })

                elif task_type == "trend_monthly":
                    trend_df, meta = build_trend_monthly(df_f, prompt, plan)
                    meta_out = meta
                    result_tables["月度趨勢"] = trend_df
                    fig = plot_trend(trend_df)
                    kind = "trend"

                    final_answer = pretty_md({
                        "title": "月度趨勢",
                        "bullets": [
                            f"使用表：{table_key}",
                            f"指標欄位：{meta['metric_col']}（{meta['metric_kind']}）",
                            (f"期間：{trend_df['年月'].min()} ~ {trend_df['年月'].max()}" if len(trend_df) else "期間：未知"),
                        ],
                        "observations": [
                            "折線圖用年月做 x 軸，數值做 y 軸。",
                        ],
                        "suggestions": [
                            "如果你想要『只看銷貨/進貨』，請告訴我哪個欄位是『單別名稱』或分類欄位，我會加上篩選。",
                            "你也可以直接說：『把剛剛那張表改成圖表』。",
                        ],
                    })

                elif task_type == "topn":
                    top_df, meta = build_topn(df_f, prompt, plan)
                    meta_out = meta
                    topn = meta.get("topn", TOPN_DEFAULT)
                    result_tables[f"TOP{topn}"] = top_df
                    fig = plot_topn(top_df, topn)
                    kind = "topn"

                    final_answer = pretty_md({
                        "title": f"TOP{topn} 排名",
                        "bullets": [
                            f"使用表：{table_key}",
                            f"維度：{meta['dim_col']}",
                            f"指標欄位：{meta['metric_col']}（{meta['metric_kind']}）",
                        ],
                        "observations": [
                            "表格已依照數值由大到小排序。",
                            "圖表使用水平長條圖，最高值在上方。",
                        ],
                        "suggestions": [
                            "如果你要『2023 vs 2024 的 TOP 差異』，請回：『比較兩年同一批項目差異』。",
                            "你也可以說：『換成別的維度』或『改看金額』。",
                        ],
                    })

                else:
                    # generic summary
                    summary_df, meta = build_generic_summary(df_f, prompt, plan)
                    meta_out = meta
                    result_tables["資料摘要"] = summary_df
                    result_tables["資料預覽"] = df_safe_preview(df_f, 30)
                    kind = "preview"

                    final_answer = pretty_md({
                        "title": "資料摘要",
                        "bullets": [
                            f"使用表：{table_key}",
                            f"總筆數：{len(df_f):,}",
                            f"欄位數：{len(df_f.columns)}",
                        ],
                        "observations": [
                            "由於未偵測到明確的分析意圖，先提供基本摘要。",
                        ],
                        "suggestions": [
                            "你可以試著說：『比較 2023 vs 2024 每月銷售數量』",
                            "或是：『做 TOP 10 產品排名』、『畫每月趨勢圖』",
                        ],
                    })

            except Exception as e:
                final_answer = pretty_md({
                    "title": "分析過程發生錯誤",
                    "bullets": [f"錯誤訊息：{str(e)[:200]}"],
                    "suggestions": [
                        "請確認資料欄位是否正確。",
                        "你可以重新描述需求，或指定具體的欄位名稱。",
                    ],
                })
                result_tables["資料預覽"] = df_safe_preview(df_f, 20)
                kind = "preview"

            # Render results
            st.markdown(final_answer)
            for name, df_out in result_tables.items():
                st.write(f"**{name}**")
                st.dataframe(df_out, use_container_width=True)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "tables": result_tables,
                "fig": fig,
            })

            # Update analysis state (memory)
            st.session_state.analysis_state = {
                "table_key": table_key,
                "task_type": task_type,
                "years": plan.get("years") or [],
                "metric_col": meta_out.get("metric_col", ""),
                "metric_kind": meta_out.get("metric_kind", ""),
                "filters": filters,
                "dim_col": meta_out.get("dim_col", ""),
                "last_table_name": table_key,
                "last_result_table_name": list(result_tables.keys())[0] if result_tables else "",
            }

            # Update last artifacts for follow-up
            st.session_state.last_artifacts = {
                "tables": result_tables,
                "fig": fig,
                "meta": meta_out,
                "kind": kind,
            }