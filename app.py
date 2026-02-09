"""
🤖 企業級 AI 智能數據分析系統 v16.1
================================================================
三引擎架構: 數據清洗 · 語意感知 · 邏輯大腦
v16.1 微調修復:
  ✅ 跨年份比較圖表修復（月份軸 + 年份分色）
  ✅ 側邊欄 UI 強化（展開狀態 + 按鈕優化）
  
繼承 v15.0 核心:
  ✅ ASP 正確公式: 未稅淨額/淨數量（禁用含稅）
  ✅ 優先使用「含正負號」欄位（系統原生淨額）
  ✅ 銷退自動轉負 + 禁止重複扣除
  ✅ 查無資料回報機制
  ✅ 圖表強制觸發（One-Shot Charting）
  ✅ 語意欄位偵測（單別≠品名）
================================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import io
import traceback

# ============================================================================
# 全域配置
# ============================================================================
PASSWORD = "0413"
EMBEDDED_API_KEY = st.secrets["OPENAI_API_KEY"]
PAGE_TITLE = "🤖 AI 智能數據分析師"
PAGE_ICON = "🤖"
GPT_MODEL = "gpt-4o"
MAX_RETRIES = 3
TEMPERATURE = 0.01

COLOR_PALETTE = {
    'default': ['#FF6B35', '#004E89', '#2ECC71', '#9B59B6', '#F39C12',
                '#1ABC9C', '#E74C3C', '#3498DB', '#E91E63', '#00BCD4'],
    'blue': ['#004E89', '#0066B3', '#3498DB', '#5DADE2', '#85C1E9', '#AED6F1'],
    'red': ['#E74C3C', '#C0392B', '#F1948A', '#EC7063', '#CD6155', '#F5B7B1'],
    'green': ['#2ECC71', '#27AE60', '#58D68D', '#82E0AA', '#ABEBC6', '#D5F5E3'],
    'orange': ['#FF6B35', '#E67E22', '#F39C12', '#F8C471', '#FAD7A0', '#FDEBD0'],
    'purple': ['#9B59B6', '#8E44AD', '#BB8FCE', '#D2B4DE', '#E8DAEF', '#F4ECF7'],
    'rainbow': ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB', '#9B59B6', '#1ABC9C'],
    'pastel': ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#E0BBE4'],
    'dark': ['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7'],
    'warm': ['#E74C3C', '#E67E22', '#F39C12', '#D35400', '#C0392B'],
    'cool': ['#3498DB', '#2980B9', '#1ABC9C', '#16A085', '#2ECC71', '#00BCD4'],
}

COLOR_NAME_MAP = {
    '藍': 'blue', '藍色': 'blue', 'blue': 'blue',
    '紅': 'red', '紅色': 'red', 'red': 'red',
    '綠': 'green', '綠色': 'green', 'green': 'green',
    '橙': 'orange', '橙色': 'orange', 'orange': 'orange', '橘': 'orange', '橘色': 'orange',
    '紫': 'purple', '紫色': 'purple', 'purple': 'purple',
    '黃': 'orange', '黃色': 'orange',
    '彩虹': 'rainbow', '多彩': 'rainbow', 'rainbow': 'rainbow',
    '柔和': 'pastel', '粉彩': 'pastel', 'pastel': 'pastel',
    '深色': 'dark', '暗色': 'dark', 'dark': 'dark',
    '暖色': 'warm', 'warm': 'warm',
    '冷色': 'cool', 'cool': 'cool',
}

CHART_TYPE_MAP = {
    '長條圖': 'bar', '柱狀圖': 'bar', '直條圖': 'bar', 'bar': 'bar',
    '分組長條圖': 'grouped_bar', '分組': 'grouped_bar', '並排': 'grouped_bar',
    'grouped_bar': 'grouped_bar',
    '堆疊長條圖': 'stacked_bar', '堆疊': 'stacked_bar', 'stacked_bar': 'stacked_bar',
    '折線圖': 'line', '線圖': 'line', '趨勢圖': 'line', 'line': 'line',
    '面積圖': 'area', 'area': 'area',
    '堆疊面積圖': 'stacked_area', 'stacked_area': 'stacked_area',
    '圓餅圖': 'pie', '餅圖': 'pie', 'pie': 'pie',
    '環形圖': 'donut', '甜甜圈': 'donut', 'donut': 'donut',
    '散點圖': 'scatter', 'scatter': 'scatter',
    '水平長條圖': 'horizontal_bar', '水平': 'horizontal_bar', '橫條圖': 'horizontal_bar',
    'horizontal_bar': 'horizontal_bar',
    '瀑布圖': 'waterfall', 'waterfall': 'waterfall',
    '漏斗圖': 'funnel', 'funnel': 'funnel',
    '雷達圖': 'radar', 'radar': 'radar',
    '熱力圖': 'heatmap', 'heatmap': 'heatmap',
    '樹狀圖': 'treemap', 'treemap': 'treemap',
    '旭日圖': 'sunburst', 'sunburst': 'sunburst',
}

CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #FF6B35 0%, #004E89 50%, #2ECC71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem;
    }
    .sub-header { text-align: center; color: #666; font-size: 1.05rem; margin-bottom: 2rem; }
    .data-header {
        background: linear-gradient(90deg, #FF6B35 0%, #FF8F6B 100%);
        color: white; padding: 0.8rem 1.2rem; border-radius: 10px 10px 0 0;
        font-weight: 700; font-size: 1.05rem;
    }
    .thinking-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #3498DB; border-radius: 0 10px 10px 0;
        padding: 1.2rem 1.5rem; margin: 1rem 0; font-size: 0.95rem;
        color: #2C3E50; line-height: 1.6;
    }
    .engine-report {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #16a34a; border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.88rem; line-height: 1.7;
    }
    .sheet-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 0.3rem 0.8rem; border-radius: 15px;
        font-size: 0.85rem; margin: 0.2rem; font-weight: 600;
    }
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;}
    /* 保留 Streamlit 的 header 以顯示側邊欄展開按鈕 */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    /* 確保側邊欄控制按鈕可見 */
    button[kind="header"] {
        visibility: visible !important;
    }
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block !important;
    }
    .stButton > button {
        border-radius: 10px; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    /* 側邊欄按鈕統一尺寸 */
    [data-testid="stSidebar"] .stButton > button {
        min-height: 38px;
        padding: 0.4rem 0.5rem;
        font-size: 0.85rem;
    }
    /* 強化側邊欄展開按鈕的可見度 */
    [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #FF6B35 0%, #004E89 100%) !important;
        color: white !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 12px 6px !important;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="collapsedControl"]:hover {
        transform: translateX(3px) !important;
        box-shadow: 3px 3px 15px rgba(0,0,0,0.4) !important;
    }
    [data-testid="collapsedControl"] svg {
        width: 24px !important;
        height: 24px !important;
        color: white !important;
    }
</style>
"""

# ============================================================================
# 安全工具
# ============================================================================
def safe_get_string(value, default=''):
    if value is None: return default
    if isinstance(value, str): return value
    if isinstance(value, (list, tuple)): return str(value[0]) if value else default
    return str(value)

def format_number(x):
    try:
        if pd.isna(x): return ''
        if isinstance(x, (int, float, np.integer, np.floating)):
            return f"{x:,.0f}" if abs(x) >= 1 else f"{x:.2f}"
        return str(x)
    except Exception:
        return str(x)


# ╔══════════════════════════════════════════════════════════════╗
# ║  ENGINE 1: 數據清洗引擎                                      ║
# ╚══════════════════════════════════════════════════════════════╝
class DataCleaningEngine:
    SUMMARY_KW = ['總表', '彙總', 'summary', 'total', '合計', '統計']
    DETAIL_KW = ['明細', '交易', 'detail', 'raw', 'transaction', '銷貨', '進貨', '出貨']
    RETURN_KW = ['銷退', '退貨', '退回', 'return', 'credit', '折讓', 'refund', '銷折']
    TYPE_KW = ['單別', '單據類型', 'type', '單別名稱']
    NUMERIC_KW = ['數量', '金額', 'qty', 'amt', '未稅', '含稅', '稅額', '單價',
                  'price', 'amount', 'total', 'cost', '成本', '營收', 'revenue',
                  '毛利', 'profit', '淨額', '庫存', 'quantity', '費用']
    ID_KW = ['id', '編號', '序號', 'no', 'code', '代號', 'sku', '單號',
             'number', '電話', '流水號']
    DATE_KW = ['日期', 'date', '時間']
    SKIP_SHEET_KW = ['index', 'readme', '說明', 'template']
    SAFE_DEDUP_COLS = ['唯一流水號(子檔)', '序號', '流水號']

    def __init__(self):
        self.log = []
        self.stats = {}
        self._reset()

    def _reset(self):
        self.log = []
        self.stats = dict(total_files=0, total_sheets_read=0,
                          sheets_skipped_summary=0, sheets_skipped_other=0,
                          rows_before_dedup=0, rows_after_dedup=0,
                          duplicates_removed=0, dedup_strategy='',
                          return_rows_negated=0, return_type_col='',
                          numeric_cols_standardized=0, date_cols_processed=0)

    def _m(self, text, keywords):
        t = text.strip().lower()
        return any(k in t for k in keywords)

    # ── 主流程 ──
    def clean(self, files, selected_sheets=None):
        self._reset()
        frames = []
        meta = dict(files=[], sheets=[], total_rows=0, columns=[],
                    numeric_columns=[], date_columns=[], categorical_columns=[],
                    years=[], sample_data={}, unique_values={},
                    load_errors=[], data_summary={})

        for f in files:
            self.stats['total_files'] += 1
            try:
                fb = f.read(); f.seek(0)
                xls = pd.ExcelFile(io.BytesIO(fb))
                fname = f.name
                to_load = self._resolve_sheets(xls.sheet_names, fname, selected_sheets)
                for sn in to_load:
                    df_s = self._load_sheet(xls, sn, fname)
                    if df_s is not None and len(df_s) > 0:
                        frames.append(df_s)
                        meta['sheets'].append(dict(file=fname, sheet=sn, rows=len(df_s),
                                                   columns=[c for c in df_s.columns if not str(c).startswith('_')]))
                        self.stats['total_sheets_read'] += 1
                        self.log.append(f"✅ [{fname}] → '{sn}' ({len(df_s):,} 行)")
                meta['files'].append(fname)
            except Exception as e:
                meta['load_errors'].append(f"{f.name}: {e}")
                self.log.append(f"❌ {f.name}: {e}")

        if not frames:
            return None, meta, self.stats

        combined = pd.concat(frames, ignore_index=True, sort=False)
        self.stats['rows_before_dedup'] = len(combined)

        combined = self._safe_dedup(combined)
        combined = self._standardize_numeric(combined)
        combined = self._negate_returns(combined)
        combined = self._convert_dates(combined)
        self._finalize_meta(combined, meta)
        self.log.append(f"🎯 清洗完畢: {len(combined):,} 行 × {len(combined.columns)} 欄")
        return combined, meta, self.stats

    def _resolve_sheets(self, names, fname, selected):
        if selected and fname in selected:
            return selected[fname]
        valid = [s for s in names if not self._m(s, self.SKIP_SHEET_KW)]
        if not valid:
            valid = names
        has_sum = any(self._m(s, self.SUMMARY_KW) for s in valid)
        has_det = any(self._m(s, self.DETAIL_KW) for s in valid)
        if has_sum and has_det:
            kept = []
            for s in valid:
                if self._m(s, self.SUMMARY_KW):
                    self.stats['sheets_skipped_summary'] += 1
                    self.log.append(f"🚫 智慧路由丟棄總表: [{fname}] → '{s}'")
                else:
                    kept.append(s)
            return kept
        return valid

    def _load_sheet(self, xls, sheet, fname):
        try:
            raw = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=15)
            if raw.empty: return None
            hr = self._detect_header(raw)
            df = pd.read_excel(xls, sheet_name=sheet, header=hr)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
            df.dropna(how='all', inplace=True)
            if len(df) == 0: return None
            df['_來源檔案'] = fname
            df['_工作表'] = sheet
            return df
        except Exception:
            return None

    def _detect_header(self, raw):
        for i in range(min(10, len(raw))):
            row = raw.iloc[i]
            v = sum(1 for val in row if pd.notna(val) and isinstance(val, str)
                    and len(str(val).strip()) > 0
                    and not str(val).strip().replace('.','').replace('-','').replace('/','').isdigit())
            if v >= max(3, len(row) * 0.3):
                return i
        return 0

    def _safe_dedup(self, df):
        for sc in self.SAFE_DEDUP_COLS:
            if sc in df.columns:
                b = len(df)
                df = df.drop_duplicates(subset=[sc], keep='first').reset_index(drop=True)
                r = b - len(df)
                self.stats.update(rows_after_dedup=len(df), duplicates_removed=r,
                                  dedup_strategy=f"基於: {sc}")
                if r > 0: self.log.append(f"🗑️ 安全去重({sc}): -{r:,}")
                return df

        key_cols = [c for c in ['日期(轉換)', '進銷單號', '產品代號', '數量'] if c in df.columns]
        if len(key_cols) >= 2:
            b = len(df)
            df = df.drop_duplicates(subset=key_cols, keep='first').reset_index(drop=True)
            r = b - len(df)
            self.stats.update(rows_after_dedup=len(df), duplicates_removed=r,
                              dedup_strategy=f"基於: {', '.join(key_cols)}")
            if r > 0: self.log.append(f"🗑️ 組合去重: -{r:,}")
        else:
            uc = [c for c in df.columns if not str(c).startswith('_')]
            b = len(df)
            df = df.drop_duplicates(subset=uc, keep='first').reset_index(drop=True)
            r = b - len(df)
            self.stats.update(rows_after_dedup=len(df), duplicates_removed=r,
                              dedup_strategy="全欄位去重")
            if r > 0: self.log.append(f"🗑️ 全欄位去重: -{r:,}")
        return df

    def _standardize_numeric(self, df):
        cnt = 0
        for col in df.columns:
            if str(col).startswith('_'): continue
            if self._m(col, self.NUMERIC_KW) and not self._m(col, self.ID_KW):
                try:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(',','',regex=False)\
                            .str.replace('$','',regex=False).str.replace('NT','',regex=False)\
                            .str.replace('￥','',regex=False).str.strip()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    cnt += 1
                except Exception: pass
        self.stats['numeric_cols_standardized'] = cnt
        if cnt > 0: self.log.append(f"🔢 數值標準化: {cnt} 欄")
        return df

    def _negate_returns(self, df):
        signed = [c for c in df.columns if '正負號' in str(c) or 'net' in str(c).lower()]
        if signed:
            self.log.append(f"ℹ️ 已有正負號欄位 {signed}，跳過銷退轉負")
            return df
        type_cols = [c for c in df.columns if self._m(c, self.TYPE_KW) and not str(c).startswith('_')]
        if not type_cols:
            self.log.append("ℹ️ 無單別欄位，跳過銷退轉負")
            return df
        num_cols = [c for c in df.columns
                    if self._m(c, self.NUMERIC_KW) and not self._m(c, self.ID_KW)
                    and not str(c).startswith('_') and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols: return df

        total = 0
        for tc in type_cols:
            mask = df[tc].apply(lambda v: False if pd.isna(v) else self._m(str(v), self.RETURN_KW))
            n = mask.sum()
            if n > 0:
                for nc in num_cols:
                    df.loc[mask, nc] = -df.loc[mask, nc].abs()
                total += n
                self.stats['return_type_col'] = tc
                self.log.append(f"🔄 銷退轉負: '{tc}' {n:,} 筆 → {len(num_cols)} 個數值欄位取負")
        self.stats['return_rows_negated'] = total
        return df

    def _convert_dates(self, df):
        cnt = 0
        for col in df.columns:
            if str(col).startswith('_'): continue
            if self._m(col, self.DATE_KW):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    cnt += 1
                except Exception: pass
        if '日期(轉換)' in df.columns:
            try:
                df['日期(轉換)'] = pd.to_datetime(df['日期(轉換)'], errors='coerce')
                df['_年份'] = df['日期(轉換)'].dt.year.astype('Int64')
                df['_月份'] = df['日期(轉換)'].dt.month.astype('Int64')
                df['_季度'] = df['日期(轉換)'].dt.quarter.astype('Int64')
                df['_年月'] = df['日期(轉換)'].dt.strftime('%Y-%m')
                cnt += 1
                self.log.append("📅 日期標準化: 日期(轉換) → _年份/_月份/_季度/_年月")
            except Exception:
                try:
                    p = pd.to_datetime(df['日期(轉換)'], errors='coerce')
                    df['_年份'] = p.dt.year; df['_月份'] = p.dt.month
                except Exception: pass
        if '_年份' not in df.columns:
            for col in df.columns:
                if self._m(col, self.DATE_KW) and not str(col).startswith('_'):
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df['_年份'] = df[col].dt.year.astype('Int64')
                        df['_月份'] = df[col].dt.month.astype('Int64')
                        df['_年月'] = df[col].dt.strftime('%Y-%m')
                        break
        self.stats['date_cols_processed'] = cnt
        return df

    def _finalize_meta(self, df, meta):
        meta['total_rows'] = len(df)
        meta['columns'] = [c for c in df.columns if not str(c).startswith('_')]
        for col in df.columns:
            if str(col).startswith('_'): continue
            if pd.api.types.is_numeric_dtype(df[col]):
                meta['numeric_columns'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                meta['date_columns'].append(col)
            else:
                meta['categorical_columns'].append(col)
        if '_年份' in df.columns:
            try:
                meta['years'] = sorted([int(y) for y in df['_年份'].dropna().unique()])
            except Exception: pass
        for col in ['對方品名/品名備註', '產品代號', '客戶供應商簡稱', '單別名稱', '_工作表']:
            if col in df.columns:
                try: meta['unique_values'][col] = df[col].dropna().unique().tolist()[:100]
                except Exception: pass
        for col in meta['columns'][:15]:
            try: meta['sample_data'][col] = df[col].dropna().head(5).tolist()
            except Exception: pass
        summary = {}
        if '_工作表' in df.columns:
            summary['sheet_distribution'] = df['_工作表'].value_counts().to_dict()
        if '_年份' in df.columns:
            summary['year_distribution'] = df['_年份'].value_counts().sort_index().to_dict()
        for col in meta['numeric_columns'][:5]:
            if col in df.columns:
                try:
                    summary[f'{col}_stats'] = dict(
                        sum=float(df[col].sum()), mean=float(df[col].mean()),
                        min=float(df[col].min()), max=float(df[col].max()))
                except Exception: pass
        meta['data_summary'] = summary


# ╔══════════════════════════════════════════════════════════════╗
# ║  ENGINE 2: 語意感知模組                                       ║
# ╚══════════════════════════════════════════════════════════════╝
class SemanticDetectionModule:
    def audit(self, df, meta):
        cols = [c for c in df.columns if not str(c).startswith('_')]
        signed_cols = [c for c in cols if '正負號' in c or '含正負' in c]
        a = dict(
            product_name_cols=self._names(cols),
            product_code_cols=self._codes(cols),
            date_cols=self._dates(cols),
            numeric_cols=meta.get('numeric_columns', []),
            type_cols=self._types(cols),
            customer_cols=self._custs(cols),
            has_signed_cols=signed_cols,
            qty_signed_col=self._find_col(signed_cols, ['數量']),
            amount_signed_col=self._find_col(signed_cols, ['金額', '未稅']),
            amount_untaxed_col=self._find_col(cols, ['未稅金額', '未稅']),
            amount_taxed_col=self._find_col(cols, ['含稅金額', '含稅']),
            qty_col=self._find_col(cols, ['數量']),
        )
        a['summary_text'] = self._summary(a)
        return a

    def _find_col(self, cols, keywords):
        """找第一個匹配的欄位名"""
        for c in cols:
            cl = c.lower()
            if any(k in cl for k in keywords):
                return c
        return None

    def _names(self, cols):
        nk = ['品名', '備註', 'name', 'description', '品項', '商品']
        ek = ['代號', 'code', 'id', 'sku', '編號', '貨號', '單別', '類別', 'type', '類型']
        return [c for c in cols if any(k in c.lower() for k in nk) and not any(k in c.lower() for k in ek)]

    def _codes(self, cols):
        ck = ['代號', 'code', 'sku', '貨號', '料號']
        return [c for c in cols if any(k in c.lower() for k in ck)]

    def _dates(self, cols):
        dk = ['日期', 'date', '時間']
        return [c for c in cols if any(k in c.lower() for k in dk)]

    def _types(self, cols):
        tk = ['單別', '類型', 'type', '單別名稱']
        return [c for c in cols if any(k in c.lower() for k in tk)]

    def _custs(self, cols):
        ck = ['客戶', '廠商', 'customer', 'vendor', '供應商', '公司', '業務']
        return [c for c in cols if any(k in c.lower() for k in ck)]

    def _summary(self, a):
        lines = [
            f"📦 產品名稱欄位: {a['product_name_cols'] or '未偵測'}",
            f"🏷️ 產品代號欄位: {a['product_code_cols'] or '未偵測'}",
            f"📅 日期欄位: {a['date_cols'] or '未偵測'}",
            f"💰 數值欄位: {a['numeric_cols'][:8]}...",
            f"📋 單別欄位: {a['type_cols'] or '未偵測'}",
            f"👤 客戶欄位: {a['customer_cols'] or '未偵測'}",
        ]
        if a.get('has_signed_cols'):
            lines.append(f"✅ 含正負號欄位: {a['has_signed_cols']}")
        if a.get('qty_signed_col'):
            lines.append(f"📊 淨數量欄位: {a['qty_signed_col']}")
        if a.get('amount_signed_col'):
            lines.append(f"💵 淨金額欄位: {a['amount_signed_col']}")
        return '\n'.join(lines)


# ╔══════════════════════════════════════════════════════════════╗
# ║  ENGINE 3: 邏輯大腦層                                        ║
# ╚══════════════════════════════════════════════════════════════╝
class LogicalBrainEngine:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = GPT_MODEL

    def _schema(self, df, meta):
        p = []
        p.append(f"## 資料: {len(df):,} 筆, 年份: {meta.get('years',[])}, 檔案: {meta.get('files',[])}, 工作表: {len(meta.get('sheets',[]))} 個")
        if meta.get('data_summary',{}).get('year_distribution'):
            p.append("\n## 年份分布")
            for y, c in meta['data_summary']['year_distribution'].items():
                p.append(f"- {y}年: {c:,} 筆")
        p.append("\n## 欄位")
        for col in df.columns:
            if str(col).startswith('_') and col not in ['_年份','_月份','_季度','_年月','_工作表']:
                continue
            try: dt, u = str(df[col].dtype), df[col].nunique()
            except: dt, u = '?', 0
            tag = ""
            if col == '對方品名/品名備註': tag = "⭐[品名-str.contains()]"
            elif col == '產品代號': tag = "⭐[代號-英數]"
            elif '正負號' in str(col) or '含正負' in str(col): tag = "⭐⭐[已含正負號-優先使用!]"
            elif '未稅' in str(col) and '正負' not in str(col): tag = "⭐[未稅金額-算ASP用此欄]"
            elif col in ('數量',): tag = "⭐[數量-後端已轉負]"
            elif '含稅' in str(col): tag = "⚠️[含稅-不要用來算ASP]"
            elif '金額' in str(col): tag = "⭐[金額]"
            elif col == '客戶供應商簡稱': tag = "⭐[客戶]"
            elif col == '單別名稱': tag = "⭐[單別-交易類型,不是產品名!]"
            elif col == '_工作表': tag = "⭐[工作表來源]"
            elif col == '_年份': tag = "[整數]"
            p.append(f"- **{col}** ({dt}) {u:,}唯一值 {tag}")
        p.append("\n### 輔助欄位: _年份(int), _月份(int), _季度(int), _年月(str), _工作表(str)")
        if meta.get('unique_values'):
            p.append("\n## 重要欄位值")
            for col, vals in meta['unique_values'].items():
                p.append(f"### {col} (共{len(vals)})\n```\n{vals[:20]}\n```")
        return '\n'.join(p)

    def _sysprompt(self, df, meta, audit, query):
        schema = self._schema(df, meta)
        nc = audit.get('product_name_cols', ['對方品名/品名備註'])
        cc = audit.get('product_code_cols', ['產品代號'])
        cu = audit.get('customer_cols', ['客戶供應商簡稱'])
        tc = audit.get('type_cols', ['單別名稱'])
        rc, rch = None, None
        for cn, ck in COLOR_NAME_MAP.items():
            if cn in query.lower(): rc = ck; break
        for cn, ct in CHART_TYPE_MAP.items():
            if cn in query.lower(): rch = ct; break

        # 安全防護：type_cols 不能混入 name_cols
        type_col_set = set(tc)
        nc_safe = [c for c in nc if c not in type_col_set]
        if not nc_safe:
            for fb in ['對方品名/品名備註', '品名備註', '品名', '產品名稱']:
                if fb in df.columns:
                    nc_safe = [fb]; break
            if not nc_safe:
                nc_safe = ['對方品名/品名備註']

        name_col = nc_safe[0]
        code_col = cc[0] if cc else '產品代號'
        cust_col = cu[0] if cu else '客戶供應商簡稱'
        type_col = tc[0] if tc else '單別名稱'

        # 偵測正負號欄位
        qty_signed = audit.get('qty_signed_col', '')
        amt_signed = audit.get('amount_signed_col', '')
        amt_untaxed = audit.get('amount_untaxed_col', '')
        has_signed = bool(qty_signed or amt_signed)

        # 決定淨額/淨量的使用欄位
        if qty_signed:
            net_qty_expr = f"df['{qty_signed}']"
            net_qty_note = f"✅ 使用已含正負號欄位 `{qty_signed}`"
        else:
            net_qty_expr = "df['數量']  # 後端已對銷退轉負"
            net_qty_note = "✅ 後端已將銷退數量轉為負數，直接 sum()"

        if amt_signed:
            net_amt_expr = f"df['{amt_signed}']"
            net_amt_note = f"✅ 使用已含正負號欄位 `{amt_signed}`"
        elif amt_untaxed:
            net_amt_expr = f"df['{amt_untaxed}']  # 後端已對銷退轉負"
            net_amt_note = f"✅ 使用未稅金額 `{amt_untaxed}`"
        else:
            net_amt_expr = "df['未稅金額']  # 後端已對銷退轉負"
            net_amt_note = "✅ 後端已將銷退金額轉為負數"

        return f"""你是 v16.1 企業級數據分析 AI，擁有會計邏輯與資料視覺化專長。
資料來源：df（已預載入，含 _年份, _月份, _季度, _年月, _工作表 輔助欄位）。

# 🛡️ 絕對安全協議（違反即失敗）
1. **禁止 import**：嚴禁 `import matplotlib`, `plt`, `pandas`。直接用環境中的 pd, df。
2. **完整性**：禁止寫 `# ...` 省略號。程式碼必須完整可執行。
3. **繪圖**：不要自己畫圖！生成 `chart_config` 字典即可，系統會自動繪圖。
4. **全量運算**：必須處理全部 {len(df):,} 行，禁止 .head() 或 .sample() 計算。

# 📐 會計運算邏輯（本系統核心 — 違反會算錯！）

## 1. 淨額原則 (Net Amount Principle)
- **公式：** Net = Sales - Returns
- **淨數量：** {net_qty_note}
  - 取值：`{net_qty_expr}`
- **淨金額：** {net_amt_note}
  - 取值：`{net_amt_expr}`
- ⚠️ 嚴禁再寫額外減法邏輯（如 sales - returns），否則重複扣除！
- ⚠️ 嚴禁使用「含稅金額」計算平均單價！

## 2. 平均單價 (ASP) 公式 — 最重要！
- **正確：** ASP = SUM(未稅淨額) / SUM(淨數量)
- **程式碼：**
```python
net_amount = filtered['{amt_signed or amt_untaxed or "未稅金額"}'].sum()
net_qty = filtered['{qty_signed or "數量"}'].sum()
asp = net_amount / net_qty if net_qty != 0 else 0
```
- ❌ 嚴禁：`含稅金額.sum() / 數量.sum()` → 這會算錯！

## 3. 產品分類定義 (Product Taxonomy)
- **發泡刷**：`df['{name_col}'].str.contains('發泡刷', case=False, na=False)`
- **DIB陶瓷刷**：`(df['{code_col}'].str.startswith('DIB', na=False)) & (df['{name_col}'].str.contains('陶瓷刷', case=False, na=False))`
- **其他陶瓷刷**：`(~df['{code_col}'].str.startswith('DIB', na=False)) & (df['{name_col}'].str.contains('陶瓷刷', case=False, na=False))`

# ⭐⭐⭐ 欄位用途對照表（違反即錯）⭐⭐⭐

| 要查什麼 | 正確欄位 | 說明 |
|----------|----------|------|
| 產品名稱 (發泡刷/陶瓷刷) | `{name_col}` | str.contains() 模糊搜尋 |
| 產品代號 (BFB-236/DIB-001) | `{code_col}` | ==, startswith |
| 客戶/廠商 (華通/欣興) | `{cust_col}` | str.contains() |
| 單別/交易類型 (銷貨/銷退) | `{type_col}` | ⚠️ 這是交易類型，不是產品！|
| 淨數量 | `{qty_signed or '數量'}` | 已含正負號或後端已轉負 |
| 淨金額(未稅) | `{amt_signed or amt_untaxed or '未稅金額'}` | 算 ASP 必須用此欄 |

## 🚨 絕對禁止
- ❌ `df['{type_col}'].str.contains('發泡刷')` → {type_col} 是交易類型！不是產品！
- ❌ `df['{code_col}'].str.contains('發泡刷')` → 代號欄是英數，不含中文！
- ❌ 用「含稅金額」算 ASP
- ✅ `df['{name_col}'].str.contains('發泡刷', case=False, na=False)` → 正確！

## 📊 圖表觸發機制 (One-Shot Charting)
- 只要問題含 ['圖', 'chart', '趨勢', '佔比', '分佈', '比例', '排名', 'top', 'pie', 'bar', 'line']：
  - `need_chart` **必須** 為 `true`
  - **必須** 生成 `chart_config`
  - 除非查無資料 (len(result_df)==0)

## 🎨 跨年份比較圖表規則 (Visual Fix v16.1) — 極重要！
⚠️ 當用戶意圖為「比較」、「趨勢」、「同期」且涉及**多個年份**時：
### 規則：
1. **X 軸必須使用 `_月份` (1-12)**，不要用 `_年月` 或時間連續軸
2. **Color 必須使用 `_年份`**，且**必須先轉字串**：`df['_年份'].astype(str)`
   - ❌ 錯誤：`color='_年份'` → 會畫成漸層色
   - ✅ 正確：先做 `df['年份(文字)'] = df['_年份'].astype(str)`，然後 `color='年份(文字)'`
3. **結果**：多條線/長條會疊加在同一個月份軸上，可進行同期比較

### 範例：2023-2025 三年同期比較
```python
# 篩選多年資料
multi_year = df[df['_年份'].isin([2023, 2024, 2025])].copy()
multi_year['年份(文字)'] = multi_year['_年份'].astype(str)  # ⭐ 關鍵步驟

# 按年份和月份分組
result_df = multi_year.groupby(['_月份', '年份(文字)'])['數量'].sum().reset_index()
result_df.columns = ['月份', '年份', '數量']
result_df = result_df.sort_values(['月份', '年份'])

# chart_config 設定
chart_config = {{
    'x': '月份',           # ⭐ 使用月份 (1-12)
    'y': '數量',
    'color': '年份',       # ⭐ 使用年份(文字) 作為分類
    'title': '2023-2025年發泡刷月銷量同期比較'
}}
```

### 何時觸發此規則：
- 問題包含：「比較」、「對比」、「同期」、「趨勢」、「vs」、「相比」
- 且涉及：2 個以上年份 (如「2024 vs 2025」、「近三年」、「歷年」)
- 圖表類型：line（折線圖）、bar（長條圖）、area（面積圖）

## 🔍 空結果處理
- 當篩選結果為空 (len==0)：
  - answer 必須寫「🔍 經查詢，該條件下無數據記錄」
  - need_chart 設為 false
  - result_df 設為空 DataFrame 加上說明欄

## ⭐ 篩選防呆
- 問題含特定實體 → 第一步 target_df = df[condition]，再對 target_df 計算

{schema}

# 用戶: {query}
# 顏色: {rc or '未指定'}, 圖表: {rch or 'AI決定'}

# 📤 JSON 格式
{{
  "answer": "分析結論（若查無資料請明確告知）。請加入商業洞察，不要只給冷冰冰的數字描述。",
  "thinking": "1. 篩選條件... 2. 使用欄位... 3. 計算邏輯...",
  "need_chart": true/false,
  "chart_type": "{rch or 'bar'}",
  "chart_color": "{rc or ''}",
  "code": "完整可執行的 Python 程式碼"
}}

# 💻 程式碼規範
- df 已在環境中，result_df = DataFrame, chart_config = dict
- 禁止 import / matplotlib / plt
- chart_config 的 x,y,color 必須是**字串**！不能是列表！
- 年份用 _年份(整數), 品名用 str.contains() 搜 '{name_col}'
- 圓餅圖最多 Top 8，其餘合併「其他」
- 排序：年月 ascending=True，數量金額 ascending=False

## 範本：三類產品佔比（圓餅圖）
```python
f2025 = df[df['_年份'] == 2025].copy()
foam = f2025[f2025['{name_col}'].str.contains('發泡刷', case=False, na=False)]
dib_cer = f2025[(f2025['{code_col}'].str.startswith('DIB', na=False)) & (f2025['{name_col}'].str.contains('陶瓷刷', case=False, na=False))]
other_cer = f2025[(~f2025['{code_col}'].str.startswith('DIB', na=False)) & (f2025['{name_col}'].str.contains('陶瓷刷', case=False, na=False))]
qty_col = '{qty_signed or "數量"}'
result_df = pd.DataFrame({{
    '品名': ['發泡刷', 'DIB陶瓷刷', '其他陶瓷刷'],
    '數量': [foam[qty_col].sum(), dib_cer[qty_col].sum(), other_cer[qty_col].sum()]
}})
chart_config = {{'x': '品名', 'y': '數量', 'title': '2025年三大產品線淨銷量佔比'}}
```

## 範本：平均單價 (ASP)
```python
filtered = df[(df['_年份'] == 2025) & (df['{name_col}'].str.contains('發泡刷', case=False, na=False))].copy()
net_amt = filtered['{amt_signed or amt_untaxed or "未稅金額"}'].sum()
net_qty = filtered['{qty_signed or "數量"}'].sum()
asp = round(net_amt / net_qty, 2) if net_qty != 0 else 0
result_df = pd.DataFrame({{'指標': ['淨數量', '未稅淨額', '平均單價(ASP)'], '值': [net_qty, net_amt, asp]}})
chart_config = {{'title': '2025年發泡刷平均單價分析'}}
```

## 範本：查無資料處理
```python
filtered = df[(df['{cust_col}'].str.contains('華通', case=False, na=False)) & (df['_年份'] == 2024) & (df['{code_col}'].str.startswith('DIB', na=False))].copy()
if len(filtered) == 0:
    result_df = pd.DataFrame({{'說明': ['🔍 經查詢，華通 2024 年無購買 DIB 陶瓷刷紀錄']}})
    chart_config = {{}}
else:
    result_df = filtered.groupby('_月份')['{qty_signed or "數量"}'].sum().reset_index()
    result_df.columns = ['月份', '淨數量']
    chart_config = {{'x': '月份', 'y': '淨數量', 'title': '華通2024年DIB陶瓷刷月銷量'}}
```

# 檢查清單
1. x,y,color 是字串 2. _年份整數 3. str.contains() 搜 '{name_col}' 4. reset_index(drop=True)
5. ASP 用未稅淨額/淨數量 6. 無銷退減法 7. 圓餅Top8 8. 程式碼完整
9. ⚠️ 絕對不用 '{type_col}' 搜產品名 10. 查無資料要回報 11. 問圖表必給 chart_config"""

    def analyze(self, query, df, meta, audit, history):
        msgs = [{"role": "system", "content": self._sysprompt(df, meta, audit, query)}]
        for h in history[-3:]:
            msgs.append({"role": "user", "content": h.get('query', '')})
            if h.get('code'):
                msgs.append({"role": "assistant", "content": json.dumps(
                    {"answer": h.get('answer',''), "code": h.get('code','')[:500]}, ensure_ascii=False)})
        msgs.append({"role": "user", "content": query})
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=TEMPERATURE,
                response_format={"type": "json_object"}, max_tokens=4000)
            result = json.loads(r.choices[0].message.content)
            if self._forbidden(result.get('code', '')): return self._fallback()

            # ⭐ 圖表觸發機制：強制 need_chart
            chart_kw = ['圖', 'chart', '趨勢', '佔比', '分佈', '比例', '排名', 'top', 'pie', 'bar', 'line']
            if any(k in query.lower() for k in chart_kw):
                result['need_chart'] = True

            return result
        except Exception as e:
            return dict(answer=f"錯誤: {e}", thinking=traceback.format_exc(),
                        need_chart=False, chart_type="none", chart_color="", code="")

    def _forbidden(self, code):
        # 精確安全檢查：只擋真正危險的操作，不誤殺正常程式碼
        dangerous_imports = ['import os', 'import sys', 'import subprocess',
                             'import shutil', 'from os', 'from sys',
                             'from subprocess', 'from shutil']
        dangerous_calls = ['exec(', 'eval(', 'os.system', 'os.popen',
                           'subprocess.', 'shutil.rmtree', '__import__']
        dangerous_other = ['matplotlib', 'plt.show', 'plt.savefig']
        all_checks = dangerous_imports + dangerous_calls + dangerous_other
        return any(f in code for f in all_checks)

    def _fallback(self):
        return dict(answer="程式碼不安全，請重新描述。", thinking="安全檢查失敗",
                    need_chart=False, chart_type="none", chart_color="", code="")

    def execute_code(self, code, df):
        if not code or not code.strip(): return False, None, {}, "無程式碼"
        if self._forbidden(code): return False, None, {}, "禁止內容"
        g = {'pd': pd, 'np': np, 'df': df.copy(), 'result_df': None, 'chart_config': {}}
        try:
            exec(code, g)
            rdf = g.get('result_df')
            cc = self._fix_cc(g.get('chart_config', {}), rdf)
            if rdf is None:
                for k, v in g.items():
                    if isinstance(v, pd.DataFrame) and k != 'df' and len(v) > 0:
                        rdf = v; break
            if rdf is None: return False, None, {}, "無 result_df"
            if isinstance(rdf, pd.Series): rdf = rdf.reset_index()
            elif not isinstance(rdf, pd.DataFrame): rdf = pd.DataFrame({'結果': [rdf]})
            # ⭐ 允許空 result_df（查無資料情境），不再回傳失敗
            return True, rdf, cc, ""
        except Exception as e:
            return False, None, {}, f"執行錯誤: {e}\n{traceback.format_exc()}"

    def _fix_cc(self, cc, rdf):
        if not cc: return {}
        f = {}
        for k in ['x','y','color']:
            if k in cc:
                v = cc[k]
                f[k] = str(v[0]) if isinstance(v, list) and v else str(v) if v else ''
        for k in ['title','labels','text']:
            if k in cc: f[k] = cc[k]
        if rdf is not None and len(rdf.columns) >= 2:
            cols = list(rdf.columns)
            if 'x' not in f or not f.get('x'): f['x'] = str(cols[0])
            if 'y' not in f or not f.get('y'):
                for c in cols[1:]:
                    if pd.api.types.is_numeric_dtype(rdf[c]): f['y'] = str(c); break
                if 'y' not in f: f['y'] = str(cols[1])
        return f

    def fix_and_retry(self, code, error, query, df, meta, audit):
        nc = audit.get('product_name_cols', ['對方品名/品名備註'])
        cc = audit.get('product_code_cols', ['產品代號'])
        tc = audit.get('type_cols', ['單別名稱'])
        name_col = nc[0] if nc else '對方品名/品名備註'
        code_col = cc[0] if cc else '產品代號'
        type_col = tc[0] if tc else '單別名稱'
        qty_s = audit.get('qty_signed_col', '')
        amt_s = audit.get('amount_signed_col', '')
        amt_u = audit.get('amount_untaxed_col', '')

        prompt = f"""程式碼失敗，修復它。
原始: ```python\n{code}\n```
錯誤: {error}
問題: {query}

修復規則:
1. chart_config x,y,color 字串
2. 年份用 _年份(整數)
3. 品名搜 '{name_col}' + str.contains()
4. 代號搜 '{code_col}'
5. ❌ 不用 '{type_col}' 搜品名！它只有銷貨/銷退值
6. 淨數量用 '{qty_s or "數量"}'（已含正負號或後端已轉負）
7. 淨金額用 '{amt_s or amt_u or "未稅金額"}'
8. ASP = 未稅淨額/淨數量，禁用含稅
9. result_df = DataFrame + reset_index(drop=True)
10. 查無資料回報說明文字

返回完整 JSON。"""
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":self._sysprompt(df,meta,audit,query)},
                          {"role":"user","content":prompt}],
                temperature=TEMPERATURE, response_format={"type":"json_object"}, max_tokens=4000)
            return json.loads(r.choices[0].message.content)
        except: return None


# ╔══════════════════════════════════════════════════════════════╗
# ║  圖表生成器 v16.1 — 跨年份比較視覺修復                          ║
# ╚══════════════════════════════════════════════════════════════╝
class ChartGenerator:
    @staticmethod
    def create(data, chart_type, config, color_scheme=None):
        if data is None or len(data) == 0: return None
        x = safe_get_string(config.get('x'))
        y = safe_get_string(config.get('y'))
        color = safe_get_string(config.get('color'))
        title = safe_get_string(config.get('title'), '數據分析圖表')

        if not x or not y:
            cols = list(data.columns)
            if len(cols) >= 2:
                if not x: x = str(cols[0])
                if not y:
                    for c in cols[1:]:
                        if pd.api.types.is_numeric_dtype(data[c]): y = str(c); break
                    if not y and len(cols) > 1: y = str(cols[1])
        if not x or not y: return None
        if x not in data.columns or y not in data.columns: return None
        if color and color not in data.columns: color = ''

        data = data.copy()
        try:
            if not pd.api.types.is_numeric_dtype(data[y]):
                data[y] = pd.to_numeric(data[y], errors='coerce')
        except: pass
        
        # ⭐ v16.1 Visual Fix: 年份必須轉字串避免 Plotly 畫成漸層色
        try:
            if x in ['年份','_年份','year'] or '年' in str(x):
                if pd.api.types.is_numeric_dtype(data[x]):
                    data[x] = data[x].astype(int).astype(str)
            # 重要：color 欄位若為年份，也必須轉字串！
            if color and '年' in str(color) and pd.api.types.is_numeric_dtype(data[color]):
                data[color] = data[color].astype(int).astype(str)
        except: pass

        colors = COLOR_PALETTE.get(color_scheme, COLOR_PALETTE['default'])

        # 圓餅圖自動合併小項
        if chart_type in ('pie','donut') and len(data) > 10:
            data = ChartGenerator._merge_small(data, x, y, 8)

        try:
            fig = ChartGenerator._build(data, chart_type, x, y, color, title, colors)
            if fig: ChartGenerator._style(fig, title, len(data), chart_type)
            return fig
        except:
            try:
                fig = px.bar(data, x=x, y=y, title=title, color_discrete_sequence=colors)
                ChartGenerator._style(fig, title, len(data), 'bar')
                return fig
            except: return None

    @staticmethod
    def _merge_small(data, x, y, n=8):
        data = data.sort_values(y, ascending=False, key=abs).reset_index(drop=True)
        if len(data) <= n: return data
        top = data.head(n).copy()
        rest = pd.DataFrame({x: ['其他'], y: [data.iloc[n:][y].sum()]})
        return pd.concat([top, rest], ignore_index=True)

    @staticmethod
    def _build(data, ct, x, y, color, title, colors):
        kw = dict(title=title, color_discrete_sequence=colors)
        ckw = {}
        if color: ckw['color'] = color
        fig = None

        if ct == 'line':
            fig = px.line(data, x=x, y=y, markers=True, **kw, **ckw)
            fig.update_traces(line=dict(width=3), marker=dict(size=10))
        elif ct in ('area','stacked_area'):
            fig = px.area(data, x=x, y=y, **kw, **ckw)
        elif ct in ('stacked_bar','stacked'):
            fig = px.bar(data, x=x, y=y, barmode='stack', text=y, **kw, **ckw)
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside', textfont=dict(size=11, color='white'))
        elif ct == 'grouped_bar':
            fig = px.bar(data, x=x, y=y, barmode='group', text=y, **kw, **ckw)
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont=dict(size=11))
        elif ct == 'horizontal_bar':
            fig = px.bar(data, x=y, y=x, orientation='h', text=y, **kw, **ckw)
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont=dict(size=11))
        elif ct == 'pie':
            fig = px.pie(data, names=x, values=y, title=title, color_discrete_sequence=colors)
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=12))
        elif ct == 'donut':
            fig = px.pie(data, names=x, values=y, title=title, hole=0.45, color_discrete_sequence=colors)
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=12))
        elif ct == 'scatter':
            fig = px.scatter(data, x=x, y=y, **kw, **ckw)
        elif ct == 'waterfall':
            try:
                fig = go.Figure(go.Waterfall(x=data[x].tolist(), y=data[y].tolist(),
                    connector={"line":{"color":"rgb(63,63,63)"}}))
                fig.update_layout(title=title)
            except: fig = px.bar(data, x=x, y=y, **kw)
        elif ct == 'funnel':
            try: fig = px.funnel(data, x=y, y=x, **kw)
            except: fig = px.bar(data, x=y, y=x, orientation='h', **kw)
        elif ct == 'radar':
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=data[y].tolist(), theta=data[x].tolist(),
                    fill='toself', name=y, line_color=colors[0]))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=title)
            except: fig = px.bar(data, x=x, y=y, **kw)
        elif ct == 'heatmap':
            try:
                if color:
                    pv = data.pivot_table(values=y, index=x, columns=color, aggfunc='sum')
                    fig = px.imshow(pv, title=title, color_continuous_scale='RdYlBu_r')
                else: fig = px.bar(data, x=x, y=y, **kw)
            except: fig = px.bar(data, x=x, y=y, **kw)
        elif ct == 'treemap':
            try: fig = px.treemap(data, path=[x], values=y, **kw)
            except: fig = px.bar(data, x=x, y=y, **kw)
        elif ct == 'sunburst':
            try: fig = px.sunburst(data, path=[color, x] if color else [x], values=y, **kw)
            except: fig = px.pie(data, names=x, values=y, title=title, color_discrete_sequence=colors)
        else:  # default bar
            bkw = dict(text=y)
            if color: bkw['color'] = color; bkw['barmode'] = 'group'
            fig = px.bar(data, x=x, y=y, **kw, **bkw)
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont=dict(size=12))
        return fig

    @staticmethod
    def _style(fig, title, n, ct='bar'):
        circ = ct in ('pie','donut','radar','treemap','sunburst')
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, family='Microsoft JhengHei', color='#1a1a2e'),
                       x=0.5, xanchor='center', y=0.98, yanchor='top'),
            font=dict(size=13, family='Microsoft JhengHei', color='#2d3436'),
            height=580, hovermode='closest' if circ else 'x unified',
            plot_bgcolor='rgba(250,250,252,1)', paper_bgcolor='white',
            margin=dict(t=80, b=100, l=80, r=60),
        )
        if circ:
            fig.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                            font=dict(size=11), bgcolor='rgba(255,255,255,0.8)'),
                margin=dict(t=80, b=160, l=40, r=40),
            )
        else:
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5,
                            font=dict(size=12), bgcolor='rgba(255,255,255,0.8)'),
                yaxis=dict(tickformat=',', gridcolor='rgba(128,128,128,0.2)'),
                xaxis=dict(tickangle=-45 if n > 8 else 0, gridcolor='rgba(128,128,128,0.2)', type='category'),
            )
            # y軸留白讓 text 不被裁切
            if ct in ('bar','grouped_bar'):
                try:
                    vals = [v for trace in fig.data for v in (trace.y if hasattr(trace,'y') and trace.y is not None else []) if v is not None]
                    if vals: fig.update_yaxes(range=[min(0, min(vals)*1.1), max(vals)*1.25])
                except: pass


# ============================================================================
# 🖥️ Streamlit 主程式
# ============================================================================
def show_login_page():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🔐 系統登入</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">企業級 AI 智能數據分析平台 v16.1</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("---")
        # 使用 form 讓密碼欄位支援 Enter 鍵登入
        with st.form(key="login_form", clear_on_submit=False):
            pwd = st.text_input("請輸入密碼", type="password", key="login_password")
            submitted = st.form_submit_button("🚀 登入", type="primary", use_container_width=True)
            
            if submitted:
                if pwd == PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ 密碼錯誤")


def show_main_app():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    # 初始化 session state（API key 使用內建值）
    defs = dict(df=None, metadata=None, history=[], ai_engine=None,
                debug_mode=False, cleaning_stats={}, cleaning_log=[], semantic_audit=None,
                available_sheets={}, selected_sheets={})
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # 自動初始化 AI 引擎（使用內建 API Key）
    if st.session_state.ai_engine is None:
        st.session_state.ai_engine = LogicalBrainEngine(EMBEDDED_API_KEY)

    st.markdown('<h1 class="main-header">🤖 AI 智能數據分析師</h1>', unsafe_allow_html=True)

    # ── 側邊欄（已移除 API Key 輸入）──
    with st.sidebar:
        st.header("⚙️ 系統設定")
        st.session_state.debug_mode = st.checkbox("🐛 除錯模式", value=st.session_state.debug_mode)

        st.divider()
        st.header("📁 上傳資料")
        files = st.file_uploader("選擇 Excel 檔案", type=['xlsx','xls'], accept_multiple_files=True)

        # ── 工作表選擇器（修復版）──
        if files:
            st.markdown("### 📋 工作表選擇")
            temp_avail = {}
            for f in files:
                try:
                    fb = f.read(); f.seek(0)
                    xls = pd.ExcelFile(io.BytesIO(fb))
                    valid = [s for s in xls.sheet_names if not any(k in s.lower() for k in DataCleaningEngine.SKIP_SHEET_KW)]
                    temp_avail[f.name] = valid if valid else xls.sheet_names
                except: continue
            st.session_state.available_sheets = temp_avail

            if temp_avail:
                show_sel = st.checkbox("🔍 手動選擇工作表", value=False,
                    help="預設載入所有工作表（智慧路由會自動丟棄總表）")

                if show_sel:
                    for fname, sheets in temp_avail.items():
                        st.markdown(f"**📄 {fname}**")
                        # 用獨立 session_state key 管理每個檔案的選擇
                        sk = f"_sheetsel_{fname}"
                        if sk not in st.session_state:
                            st.session_state[sk] = sheets.copy()

                        selected = st.multiselect("選擇工作表", options=sheets,
                            default=st.session_state[sk], key=f"ms_{fname}", label_visibility="collapsed")
                        st.session_state[sk] = selected
                        st.session_state.selected_sheets[fname] = selected
                        st.caption(f"已選 {len(selected)}/{len(sheets)}")
                        st.markdown("---")
                else:
                    st.session_state.selected_sheets = {}
                    total = sum(len(s) for s in temp_avail.values())
                    st.info(f"💡 預設全部載入 ({total} 個工作表)，智慧路由自動處理總表")

        # 載入按鈕
        if files:
            if st.button("🚀 載入並清洗資料", type="primary", use_container_width=True, key="sidebar_load_data"):
                with st.spinner("🔄 三大引擎啟動中..."):
                    try:
                        sel = st.session_state.selected_sheets if st.session_state.selected_sheets else None
                        cleaner = DataCleaningEngine()
                        df, meta, stats = cleaner.clean(files, sel)
                        if df is not None and len(df) > 0:
                            st.session_state.df = df
                            st.session_state.metadata = meta
                            st.session_state.cleaning_stats = stats
                            st.session_state.cleaning_log = cleaner.log
                            st.session_state.history = []
                            detector = SemanticDetectionModule()
                            st.session_state.semantic_audit = detector.audit(df, meta)
                            st.success(f"✅ 載入 {len(df):,} 筆（銷退已轉負，總表已過濾）")
                        else:
                            st.error("❌ 無有效資料")
                    except Exception as e:
                        st.error(f"❌ 載入失敗: {e}")

        # 引擎狀態
        if st.session_state.df is not None:
            stats = st.session_state.cleaning_stats
            st.divider()
            st.markdown("### 🏗️ 引擎狀態")
            st.markdown(f"""<div class="engine-report">
<b>🧹 數據清洗引擎</b><br>
檔案: {stats.get('total_files',0)} | 工作表: {stats.get('total_sheets_read',0)} |
丟棄總表: {stats.get('sheets_skipped_summary',0)}<br>
去重: {stats.get('dedup_strategy','-')} | 移除: {stats.get('duplicates_removed',0):,}<br>
<b>🔄 銷退轉負: {stats.get('return_rows_negated',0):,} 筆</b> (欄位: {stats.get('return_type_col','-')})<br>
數值: {stats.get('numeric_cols_standardized',0)} 欄 | 日期: {stats.get('date_cols_processed',0)} 欄
</div>""", unsafe_allow_html=True)

            audit = st.session_state.semantic_audit
            if audit:
                st.markdown(f"""<div class="engine-report">
<b>🎯 語意感知</b><br>{audit.get('summary_text','').replace(chr(10), '<br>')}
</div>""", unsafe_allow_html=True)

            with st.expander("📋 清洗日誌", expanded=False):
                for l in st.session_state.cleaning_log:
                    st.markdown(f"- {l}")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ 清除資料", key="sidebar_clear_data"):
                    for k in ['df','metadata','history','cleaning_stats','cleaning_log','semantic_audit']:
                        st.session_state[k] = defs.get(k)
                    st.rerun()
            with c2:
                if st.button("🔄 清除對話", key="sidebar_clear_history"):
                    st.session_state.history = []
                    st.rerun()

            with st.expander("👀 資料預覽"):
                pcols = [c for c in st.session_state.df.columns if not str(c).startswith('_')]
                st.dataframe(st.session_state.df[pcols].head(10), use_container_width=True)

    # ══════════════════════════════════════════════
    # 主畫面
    # ══════════════════════════════════════════════
    if st.session_state.df is None:
        st.info("👈 請先在左側上傳 Excel 檔案")
        return

    df = st.session_state.df
    meta = st.session_state.metadata
    audit = st.session_state.semantic_audit or {}
    ai = st.session_state.ai_engine
    debug = st.session_state.debug_mode

    # 顯示歷史
    for i, item in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(item.get('query', ''))
        with st.chat_message("assistant"):
            if item.get('answer'): st.markdown(item['answer'])
            if item.get('thinking'):
                with st.expander("🧠 分析思路", expanded=False):
                    st.markdown(f'<div class="thinking-box">{item["thinking"]}</div>', unsafe_allow_html=True)
            rdf = item.get('result_df')
            if item.get('need_chart') and rdf is not None and len(rdf) > 0:
                try:
                    fig = ChartGenerator.create(rdf, item.get('chart_type','bar'),
                        item.get('chart_config',{}), item.get('chart_color',''))
                    if fig: st.plotly_chart(fig, use_container_width=True, key=f"h_{i}_{hash(str(item.get('query','')))}")
                except: pass
            if rdf is not None and len(rdf) > 0:
                st.markdown(f'<div class="data-header">📋 查詢結果 ({len(rdf):,} 筆)</div>', unsafe_allow_html=True)
                ddf = rdf.copy(); ddf.index = range(1, len(ddf)+1)
                for col in ddf.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(ddf[col]): ddf[col] = ddf[col].apply(format_number)
                    except: pass
                st.dataframe(ddf, use_container_width=True, height=min(400, len(ddf)*35+50))
            if debug and item.get('code'):
                with st.expander("💻 程式碼"): st.code(item['code'], language='python')

    # 輸入
    query = st.chat_input("請輸入問題...")
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("🧠 AI 分析中..."):
                ai_result = ai.analyze(query, df, meta, audit, st.session_state.history)
                answer = ai_result.get('answer', '')
                thinking = ai_result.get('thinking', '')
                need_chart = ai_result.get('need_chart', False)
                chart_type = ai_result.get('chart_type', 'bar')
                chart_color = ai_result.get('chart_color', '')
                code = ai_result.get('code', '')
                result_df = None
                chart_config = {}

                if code:
                    success, result_df, chart_config, error = ai.execute_code(code, df)
                    if not success:
                        for retry in range(MAX_RETRIES):
                            if debug: st.warning(f"⚠️ 第 {retry+1} 次修復...")
                            fixed = ai.fix_and_retry(code, error, query, df, meta, audit)
                            if fixed and fixed.get('code'):
                                success, result_df, chart_config, nerr = ai.execute_code(fixed['code'], df)
                                if success:
                                    code = fixed['code']
                                    answer = fixed.get('answer', answer)
                                    thinking = fixed.get('thinking', thinking)
                                    if debug: st.success("✅ 修復成功!")
                                    break
                                error = nerr
                        if not success and debug: st.error(f"❌ 失敗: {error}")

                st.session_state.history.append(dict(
                    query=query, answer=answer, thinking=thinking,
                    need_chart=need_chart, chart_type=chart_type, chart_color=chart_color,
                    code=code, result_df=result_df, chart_config=chart_config))

            if answer: st.markdown(answer)
            if thinking:
                with st.expander("🧠 分析思路", expanded=False):
                    st.markdown(f'<div class="thinking-box">{thinking}</div>', unsafe_allow_html=True)
            if need_chart and result_df is not None and len(result_df) > 0:
                try:
                    fig = ChartGenerator.create(result_df, chart_type, chart_config, chart_color)
                    if fig: st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    if debug: st.error(f"圖表錯誤: {e}")
            if result_df is not None and len(result_df) > 0:
                st.markdown(f'<div class="data-header">📋 查詢結果 ({len(result_df):,} 筆)</div>', unsafe_allow_html=True)
                ddf = result_df.copy(); ddf.index = range(1, len(ddf)+1)
                for col in ddf.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(ddf[col]): ddf[col] = ddf[col].apply(format_number)
                    except: pass
                st.dataframe(ddf, use_container_width=True, height=min(450, len(ddf)*35+50))
            elif code and (result_df is None or len(result_df) == 0):
                st.warning("⚠️ 查詢沒有結果，請檢查條件")
            if debug and code:
                with st.expander("💻 程式碼"): st.code(code, language='python')
                if chart_config:
                    with st.expander("📊 chart_config"): st.json(chart_config)


def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="expanded")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()