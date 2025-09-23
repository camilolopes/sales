
# app.py
# -*- coding: utf-8 -*-
import io
import re
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import unicodedata

APP_VERSION = "v4.3.1 (fix Excel zerobyte | soma linha a linha + parser monetário robusto)"

@st.cache_data(show_spinner=False)
def detect_delimiter(sample_text: str, default=","):
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;|\t")
        return dialect.delimiter
    except Exception:
        return default

def unaccent(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("utf-8")

def normalize_columns(cols):
    out = []
    for c in cols:
        c = unaccent(str(c)).lower().strip()
        for ch in ["-", "_", "/", "\\", "  "]:
            c = c.replace(ch, " ")
        c = " ".join(c.split())
        out.append(c)
    return out

def _parse_money_cell(cell: str):
    import re, numpy as np
    if cell is None:
        return np.nan
    s = str(cell).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    s = re.sub(r"[^\d,.\-]", "", s)
    last_comma = s.rfind(","); last_dot = s.rfind(".")
    has_comma = last_comma != -1; has_dot = last_dot != -1
    decimal_sep = None; thousands_sep = None
    if has_comma and has_dot:
        decimal_sep = "," if last_comma > last_dot else "."
        thousands_sep = "." if decimal_sep == "," else ","
    elif has_comma:
        decimals = len(s) - last_comma - 1
        if 1 <= decimals <= 2: decimal_sep = ","
        else: thousands_sep = ","
    elif has_dot:
        decimals = len(s) - last_dot - 1
        if 1 <= decimals <= 2: decimal_sep = "."
        else: thousands_sep = "."
    try:
        if thousands_sep: s = s.replace(thousands_sep, "")
        if decimal_sep and decimal_sep != ".": s = s.replace(decimal_sep, ".")
        if not decimal_sep: s = s.replace(",", "").replace(".", "")
        return float(s)
    except Exception:
        try:
            s2 = re.sub(r"[^\d\-]", "", s)
            return float(s2)
        except Exception:
            return np.nan

def to_float_series_robust(series: pd.Series) -> pd.Series:
    return series.map(_parse_money_cell).astype(float)

def compute_indicators_v43(df: pd.DataFrame, cols: dict):
    vcol = cols["value"]; qcol = cols["qty"]
    store_col = cols["store_code"]; seller_col = cols.get("seller_code")
    df[vcol] = to_float_series_robust(df[vcol]).fillna(0.0)
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce").fillna(0).astype(float)
    faturamento_rede = float(df[vcol].sum())
    total_cupons = float(df[qcol].sum())
    fat_loja = df.groupby(store_col, dropna=False)[vcol].sum().reset_index(); fat_loja.columns = ["codigo_loja", "faturamento"]
    cupons_loja = df.groupby(store_col, dropna=False)[qcol].sum().reset_index(); cupons_loja.columns = ["codigo_loja", "cupons"]
    lojas = fat_loja.merge(cupons_loja, on="codigo_loja", how="outer").fillna(0)
    lojas["ticket_medio_loja"] = np.where(lojas["cupons"]>0, lojas["faturamento"]/lojas["cupons"], np.nan)
    if seller_col and seller_col in df.columns:
        vendedores = df.groupby(seller_col, dropna=False)[vcol].sum().reset_index(); vendedores.columns = ["codigo_vendedor", "faturamento"]
    else:
        vendedores = pd.DataFrame(columns=["codigo_vendedor", "faturamento"])
    resumo = pd.DataFrame({"Indicador": ["Faturamento da rede (R$)","Total de cupons (quantidade vendida)","Quantidade de lojas ativas","Quantidade de vendedores (com vendas)"],
                           "Valor": [faturamento_rede, total_cupons, int(lojas["codigo_loja"].nunique()), int(vendedores["codigo_vendedor"].nunique()) if not vendedores.empty else 0]})
    return resumo, lojas, vendedores, faturamento_rede, total_cupons

def build_excel_v43(df, resumo, lojas, vendedores):
    # ✅ usar buffer BytesIO externo e retornar buffer.getvalue()
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Base", index=False)
        resumo.to_excel(writer, sheet_name="Resumo", index=False)
        lojas.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="Lojas", index=False)
        vendedores.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="Vendedores", index=False)

        wb = writer.book
        ws_resumo = writer.sheets["Resumo"]; ws_lojas = writer.sheets["Lojas"]; ws_vend = writer.sheets["Vendedores"]
        fmt_money = wb.add_format({'num_format': 'R$ #,##0.00'}); fmt_int = wb.add_format({'num_format': '0'})

        ws_resumo.set_column("A:A", 40); ws_resumo.set_column("B:B", 22, fmt_money)
        for r in range(0, len(resumo)):
            if "cupons" in str(resumo.iloc[r, 0]).lower():
                ws_resumo.write_number(r + 1, 1, float(resumo.iloc[r, 1]), fmt_int)

        ws_lojas.set_column("A:A", 18); ws_lojas.set_column("B:B", 18, fmt_money); ws_lojas.set_column("C:C", 12, fmt_int); ws_lojas.set_column("D:D", 20, fmt_money)
        ws_vend.set_column("A:A", 22); ws_vend.set_column("B:B", 18, fmt_money)

        top10 = lojas.sort_values("faturamento", ascending=False).head(10)
        chart = wb.add_chart({'type': 'column'})
        chart.add_series({'name': 'Faturamento (R$)',
                          'categories': f"='Lojas'!$A$2:$A${len(top10) + 1}",
                          'values':     f"='Lojas'!$B$2:$B${len(top10) + 1}",
                          'data_labels': {'value': True, 'num_format': 'R$ #,##0.00'}})
        chart.set_title({'name': 'Top 10 Lojas por Faturamento'}); chart.set_x_axis({'name': 'Código da Loja'}); chart.set_y_axis({'name': 'R$'})
        ws_resumo.insert_chart('D2', chart, {'x_scale': 1.2, 'y_scale': 1.2})

    # muito importante: mover o cursor para o início antes de ler
    buffer.seek(0)
    return buffer.getvalue()

st.set_page_config(page_title="Indicadores Drogaria – v4.3.1 (produção)", layout="wide")
st.title("📈 Indicadores de Vendas – Rede de Drogaria")
st.caption("Versão " + APP_VERSION + " – Fix do Excel zerobyte: gravação correta em BytesIO.")

uploaded = st.file_uploader("Envie seu arquivo (.csv, .xlsx)", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
if uploaded:
    if uploaded.name.lower().endswith(".csv"):
        sample = uploaded.getvalue().decode("utf-8", errors="ignore")[:5000]
        sep = detect_delimiter(sample, default=",")
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=sep, encoding="utf-8", low_memory=False)
    else:
        xls = pd.ExcelFile(uploaded); aba = st.selectbox("Selecione a planilha (aba)", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=aba)

    display_cols = list(df.columns); normalized = normalize_columns(display_cols)
    norm_to_display = {n: d for n, d in zip(normalized, display_cols)}

    def sel(label, key): return st.selectbox(label, ["(vazio)"] + normalized, index=0, key=key)

    store_code = sel("Coluna de **Código da Loja**", "store_code")
    seller_code = st.selectbox("Coluna de **Código do Vendedor** (opcional)", ["(vazio)"] + normalized, index=0, key="seller_code")
    value_col = sel("Coluna de **Valor Total Venda (R$)**", "value_col")
    qty_col = sel("Coluna de **Quantidade Vendida**", "qty_col")

    if st.button("Gerar Indicadores (v4.3.1)"):
        mapped = {
            "store_code": norm_to_display.get(store_code) if store_code != "(vazio)" else None,
            "seller_code": norm_to_display.get(seller_code) if seller_code != "(vazio)" else None,
            "value": norm_to_display.get(value_col) if value_col != "(vazio)" else None,
            "qty": norm_to_display.get(qty_col) if qty_col != "(vazio)" else None,
        }
        missing = [k for k in ["store_code","value","qty"] if mapped.get(k) is None]
        if missing:
            st.error("Mapeie as colunas obrigatórias: **Código da Loja**, **Valor Total Venda** e **Quantidade Vendida**.")
        else:
            resumo, lojas, vendedores, fat_rede, cupons_total = compute_indicators_v43(df.copy(), mapped)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faturamento da Rede", f"R$ {fat_rede:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            col2.metric("Total de Cupons", f"{int(cupons_total):,}".replace(",", "."))
            col3.metric("Lojas Ativas", f"{int(lojas['codigo_loja'].nunique())}")
            col4.metric("Vendedores com Vendas", f"{int(vendedores['codigo_vendedor'].nunique()) if not vendedores.empty else 0}")

            st.markdown("#### 🏬 Faturamento por Loja")
            st.dataframe(lojas.sort_values("faturamento", ascending=False), use_container_width=True)
            st.markdown("#### 👤 Faturamento por Vendedor")
            st.dataframe(vendedores.sort_values("faturamento", ascending=False), use_container_width=True)

            excel_bytes = build_excel_v43(df.copy(), resumo, lojas, vendedores)
            st.download_button(
                label="Baixar Excel (v4.3.1)",
                data=excel_bytes,
                file_name=f"Indicadores_Drogaria_v4_3_1_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
