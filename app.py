
# app.py
# -*- coding: utf-8 -*-
import io
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import unicodedata

APP_VERSION = "v4.1 (faturamento e cupons somados por loja)"

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

def to_float_series(s: pd.Series):
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r"[^0-9,.\-]", "", regex=True)
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False),
        errors="coerce"
    )

def compute_indicators_v41(df: pd.DataFrame, cols: dict):
    vcol = cols["value"]
    qcol = cols["qty"]
    store_col = cols["store_code"]
    seller_col = cols.get("seller_code")

    df[vcol] = to_float_series(df[vcol]).fillna(0.0)
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce").fillna(0).astype(float)

    fat_loja = df.groupby(store_col, dropna=False)[vcol].sum().reset_index()
    fat_loja.columns = ["codigo_loja", "faturamento"]

    cupons_loja = df.groupby(store_col, dropna=False)[qcol].sum().reset_index()
    cupons_loja.columns = ["codigo_loja", "cupons"]

    lojas = fat_loja.merge(cupons_loja, on="codigo_loja", how="outer").fillna(0)
    lojas["ticket_medio_loja"] = np.where(lojas["cupons"] > 0, lojas["faturamento"] / lojas["cupons"], np.nan)

    faturamento_rede = float(lojas["faturamento"].sum())
    total_cupons = float(lojas["cupons"].sum())

    if seller_col and seller_col in df.columns:
        vendedores = df.groupby(seller_col, dropna=False)[vcol].sum().reset_index()
        vendedores.columns = ["codigo_vendedor", "faturamento"]
    else:
        vendedores = pd.DataFrame(columns=["codigo_vendedor", "faturamento"])

    resumo = pd.DataFrame({
        "Indicador": [
            "Faturamento da rede (R$)",
            "Total de cupons (quantidade vendida)",
            "Quantidade de lojas ativas",
            "Quantidade de vendedores (com vendas)"
        ],
        "Valor": [
            faturamento_rede,
            total_cupons,
            int(lojas["codigo_loja"].nunique()),
            int(vendedores["codigo_vendedor"].nunique()) if not vendedores.empty else 0
        ]
    })

    return resumo, lojas, vendedores, faturamento_rede, total_cupons

def build_excel_v41(df, resumo, lojas, vendedores):
    with pd.ExcelWriter(io.BytesIO(), engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Base", index=False)
        resumo.to_excel(writer, sheet_name="Resumo", index=False)
        lojas.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="Lojas", index=False)
        vendedores.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="Vendedores", index=False)

        wb = writer.book
        ws_resumo = writer.sheets["Resumo"]
        ws_lojas = writer.sheets["Lojas"]
        ws_vend = writer.sheets["Vendedores"]

        fmt_money = wb.add_format({'num_format': 'R$ #,##0.00'})
        fmt_int = wb.add_format({'num_format': '0'})

        ws_resumo.set_column("A:A", 40)
        ws_resumo.set_column("B:B", 22, fmt_money)
        for r in range(0, len(resumo)):
            if "cupons" in str(resumo.iloc[r, 0]).lower():
                ws_resumo.write_number(r + 1, 1, float(resumo.iloc[r, 1]), fmt_int)

        ws_lojas.set_column("A:A", 18)
        ws_lojas.set_column("B:B", 18, fmt_money)
        ws_lojas.set_column("C:C", 12, fmt_int)
        ws_lojas.set_column("D:D", 20, fmt_money)

        ws_vend.set_column("A:A", 22)
        ws_vend.set_column("B:B", 18, fmt_money)

        top10 = lojas.sort_values("faturamento", ascending=False).head(10)
        chart = wb.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'Faturamento (R$)',
            'categories': f"='Lojas'!$A$2:$A${len(top10) + 1}",
            'values':     f"='Lojas'!$B$2:$B${len(top10) + 1}",
            'data_labels': {'value': True, 'num_format': 'R$ #,##0.00'}
        })
        chart.set_title({'name': 'Top 10 Lojas por Faturamento'})
        chart.set_x_axis({'name': 'Código da Loja'})
        chart.set_y_axis({'name': 'R$'})
        ws_resumo.insert_chart('D2', chart, {'x_scale': 1.2, 'y_scale': 1.2})

        return writer.book.filename.getvalue()

st.set_page_config(page_title="Indicadores Drogaria – v4.1", layout="wide")
st.title("📈 Indicadores de Vendas – Rede de Drogaria")
st.caption("Versão " + APP_VERSION + " – Ajustes: faturamento e cupons somados por loja.")

uploaded = st.file_uploader("Envie seu arquivo (.csv, .xlsx)", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

if uploaded:
    if uploaded.name.lower().endswith(".csv"):
        sample = uploaded.getvalue().decode("utf-8", errors="ignore")[:5000]
        sep = detect_delimiter(sample, default=",")
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=sep, encoding="utf-8", low_memory=False)
    else:
        xls = pd.ExcelFile(uploaded)
        aba = st.selectbox("Selecione a planilha (aba)", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=aba)

    display_cols = list(df.columns)
    normalized = normalize_columns(display_cols)
    norm_to_display = {n: d for n, d in zip(normalized, display_cols)}

    def sel(label, key):
        options = ["(vazio)"] + normalized
        return st.selectbox(label, options, index=0, key=key)

    store_code = sel("Coluna de **Código da Loja**", "store_code")
    seller_code = st.selectbox("Coluna de **Código do Vendedor** (opcional)", ["(vazio)"] + normalized, index=0, key="seller_code")
    value_col = sel("Coluna de **Valor Total Venda (R$)**", "value_col")
    qty_col = sel("Coluna de **Quantidade Vendida**", "qty_col")

    proceed = st.button("Gerar Indicadores (v4.1)")

    if proceed:
        mapped = {
            "store_code": norm_to_display.get(store_code) if store_code != "(vazio)" else None,
            "seller_code": norm_to_display.get(seller_code) if seller_code != "(vazio)" else None,
            "value": norm_to_display.get(value_col) if value_col != "(vazio)" else None,
            "qty": norm_to_display.get(qty_col) if qty_col != "(vazio)" else None,
        }
        missing = [k for k in ["store_code", "value", "qty"] if mapped.get(k) is None]
        if missing:
            st.error("Mapeie as colunas obrigatórias.")
        else:
            resumo, lojas, vendedores, fat_rede, cupons_total = compute_indicators_v41(df.copy(), mapped)
            st.subheader("📌 Resumo do Período (Regras v4.1)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faturamento da Rede", f"R$ {fat_rede:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            col2.metric("Total de Cupons", f"{int(cupons_total):,}".replace(",", "."))
            col3.metric("Lojas Ativas", f"{int(lojas['codigo_loja'].nunique())}")
            col4.metric("Vendedores com Vendas", f"{int(vendedores['codigo_vendedor'].nunique()) if not vendedores.empty else 0}")

            st.dataframe(lojas.sort_values("faturamento", ascending=False))
            st.dataframe(vendedores.sort_values("faturamento", ascending=False))

            excel_bytes = build_excel_v41(df.copy(), resumo, lojas, vendedores)
            st.download_button(
                label="Baixar Excel (v4.1)",
                data=excel_bytes,
                file_name=f"Indicadores_Drogaria_v4_1_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
