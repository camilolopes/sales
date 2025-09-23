# app.py
# -*- coding: utf-8 -*-
import io
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import unicodedata

APP_VERSION = "v4.0 (regras de negócio atualizadas)"

# ==============================
# Utilitários
# ==============================
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
         .str.replace(r"[^0-9,.\-]", "", regex=True)  # mantém dígitos, ., , e -
         .str.replace(".", "", regex=False)           # remove separador de milhar
         .str.replace(",", ".", regex=False),         # converte vírgula em ponto
        errors="coerce"
    )

# ==============================
# Cálculos (REGRAS NOVAS)
# ==============================
def compute_indicators_v4(df: pd.DataFrame, cols: dict):
    """
    Regras solicitadas:
    1) Faturamento da rede = soma da coluna 'valor total venda' no período (sem filtro).
    2) Faturamento da loja = soma de 'valor total venda' agrupando por 'codigo da loja'.
    3) Cupons por loja = soma da 'quantidade vendida' agrupando por 'codigo da loja'.
    4) Faturamento do vendedor = soma de 'valor total venda' agrupando por 'codigo do vendedor'.
    5) Total de cupons (rede) = soma de 'quantidade vendida' (sem agrupamento).
    """
    vcol = cols["value"]
    qcol = cols["qty"]
    store_col = cols["store_code"]
    seller_col = cols.get("seller_code")

    # normalizar tipos
    df[vcol] = to_float_series(df[vcol]).fillna(0.0)
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce").fillna(0).astype(float)

    # 1) Faturamento da rede
    faturamento_rede = float(df[vcol].sum())

    # 2) Faturamento por loja
    fat_loja = df.groupby(store_col, dropna=False)[vcol].sum().reset_index()
    fat_loja.columns = ["codigo_loja", "faturamento"]

    # 3) Cupons por loja (soma da quantidade vendida)
    cupons_loja = df.groupby(store_col, dropna=False)[qcol].sum().reset_index()
    cupons_loja.columns = ["codigo_loja", "cupons"]

    lojas = fat_loja.merge(cupons_loja, on="codigo_loja", how="outer").fillna(0)
    # ticket médio por loja (opcional): faturamento / cupons
    lojas["ticket_medio_loja"] = np.where(lojas["cupons"]>0, lojas["faturamento"]/lojas["cupons"], np.nan)

    # 4) Faturamento por vendedor (agrupado por codigo do vendedor)
    if seller_col and seller_col in df.columns:
        vendedores = df.groupby(seller_col, dropna=False)[vcol].sum().reset_index()
        vendedores.columns = ["codigo_vendedor", "faturamento"]
    else:
        vendedores = pd.DataFrame(columns=["codigo_vendedor", "faturamento"])

    # 5) Total de cupons (rede) = soma da 'quantidade vendida'
    total_cupons = float(df[qcol].sum())

    # KPIs gerais
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

# ==============================
# Excel com dashboard
# ==============================
def build_excel_v4(df, resumo, lojas, vendedores):
    import io
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
        fmt_header = wb.add_format({'bold': True, 'bg_color': '#EEEEEE', 'border': 1})

        # formatação
        ws_resumo.set_column("A:A", 40)
        ws_resumo.set_column("B:B", 22, fmt_money)
        for r in range(0, len(resumo)):
            if "cupons" in str(resumo.iloc[r,0]).lower():
                ws_resumo.write_number(r+1, 1, float(resumo.iloc[r,1]), fmt_int)

        ws_lojas.set_column("A:A", 18)
        ws_lojas.set_column("B:B", 18, fmt_money)
        ws_lojas.set_column("C:C", 12, fmt_int)
        ws_lojas.set_column("D:D", 20, fmt_money)

        ws_vend.set_column("A:A", 22)
        ws_vend.set_column("B:B", 18, fmt_money)

        # Gráfico Top 10 Lojas por faturamento
        top10 = lojas.sort_values("faturamento", ascending=False).head(10)
        chart = wb.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'Faturamento (R$)',
            'categories': f"='Lojas'!$A$2:$A${len(top10)+1}",
            'values':     f"='Lojas'!$B$2:$B${len(top10)+1}",
            'data_labels': {'value': True, 'num_format': 'R$ #,##0.00'}
        })
        chart.set_title({'name': 'Top 10 Lojas por Faturamento'})
        chart.set_x_axis({'name': 'Código da Loja'})
        chart.set_y_axis({'name': 'R$'})
        ws_resumo.insert_chart('D2', chart, {'x_scale': 1.2, 'y_scale': 1.2})

        return writer.book.filename.getvalue()

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Indicadores Drogaria – v4.0", layout="wide")
st.title("📈 Indicadores de Vendas – Rede de Drogaria")
st.caption("Versão " + APP_VERSION + " – Regras de cálculo atualizadas conforme solicitação.")

uploaded = st.file_uploader("Envie seu arquivo (.csv, .xlsx)", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

if uploaded:
    # leitura
    if uploaded.name.lower().endswith(".csv"):
        sample = uploaded.getvalue().decode("utf-8", errors="ignore")[:5000]
        sep = detect_delimiter(sample, default=",")
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=sep, encoding="utf-8", low_memory=False)
    else:
        xls = pd.ExcelFile(uploaded)
        aba = st.selectbox("Selecione a planilha (aba)", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=aba)

    # mapeamento
    st.subheader("🔧 Mapeamento de colunas (obrigatório)")
    st.caption("Selecione as colunas equivalentes no seu arquivo.")

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

    proceed = st.button("Gerar Indicadores (v4.0)")

    if proceed:
        # validação de obrigatórios
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
            try:
                resumo, lojas, vendedores, fat_rede, cupons_total = compute_indicators_v4(df.copy(), mapped)
            except Exception as e:
                st.error(f"Erro ao calcular indicadores: {e}")
            else:
                # Resumo
                st.subheader("📌 Resumo do Período (Regras novas)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Faturamento da Rede", f"R$ {fat_rede:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                col2.metric("Total de Cupons (Qtd. Vendida)", f"{int(cupons_total):,}".replace(",", "."))
                col3.metric("Lojas Ativas", f"{int(lojas['codigo_loja'].nunique())}")
                col4.metric("Vendedores com Vendas", f"{int(vendedores['codigo_vendedor'].nunique()) if not vendedores.empty else 0}")

                # Tabelas
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🏬 Faturamento por Loja")
                    st.dataframe(lojas.sort_values("faturamento", ascending=False), use_container_width=True)
                with c2:
                    st.markdown("#### 👤 Faturamento por Vendedor")
                    st.dataframe(vendedores.sort_values("faturamento", ascending=False), use_container_width=True)

                # Download Excel
                st.divider()
                st.markdown("### ⬇️ Download do Excel")
                excel_bytes = build_excel_v4(df.copy(), resumo, lojas, vendedores)
                st.download_button(
                    label="Baixar Excel (v4.0)",
                    data=excel_bytes,
                    file_name=f"Indicadores_Drogaria_v4_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Validação específica do arquivo citado
                st.divider()
                st.markdown("### ✅ Validação solicitada (apenas para o arquivo do período informado)")
                esperado_cupons = 283713
                esperado_fat = 6208840.60
                match_cupons = int(round(cupons_total)) == esperado_cupons
                match_fat = round(fat_rede, 2) == round(esperado_fat, 2)

                colv1, colv2 = st.columns(2)
                colv1.metric("Esperado – Total de Cupons", f"{esperado_cupons:,}".replace(",", "."))
                colv1.metric("Calculado – Total de Cupons", f"{int(cupons_total):,}".replace(",", "."), delta="OK" if match_cupons else "Difere")

                colv2.metric("Esperado – Faturamento Rede", f"R$ {esperado_fat:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                colv2.metric("Calculado – Faturamento Rede", f"R$ {fat_rede:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), delta="OK" if match_fat else "Difere")

                if match_cupons and match_fat:
                    st.success("Validação OK: os totais calculados batem com os valores esperados.")
                else:
                    st.warning("Validação não bateu. Confira o mapeamento das colunas e o período do arquivo enviado.")
