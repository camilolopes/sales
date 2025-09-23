
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

APP_VERSION = "v4.4 (produção | Excel aprimorado: vendedores/nome, inteiros, departamentos %, vendas diárias)"

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

def _parse_money_cell(cell: str):
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

def parse_date_series(series: pd.Series) -> pd.Series:
    # tenta converter várias formas de data
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.date

# ==============================
# Cálculos (REGRAS v4.4)
# ==============================
def compute_indicators_v44(df: pd.DataFrame, cols: dict):
    """
    Linhas representam cupons consolidados.
    - Faturamento da rede = soma direta (linha a linha) de 'valor total venda' (parser robusto).
    - Total de cupons (rede) = soma direta (linha a linha) de 'quantidade vendida'.
    - Faturamento/Cupons por LOJA = somas por agrupamento.
    - Faturamento por VENDEDOR = soma por agrupamento; inclui nome do vendedor se mapeado.
    - Departamentos: participação (%) por código de departamento; traz nome se mapeado.
    - Vendas diárias: soma por dia (se data mapeada).
    """
    vcol = cols["value"]
    qcol = cols["qty"]
    store_col = cols["store_code"]
    seller_code = cols.get("seller_code")
    seller_name = cols.get("seller_name")
    dept_code = cols.get("dept_code")
    dept_name = cols.get("dept_name")
    date_col = cols.get("date_col")

    # Normalização robusta
    df[vcol] = to_float_series_robust(df[vcol]).fillna(0.0)
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce").fillna(0).astype(float)

    # Datas (opcional)
    if date_col and date_col in df.columns:
        df["_data_venda"] = parse_date_series(df[date_col])
    else:
        df["_data_venda"] = pd.NaT

    # 1) Faturamento da rede (linha a linha)
    faturamento_rede = float(df[vcol].sum())

    # 2) Total de cupons (linha a linha)
    total_cupons = float(df[qcol].sum())

    # 3) Faturamento por loja
    fat_loja = df.groupby(store_col, dropna=False)[vcol].sum().reset_index()
    fat_loja.columns = ["codigo_loja", "faturamento"]

    # 4) Cupons por loja
    cupons_loja = df.groupby(store_col, dropna=False)[qcol].sum().reset_index()
    cupons_loja.columns = ["codigo_loja", "cupons"]

    lojas = fat_loja.merge(cupons_loja, on="codigo_loja", how="outer").fillna(0)
    lojas["ticket_medio_loja"] = np.where(lojas["cupons"]>0, lojas["faturamento"]/lojas["cupons"], np.nan)

    # 5) Vendedores (com nome)
    if seller_code and seller_code in df.columns:
        if seller_name and seller_name in df.columns:
            vendedores = df.groupby([seller_code, seller_name], dropna=False)[vcol].sum().reset_index()
            vendedores.columns = ["codigo_vendedor", "nome_vendedor", "faturamento"]
        else:
            vendedores = df.groupby(seller_code, dropna=False)[vcol].sum().reset_index()
            vendedores["nome_vendedor"] = ""
            vendedores.columns = ["codigo_vendedor", "faturamento", "nome_vendedor"]
            vendedores = vendedores[["codigo_vendedor", "nome_vendedor", "faturamento"]]
    else:
        vendedores = pd.DataFrame(columns=["codigo_vendedor", "nome_vendedor", "faturamento"])

    # 6) Departamentos (% sobre a rede) por código, com nome se existir
    if dept_code and dept_code in df.columns:
        dept = df.groupby(dept_code, dropna=False)[vcol].sum().reset_index()
        dept.columns = ["codigo_departamento", "faturamento"]
        if dept_name and dept_name in df.columns:
            names = df[[dept_code, dept_name]].dropna().drop_duplicates(subset=[dept_code])
            names.columns = ["codigo_departamento", "nome_departamento"]
            dept = dept.merge(names, on="codigo_departamento", how="left")
        else:
            dept["nome_departamento"] = ""
        # participação
        rede = faturamento_rede if faturamento_rede != 0 else np.nan
        dept["participacao_%"] = np.where(rede>0, dept["faturamento"] / rede * 100.0, np.nan)
        # ordenar por participação
        dept = dept[["codigo_departamento", "nome_departamento", "faturamento", "participacao_%"]]\
               .sort_values("participacao_%", ascending=False)
    else:
        dept = pd.DataFrame(columns=["codigo_departamento", "nome_departamento", "faturamento", "participacao_%"])

    # 7) Vendas diárias (rede)
    if df["_data_venda"].notna().any():
        vendas_diarias = df.groupby("_data_venda", dropna=False)[vcol].sum().reset_index()
        vendas_diarias.columns = ["data", "faturamento_dia"]
        vendas_diarias = vendas_diarias.sort_values("data")
    else:
        vendas_diarias = pd.DataFrame(columns=["data", "faturamento_dia"])

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

    return resumo, lojas, vendedores, faturamento_rede, total_cupons, dept, vendas_diarias

# ==============================
# Excel com dashboard
# ==============================
def build_excel_v44(df, resumo, lojas, vendedores, dept, vendas_diarias):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        # Abas
        df.to_excel(writer, sheet_name="Base", index=False)
        resumo.to_excel(writer, sheet_name="Resumo", index=False)
        lojas.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="Lojas", index=False)
        vendedores.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="Vendedores", index=False)
        dept.to_excel(writer, sheet_name="Departamentos", index=False)
        vendas_diarias.to_excel(writer, sheet_name="Vendas Diarias", index=False)

        wb = writer.book
        ws_resumo = writer.sheets["Resumo"]
        ws_lojas = writer.sheets["Lojas"]
        ws_vend = writer.sheets["Vendedores"]
        ws_dept = writer.sheets["Departamentos"]
        ws_dias = writer.sheets["Vendas Diarias"]

        # Formatos
        fmt_money = wb.add_format({'num_format': 'R$ #,##0.00'})
        fmt_int = wb.add_format({'num_format': '0'})
        fmt_pct = wb.add_format({'num_format': '0.00%'})
        fmt_pct_2d = wb.add_format({'num_format': '0.00" %"'})  # para exibir com símbolo e espaço

        # --- Resumo: inteiros para lojas/vendedores e cupons ---
        ws_resumo.set_column("A:A", 45)
        ws_resumo.set_column("B:B", 22, fmt_money)
        # varre e ajusta formatação para linhas específicas
        for r in range(0, len(resumo)):
            label = str(resumo.iloc[r, 0]).lower()
            if ("cupons" in label) or ("lojas" in label) or ("vendedores" in label):
                ws_resumo.write_number(r + 1, 1, float(resumo.iloc[r, 1]), fmt_int)

        # Lojas
        ws_lojas.set_column("A:A", 18)
        ws_lojas.set_column("B:B", 18, fmt_money)
        ws_lojas.set_column("C:C", 12, fmt_int)
        ws_lojas.set_column("D:D", 20, fmt_money)

        # Vendedores (inclui nome)
        ws_vend.set_column("A:A", 22)  # código
        ws_vend.set_column("B:B", 30)  # nome
        ws_vend.set_column("C:C", 18, fmt_money)  # faturamento

        # Departamentos: formatar percentuais e dinheiro
        ws_dept.set_column("A:A", 22)  # codigo
        ws_dept.set_column("B:B", 28)  # nome
        ws_dept.set_column("C:C", 18, fmt_money)  # faturamento
        ws_dept.set_column("D:D", 16, fmt_pct)    # participação

        # Vendas Diarias
        ws_dias.set_column("A:A", 14)  # data
        ws_dias.set_column("B:B", 18, fmt_money)  # faturamento_dia

        # Gráfico Top 10 Lojas por faturamento no Resumo
        top10 = lojas.sort_values("faturamento", ascending=False).head(10)
        chart_col = wb.add_chart({'type': 'column'})
        chart_col.add_series({
            'name': 'Faturamento (R$)',
            'categories': f"='Lojas'!$A$2:$A${len(top10) + 1}",
            'values':     f"='Lojas'!$B$2:$B${len(top10) + 1}",
            'data_labels': {'value': True, 'num_format': 'R$ #,##0.00'}
        })
        chart_col.set_title({'name': 'Top 10 Lojas por Faturamento'})
        chart_col.set_x_axis({'name': 'Código da Loja'})
        chart_col.set_y_axis({'name': 'R$'})
        ws_resumo.insert_chart('D2', chart_col, {'x_scale': 1.15, 'y_scale': 1.2})

        # Gráfico de Pizza de Participação por Departamento
        if len(dept) > 0:
            # colocar os dados no sheet Departamentos (já estão). Criar gráfico tipo pie usando nome como categoria e participação como valor.
            chart_pie = wb.add_chart({'type': 'pie'})
            last_row = len(dept) + 1
            chart_pie.add_series({
                'name': 'Participação por Departamento',
                'categories': f"='Departamentos'!$B$2:$B${last_row}",   # nome_departamento
                'values':     f"='Departamentos'!$D$2:$D${last_row}",   # participacao_% (0-100 -> precisamos dividir por 100?)
                'data_labels': {'percentage': True}
            })
            chart_pie.set_title({'name': 'Participação no Faturamento por Departamento'})
            ws_dept.insert_chart('F2', chart_pie, {'x_scale': 1.2, 'y_scale': 1.2})

    buffer.seek(0)
    return buffer.getvalue()

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Indicadores Drogaria – v4.4 (produção)", layout="wide")
st.title("📈 Indicadores de Vendas – Rede de Drogaria")
st.caption("Versão " + APP_VERSION + " – Aprimorado: vendedores (nome), inteiros no Resumo, departamentos (%), vendas diárias.")

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
    st.caption("Selecione as colunas equivalentes no seu arquivo. (Os campos opcionais habilitam recursos extras.)")

    display_cols = list(df.columns)
    normalized = normalize_columns(display_cols)
    norm_to_display = {n: d for n, d in zip(normalized, display_cols)}

    def sel(label, key):
        options = ["(vazio)"] + normalized
        return st.selectbox(label, options, index=0, key=key)

    store_code = sel("Coluna de **Código da Loja**", "store_code")
    seller_code = st.selectbox("Coluna de **Código do Vendedor** (opcional, para KPIs por vendedor)", ["(vazio)"] + normalized, index=0, key="seller_code")
    seller_name = st.selectbox("Coluna de **Nome do Vendedor** (opcional, aparece na aba Vendedores)", ["(vazio)"] + normalized, index=0, key="seller_name")
    value_col = sel("Coluna de **Valor Total Venda (R$)**", "value_col")
    qty_col = sel("Coluna de **Quantidade Vendida**", "qty_col")
    dept_code = st.selectbox("Coluna de **Código de Departamento** (opcional, para participação %)", ["(vazio)"] + normalized, index=0, key="dept_code")
    dept_name = st.selectbox("Coluna de **Nome do Departamento** (opcional)", ["(vazio)"] + normalized, index=0, key="dept_name")
    date_col = st.selectbox("Coluna de **Data da Venda** (opcional, para vendas diárias)", ["(vazio)"] + normalized, index=0, key="date_col")

    proceed = st.button("Gerar Indicadores (v4.4)")

    if proceed:
        mapped = {
            "store_code": norm_to_display.get(store_code) if store_code != "(vazio)" else None,
            "seller_code": norm_to_display.get(seller_code) if seller_code != "(vazio)" else None,
            "seller_name": norm_to_display.get(seller_name) if seller_name != "(vazio)" else None,
            "value": norm_to_display.get(value_col) if value_col != "(vazio)" else None,
            "qty": norm_to_display.get(qty_col) if qty_col != "(vazio)" else None,
            "dept_code": norm_to_display.get(dept_code) if dept_code != "(vazio)" else None,
            "dept_name": norm_to_display.get(dept_name) if dept_name != "(vazio)" else None,
            "date_col": norm_to_display.get(date_col) if date_col != "(vazio)" else None,
        }
        missing = [k for k in ["store_code","value","qty"] if mapped.get(k) is None]
        if missing:
            st.error("Mapeie as colunas obrigatórias: **Código da Loja**, **Valor Total Venda** e **Quantidade Vendida**.")
        else:
            try:
                (resumo, lojas, vendedores, fat_rede, cupons_total,
                 dept, vendas_diarias) = compute_indicators_v44(df.copy(), mapped)
            except Exception as e:
                st.error(f"Erro ao calcular indicadores: {e}")
            else:
                # KPIs no app
                st.subheader("📌 Resumo do Período (Regras v4.4)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Faturamento da Rede (linha a linha)", f"R$ {fat_rede:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                col2.metric("Total de Cupons (linha a linha)", f"{int(cupons_total):,}".replace(",", "."))
                col3.metric("Lojas Ativas", f"{int(lojas['codigo_loja'].nunique())}")
                col4.metric("Vendedores com Vendas", f"{int(vendedores['codigo_vendedor'].nunique()) if not vendedores.empty else 0}")

                # Tabelas
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🏬 Faturamento por Loja")
                    st.dataframe(lojas.sort_values("faturamento", ascending=False), use_container_width=True)
                with c2:
                    st.markdown("#### 👤 Faturamento por Vendedor (com nome)")
                    st.dataframe(vendedores.sort_values("faturamento", ascending=False), use_container_width=True)

                # Departamentos preview
                if len(dept) > 0:
                    st.markdown("#### 🧪 Participação por Departamento")
                    st.dataframe(dept, use_container_width=True)

                # Vendas diárias preview
                if len(vendas_diarias) > 0:
                    st.markdown("#### 📅 Vendas Diárias da Rede")
                    st.dataframe(vendas_diarias, use_container_width=True)

                # Download Excel
                st.divider()
                st.markdown("### ⬇️ Download do Excel (v4.4)")
                excel_bytes = build_excel_v44(df.copy(), resumo, lojas, vendedores, dept, vendas_diarias)
                st.download_button(
                    label="Baixar Excel (v4.4)",
                    data=excel_bytes,
                    file_name=f"Indicadores_Drogaria_v4_4_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
