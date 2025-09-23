# app.py
# -*- coding: utf-8 -*-
import io
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from unidecode import unidecode

# ------------------------------
# Helpers
# ------------------------------
REQUIRED_ROLES = {
    "date": ["data", "data da venda", "emissao", "dt", "dia"],
    "store": ["loja", "filial", "estabelecimento", "pdv", "ponto de venda"],
    "seller": ["vendedor", "operador", "colaborador", "atendente"],
    "dept": ["departamento", "secao", "categoria", "grupo"],
    "tx": ["cupom", "id venda", "numero venda", "nf", "nfe", "pedido", "transacao", "comprovante", "codigo do produto"],
    "value": ["valor total", "valor venda", "total", "faturamento", "receita", "valor liquido", "preco total", "valor"]
}

@st.cache_data(show_spinner=False)
def detect_delimiter(sample_text: str, default=","):
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;|\t")
        return dialect.delimiter
    except Exception:
        return default

def normalize_columns(cols):
    norm = []
    for c in cols:
        c = unidecode(str(c)).lower().strip()
        for ch in ["-", "_", "/", "\\", "  "]:
            c = c.replace(ch, " ")
        c = " ".join(c.split())
        norm.append(c)
    return norm

def guess_mapping(columns):
    mapping = {k: None for k in REQUIRED_ROLES.keys()}
    for role, keys in REQUIRED_ROLES.items():
        for key in keys:
            for col in columns:
                if key in col:
                    mapping[role] = col
                    break
            if mapping[role]:
                break
    return mapping

def ensure_transaction_id(df, tx_col):
    if tx_col is None or tx_col not in df.columns:
        df["__tx_id"] = np.arange(len(df))
        return "__tx_id"
    return tx_col

def to_float_series(s: pd.Series):
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r"[^0-9,.-]", "", regex=True)
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False),
        errors="coerce"
    )

def make_dashboard_excel(df, cols_map) -> bytes:
    date_col = cols_map["date"]
    store_col = cols_map["store"]
    seller_col = cols_map["seller"]
    dept_col = cols_map["dept"]
    tx_col = ensure_transaction_id(df, cols_map.get("tx"))
    value_col = cols_map["value"]

    # Coerce types
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df[value_col] = to_float_series(df[value_col])
    df = df.dropna(subset=[value_col])

    # Fallbacks
    if store_col is None:
        df["__loja"] = "Loja Unica"
        store_col = "__loja"
    if seller_col is None:
        df["__vendedor"] = "Nao informado"
        seller_col = "__vendedor"
    if dept_col is None:
        df["__departamento"] = "Nao informado"
        dept_col = "__departamento"

    # KPIs
    total_vendas = float(df[value_col].sum())
    n_transacoes = int(df[tx_col].nunique()) if tx_col in df.columns else len(df)
    ticket_medio_rede = (total_vendas / n_transacoes) if n_transacoes else np.nan
    qtd_lojas = int(df[store_col].nunique())
    qtd_vendedores = int(df[seller_col].nunique())

    lojas = df.groupby(store_col).agg(
        faturamento=(value_col, "sum"),
        cupons=(tx_col, "nunique") if tx_col in df.columns else (value_col, "count")
    ).reset_index().rename(columns={store_col: "loja"})
    lojas["ticket_medio_loja"] = lojas["faturamento"] / lojas["cupons"]

    vendedores = df.groupby(seller_col).agg(
        faturamento=(value_col, "sum"),
        cupons=(tx_col, "nunique") if tx_col in df.columns else (value_col, "count")
    ).reset_index().rename(columns={seller_col: "vendedor"})
    vendedores["ticket_medio_vendedor"] = vendedores["faturamento"] / vendedores["cupons"]

    departamentos = df.groupby(dept_col).agg(
        faturamento=(value_col, "sum")
    ).reset_index().rename(columns={dept_col: "departamento"})
    departamentos["participacao"] = departamentos["faturamento"] / total_vendas if total_vendas else 0

    top10_lojas = lojas.sort_values("faturamento", ascending=False).head(10)

    if date_col is not None and df[date_col].notna().any():
        vendas_dia = df.groupby(df[date_col].dt.date).agg(faturamento=(value_col, "sum")).reset_index().rename(columns={date_col: "data"})
    else:
        vendas_dia = pd.DataFrame({"data": [], "faturamento": []})

    resumo = pd.DataFrame({
        "Indicador": [
            "Total de vendas no período (R$)",
            "Ticket médio da rede (R$)",
            "Quantidade de lojas",
            "Quantidade de vendedores ativos",
            "Número de transações (cupons)"
        ],
        "Valor": [total_vendas, ticket_medio_rede, qtd_lojas, qtd_vendedores, n_transacoes]
    })

    # Prepare summarized dataset (minimal columns)
    minimal_cols = {
        "data": date_col if date_col else None,
        "loja": store_col,
        "vendedor": seller_col,
        "departamento": dept_col,
        "cupom": tx_col if tx_col in df.columns else None,
        "valor": value_col
    }
    minimal_cols = {k: v for k, v in minimal_cols.items() if v is not None}
    resumo_min = df[list(minimal_cols.values())].copy()
    resumo_min.columns = list(minimal_cols.keys())

    # Build Excel into memory
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Base", index=False)
        resumo.to_excel(writer, sheet_name="Resumo", index=False)
        lojas.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="KPIs por Loja", index=False)
        vendedores.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="KPIs por Vendedor", index=False)
        departamentos.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="Departamentos", index=False)
        top10_lojas.to_excel(writer, sheet_name="Top10 Lojas", index=False)
        vendas_dia.to_excel(writer, sheet_name="Vendas por Dia", index=False)

        wb = writer.book
        ws_resumo = writer.sheets["Resumo"]
        ws_depto = writer.sheets["Departamentos"]
        ws_top10 = writer.sheets["Top10 Lojas"]
        ws_lojas = writer.sheets["KPIs por Loja"]
        ws_dia = writer.sheets.get("Vendas por Dia", None)

        fmt_money = wb.add_format({'num_format': 'R$ #,##0.00'})
        fmt_int = wb.add_format({'num_format': '0'})
        fmt_header = wb.add_format({'bold': True, 'bg_color': '#EEEEEE', 'border': 1})

        # Style header Resumo
        ws_resumo.write(0, 0, "Indicador", fmt_header)
        ws_resumo.write(0, 1, "Valor", fmt_header)
        for r in range(1, len(resumo) + 1):
            label = resumo.iloc[r-1, 0]
            val = resumo.iloc[r-1, 1]
            if "Ticket" in label or "vendas" in label:
                try:
                    ws_resumo.write_number(r, 1, float(val), fmt_money)
                except Exception:
                    ws_resumo.write(r, 1, str(val))
            elif "Quantidade" in label or "Número" in label:
                try:
                    ws_resumo.write_number(r, 1, float(val), fmt_int)
                except Exception:
                    ws_resumo.write(r, 1, str(val))
            else:
                try:
                    ws_resumo.write_number(r, 1, float(val))
                except Exception:
                    ws_resumo.write(r, 1, str(val))

        # Charts: Top10 lojas (column)
        chart_top10 = wb.add_chart({'type': 'column'})
        chart_top10.add_series({
            'name': 'Faturamento',
            'categories': "='Top10 Lojas'!$A$2:$A$" + str(len(top10_lojas) + 1),
            'values':     "='Top10 Lojas'!$B$2:$B$" + str(len(top10_lojas) + 1),
            'data_labels': {'value': True}
        })
        chart_top10.set_title({'name': 'Top 10 Lojas por Faturamento'})
        chart_top10.set_x_axis({'name': 'Loja'})
        chart_top10.set_y_axis({'name': 'R$'})
        ws_resumo.insert_chart('D2', chart_top10, {'x_scale': 1.2, 'y_scale': 1.2})

        # Charts: Participação por departamento (pie)
        chart_depto = wb.add_chart({'type': 'pie'})
        chart_depto.add_series({
            'name': 'Participação por Departamento',
            'categories': "='Departamentos'!$A$2:$A$" + str(len(departamentos) + 1),
            'values':     "='Departamentos'!$C$2:$C$" + str(len(departamentos) + 1),
            'data_labels': {'percentage': True}
        })
        chart_depto.set_title({'name': 'Participação por Departamento'})
        ws_resumo.insert_chart('D20', chart_depto, {'x_scale': 1.1, 'y_scale': 1.1})

        # Charts: Ticket médio por loja (Top10)
        lojas_sorted = lojas.sort_values("faturamento", ascending=False).head(10).reset_index(drop=True)
        start_row = len(lojas) + 3
        ws_lojas.write(start_row, 0, "Loja (Top10 por faturamento)", fmt_header)
        ws_lojas.write(start_row, 1, "Ticket médio", fmt_header)
        for i, row in lojas_sorted.iterrows():
            ws_lojas.write(start_row + 1 + i, 0, row["loja"])
            ws_lojas.write_number(start_row + 1 + i, 1, float(row["ticket_medio_loja"]))

        chart_ticket = wb.add_chart({'type': 'column'})
        chart_ticket.add_series({
            'name': 'Ticket médio Top10',
            'categories': "='KPIs por Loja'!$A$" + str(start_row + 2) + ":$A$" + str(start_row + 1 + len(lojas_sorted)),
            'values':     "='KPIs por Loja'!$B$" + str(start_row + 2) + ":$B$" + str(start_row + 1 + len(lojas_sorted)),
            'data_labels': {'value': True}
        })
        chart_ticket.set_title({'name': 'Ticket médio por Loja (Top 10)'})
        chart_ticket.set_x_axis({'name': 'Loja'})
        chart_ticket.set_y_axis({'name': 'R$'})
        ws_resumo.insert_chart('L2', chart_ticket, {'x_scale': 1.1, 'y_scale': 1.1})

        # Charts: Faturamento por dia (line)
        if not vendas_dia.empty and len(vendas_dia) > 1:
            chart_dia = wb.add_chart({'type': 'line'})
            chart_dia.add_series({
                'name': 'Faturamento por Dia',
                'categories': "='Vendas por Dia'!$A$2:$A$" + str(len(vendas_dia) + 1),
                'values':     "='Vendas por Dia'!$B$2:$B$" + str(len(vendas_dia) + 1)
            })
            chart_dia.set_title({'name': 'Faturamento Diário'})
            chart_dia.set_x_axis({'name': 'Data'})
            chart_dia.set_y_axis({'name': 'R$'})
            ws_resumo.insert_chart('L20', chart_dia, {'x_scale': 1.2, 'y_scale': 1.1})

    return buf.getvalue(), resumo, lojas, vendedores, departamentos, top10_lojas, vendas_dia, resumo_min

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Indicadores Drogaria", layout="wide")
st.title("📈 Indicadores de Vendas – Rede de Drogaria")
st.caption("Envie um CSV ou Excel, mapeie as colunas e gere o dashboard + arquivo resumo para download.")

uploaded = st.file_uploader("Envie seu arquivo (.csv, .xlsx)", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

if uploaded:
    name = uploaded.name.lower()
    # Load data
    if name.endswith(".csv"):
        # Read small sample to detect delimiter
        sample_text = uploaded.getvalue().decode("utf-8", errors="ignore")[:5000]
        delim = detect_delimiter(sample_text, default=",")
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=delim, encoding="utf-8", low_memory=False)
    else:
        # Excel – if multiple sheets, let user choose
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Selecione a planilha (aba)", options=xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)

    # Normalize column display names
    display_cols = list(df.columns)
    normalized = normalize_columns(display_cols)
    norm_to_display = {n: d for n, d in zip(normalized, display_cols)}

    # Guess mapping
    mapping_guess = guess_mapping(normalized)

    st.subheader("🔧 Mapeamento de colunas")
    st.caption("Confirme/ajuste abaixo para cada papel. Campos opcionais podem ser deixados em branco.")

    def sel(label, key, optional=False):
        options = ["(vazio)"] + normalized
        default_idx = 0
        if mapping_guess.get(key) in normalized:
            default_idx = options.index(mapping_guess[key])
        chosen = st.selectbox(label, options, index=default_idx, help=None)
        return None if chosen == "(vazio)" else chosen

    date_col = sel("Coluna de Data da venda", "date", optional=True)
    store_col = sel("Coluna de Loja", "store")
    seller_col = sel("Coluna de Vendedor", "seller", optional=True)
    dept_col = sel("Coluna de Departamento/Categoria", "dept", optional=True)
    tx_col = sel("Coluna de Cupom/Transação (opcional)", "tx", optional=True)
    value_col = sel("Coluna de Valor da Venda (R$)", "value")

    proceed = st.button("Gerar Indicadores e Arquivos")

    if proceed:
        # Build cols_map using display names
        cols_map = {
            "date": norm_to_display.get(date_col) if date_col else None,
            "store": norm_to_display.get(store_col) if store_col else None,
            "seller": norm_to_display.get(seller_col) if seller_col else None,
            "dept": norm_to_display.get(dept_col) if dept_col else None,
            "tx": norm_to_display.get(tx_col) if tx_col else None,
            "value": norm_to_display.get(value_col) if value_col else None,
        }

        if cols_map["store"] is None or cols_map["value"] is None:
            st.error("Mapeie pelo menos as colunas de **Loja** e **Valor** para continuar.")
        else:
            data_bytes, resumo, lojas, vendedores, departamentos, top10, vendas_dia, resumo_min = make_dashboard_excel(df.copy(), cols_map)

            # KPIs on screen
            st.subheader("📌 Resumo do Período")
            col1, col2, col3, col4, col5 = st.columns(5)
            try:
                total = float(resumo.loc[resumo['Indicador'].str.contains('Total de vendas'), 'Valor'].iloc[0])
                ticket = float(resumo.loc[resumo['Indicador'].str.contains('Ticket médio'), 'Valor'].iloc[0])
                q_lojas = int(resumo.loc[resumo['Indicador'].str.contains('Quantidade de lojas'), 'Valor'].iloc[0])
                q_vend = int(resumo.loc[resumo['Indicador'].str.contains('vendedores ativos'), 'Valor'].iloc[0])
                n_cupons = int(resumo.loc[resumo['Indicador'].str.contains('transacoes'), 'Valor'].iloc[0])
            except Exception:
                total = ticket = q_lojas = q_vend = n_cupons = np.nan

            col1.metric("Total de Vendas", f"R$ {total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            col2.metric("Ticket Médio da Rede", f"R$ {ticket:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            col3.metric("Lojas Ativas", f"{q_lojas}")
            col4.metric("Vendedores Ativos", f"{q_vend}")
            col5.metric("Transações", f"{n_cupons}")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🏬 Top 10 Lojas por Faturamento")
                st.dataframe(top10, use_container_width=True)
            with c2:
                st.markdown("#### 🧩 Participação por Departamento")
                st.dataframe(departamentos.sort_values("faturamento", ascending=False), use_container_width=True)

            st.markdown("#### 📅 Vendas por Dia")
            st.dataframe(vendas_dia, use_container_width=True)

            st.divider()
            st.markdown("### ⬇️ Downloads")
            st.download_button(
                label="Baixar Excel com Dashboard (XLSX)",
                data=data_bytes,
                file_name=f"Indicadores_Drogaria_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            csv_min = resumo_min.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Baixar CSV Resumido (para reprocessar)",
                data=csv_min,
                file_name=f"Resumo_Minimo_Drogaria_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# ------------------------------
# Footer note
# ------------------------------
st.caption("© 2025 – App de indicadores para varejo farmacêutico. Suporta CSV/Excel e gera dashboard com KPIs, ranking e participação por departamento.")
