# app.py
# -*- coding: utf-8 -*-
import io
import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import unicodedata  # ensure unicodedata is imported for unaccent()

# Silence watch observer errors on Streamlit Cloud (inotify limit)
st.set_option('server.fileWatcherType', 'none')

APP_VERSION = "v3.2 (produto obrigatório + unicodedata import + watcher off)"

# ------------------------------
# Helpers
# ------------------------------
REQUIRED_ROLES = {
    "date": ["data", "data da venda", "emissao", "dt", "dia"],
    "store": ["loja", "filial", "estabelecimento", "pdv", "ponto de venda"],
    "seller": ["vendedor", "operador", "colaborador", "atendente"],
    "dept": ["departamento", "secao", "categoria", "grupo"],
    "tx": ["cupom", "id venda", "numero venda", "nf", "nfe", "pedido", "transacao", "comprovante", "codigo do cupom"],
    "product": ["codigo do produto", "cod produto", "produto", "id produto", "sku"],
    "value": ["valor total", "valor venda", "total", "faturamento", "receita", "valor liquido", "preco total", "valor"],
    "qty": ["qtd", "quantidade", "qtde", "quantidade vendida"]
}

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
    norm = []
    for c in cols:
        c = unaccent(str(c)).lower().strip()
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

def ensure_required(df, cols_map):
    missing = [k for k in ["store","value","tx","product"] if (cols_map.get(k) is None or cols_map.get(k) not in df.columns)]
    if missing:
        raise ValueError("Mapeamento faltando para: " + ", ".join(missing))

def to_float_series(s: pd.Series):
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r"[^0-9,.\-]", "", regex=True)  # keep digits, dots, commas, minus
         .str.replace(".", "", regex=False)           # thousands sep
         .str.replace(",", ".", regex=False),         # decimal to dot
        errors="coerce"
    )

def robust_deduplicate(df: pd.DataFrame, cols_candidates: list) -> pd.DataFrame:
    subset = [c for c in cols_candidates if c and c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    return df

def aggregate_tx_sum_with_product(df: pd.DataFrame, tx_col: str, product_col: str, value_col: str, qty_col: str|None):
    """
    Calcula o valor da transação (cupom) evitando duplicidades.
    Regra:
      - Se o valor do cupom for idêntico em todas as linhas do cupom (total repetido), usa apenas UMA vez.
      - Caso contrário, soma os valores por (cupom, produto) únicos.
    """
    parts = []
    for tx, g in df.groupby(tx_col, dropna=False):
        # total do cupom repetido nas linhas
        if g[value_col].nunique() == 1 and len(g) > 1:
            total = float(g[value_col].iloc[0])
            parts.append((tx, total))
            continue
        # deduplicar por produto dentro do cupom
        g_dedup = g.drop_duplicates(subset=[tx_col, product_col])
        total = float(pd.to_numeric(g_dedup[value_col], errors="coerce").fillna(0).sum())
        parts.append((tx, total))
    return pd.DataFrame(parts, columns=[tx_col, "valor_tx"])

def allocate_department_values(df, cols_map):
    dept_col = cols_map.get("dept")
    if not dept_col or dept_col not in df.columns:
        # Se não há departamento mapeado, retornar vazio
        return pd.DataFrame(columns=["departamento", "faturamento", "participacao"])

    tx_col = cols_map["tx"]
    product_col = cols_map["product"]
    value_col = cols_map["value"]
    qty_col = cols_map.get("qty")

    allocated_values = []
    for tx, g in df.groupby(tx_col, dropna=False):
        if g[value_col].nunique() == 1 and len(g) > 1:
            total = float(g[value_col].iloc[0])
            if qty_col and qty_col in g.columns:
                weights = pd.to_numeric(g[qty_col], errors="coerce").fillna(0).astype(float)
                vals = total * (weights / weights.sum()) if weights.sum() > 0 else pd.Series([total/len(g)]*len(g), index=g.index)
            else:
                # dividir por produtos únicos se houver
                if product_col and product_col in g.columns and g[product_col].nunique() > 0:
                    n = g[product_col].nunique()
                    vals = pd.Series([total/n]*len(g), index=g.index)
                else:
                    vals = pd.Series([total/len(g)]*len(g), index=g.index)
        else:
            # somar por (tx, produto) únicos
            g_dedup = g.drop_duplicates(subset=[tx_col, product_col])
            # mapear valor por produto em g original
            vals_map = g_dedup.set_index(product_col)[value_col].to_dict()
            vals = g[product_col].map(vals_map)

        tmp = g.copy()
        tmp["_allocated_value"] = pd.to_numeric(vals, errors="coerce").fillna(0.0)
        allocated_values.append(tmp[[dept_col, "_allocated_value"]])

    if not allocated_values:
        return pd.DataFrame(columns=["departamento", "faturamento", "participacao"])

    allo = pd.concat(allocated_values, axis=0)
    fat_depto = allo.groupby(dept_col)["_allocated_value"].sum().reset_index().rename(columns={dept_col: "departamento", "_allocated_value": "faturamento"})
    return fat_depto

def compute_indicators(df: pd.DataFrame, cols_map: dict):
    date_col = cols_map.get("date")
    store_col = cols_map["store"]
    seller_col = cols_map.get("seller")
    dept_col = cols_map.get("dept")
    tx_col = cols_map["tx"]
    product_col = cols_map["product"]
    value_col = cols_map["value"]
    qty_col = cols_map.get("qty")

    # Types & cleaning
    if date_col is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df[value_col] = to_float_series(df[value_col])

    # Deduplicate rows conservadoramente
    df = robust_deduplicate(df, [date_col, store_col, seller_col, dept_col, tx_col, product_col, value_col, qty_col])

    # --- Total por transação, deduplicado por produto
    tx_total = aggregate_tx_sum_with_product(df, tx_col=tx_col, product_col=product_col, value_col=value_col, qty_col=qty_col)

    # Map store/seller/date por transação (modo/primeiro)
    loja_tx = df.groupby(tx_col)[store_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).rename("loja")
    if seller_col and seller_col in df.columns:
        vend_tx = df.groupby(tx_col)[seller_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).rename("vendedor")
    else:
        vend_tx = pd.Series(index=tx_total[tx_col], dtype=object, name="vendedor")

    if date_col and date_col in df.columns:
        data_tx = df.groupby(tx_col)[date_col].agg(lambda s: s.min()).rename("data")
    else:
        data_tx = pd.Series(index=tx_total[tx_col], dtype="datetime64[ns]", name="data")

    # Faturamento por loja
    lojas = tx_total.set_index(tx_col).join(loja_tx).groupby("loja", dropna=False)["valor_tx"].agg(["sum", "count"]).reset_index()
    lojas.columns = ["loja", "faturamento", "cupons"]
    lojas["ticket_medio_loja"] = lojas["faturamento"] / lojas["cupons"]

    # Faturamento por vendedor
    vendedores = tx_total.set_index(tx_col).join(vend_tx).groupby("vendedor", dropna=False)["valor_tx"].agg(["sum", "count"]).reset_index()
    vendedores.columns = ["vendedor", "faturamento", "cupons"]
    vendedores["ticket_medio_vendedor"] = vendedores["faturamento"] / vendedores["cupons"]

    # Totais e ticket médio da rede
    total_vendas = float(tx_total["valor_tx"].sum())
    n_transacoes = int(len(tx_total))
    ticket_medio_rede = (total_vendas / n_transacoes) if n_transacoes else np.nan

    # Participação por departamento
    fat_depto = allocate_department_values(df, cols_map)
    fat_depto = fat_depto.sort_values("faturamento", ascending=False)
    fat_depto["participacao"] = fat_depto["faturamento"] / total_vendas if total_vendas else 0

    # Vendas por dia a partir de transações (data por cupom)
    if data_tx is not None and data_tx.notna().any():
        vendas_dia = tx_total.set_index(tx_col).join(data_tx).groupby(data_tx.name)["valor_tx"].sum().reset_index()
        vendas_dia.columns = ["data", "faturamento"]
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
        "Valor": [total_vendas, ticket_medio_rede, int(lojas['loja'].nunique()), int(vendedores['vendedor'].nunique()), n_transacoes]
    })

    return resumo, lojas, vendedores, fat_depto, vendas_dia, total_vendas

def choose_label_format(max_value: float) -> str:
    if max_value is None or (isinstance(max_value, float) and np.isnan(max_value)):
        return 'R$ #,##0.00'
    if max_value >= 1_000_000:
        return 'R$ #,##0.00,,"M"'
    if max_value >= 1_000:
        return 'R$ #,##0.00,"K"'
    return 'R$ #,##0.00'

def build_excel(df, resumo, lojas, vendedores, departamentos, vendas_dia):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Base", index=False)
        resumo.to_excel(writer, sheet_name="Resumo", index=False)
        lojas.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="KPIs por Loja", index=False)
        vendedores.sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="KPIs por Vendedor", index=False)
        departamentos.to_excel(writer, sheet_name="Departamentos", index=False)
        top10 = lojas.sort_values("faturamento", ascending=False).head(10)
        top10.to_excel(writer, sheet_name="Top10 Lojas", index=False)
        vendas_dia.to_excel(writer, sheet_name="Vendas por Dia", index=False)

        wb  = writer.book
        ws_resumo = writer.sheets["Resumo"]
        ws_depto = writer.sheets["Departamentos"]
        ws_top10 = writer.sheets["Top10 Lojas"]
        ws_lojas = writer.sheets["KPIs por Loja"]
        ws_vend  = writer.sheets["KPIs por Vendedor"]
        ws_dia = writer.sheets.get("Vendas por Dia", None)

        fmt_money = wb.add_format({'num_format': 'R$ #,##0.00'})
        fmt_pct = wb.add_format({'num_format': '0.00%'})
        fmt_int = wb.add_format({'num_format': '0'})
        fmt_header = wb.add_format({'bold': True, 'bg_color': '#EEEEEE', 'border': 1})

        # Format columns
        ws_lojas.set_column('A:A', 28)
        ws_lojas.set_column('B:B', 18, fmt_money)
        ws_lojas.set_column('C:C', 10, fmt_int)
        ws_lojas.set_column('D:D', 18, fmt_money)

        ws_vend.set_column('A:A', 32)
        ws_vend.set_column('B:B', 18, fmt_money)
        ws_vend.set_column('C:C', 10, fmt_int)
        ws_vend.set_column('D:D', 18, fmt_money)

        ws_depto.set_column('A:A', 28)
        ws_depto.set_column('B:B', 18, fmt_money)
        ws_depto.set_column('C:C', 12, fmt_pct)

        ws_top10.set_column('A:A', 28)
        ws_top10.set_column('B:B', 18, fmt_money)

        if ws_dia is not None:
            ws_dia.set_column('A:A', 14)
            ws_dia.set_column('B:B', 18, fmt_money)

        # Chart Top10 lojas
        chart_top10 = wb.add_chart({'type': 'column'})
        chart_top10.add_series({
            'name':       'Faturamento',
            'categories': "='Top10 Lojas'!$A$2:$A$" + str(len(top10)+1),
            'values':     "='Top10 Lojas'!$B$2:$B$" + str(len(top10)+1),
            'data_labels': {'value': True, 'num_format': 'R$ #,##0.00'}
        })
        chart_top10.set_title({'name': 'Top 10 Lojas por Faturamento'})
        chart_top10.set_x_axis({'name': 'Loja'})
        chart_top10.set_y_axis({'name': 'R$'})
        ws_resumo.insert_chart('D2', chart_top10, {'x_scale': 1.2, 'y_scale': 1.2})

        # Chart participação por departamento (nome + %)
        chart_depto = wb.add_chart({'type': 'pie'})
        chart_depto.add_series({
            'name':       'Participação por Departamento',
            'categories': "='Departamentos'!$A$2:$A$" + str(len(departamentos)+1),
            'values':     "='Departamentos'!$B$2:$B$" + str(len(departamentos)+1),
            'data_labels': {'percentage': True, 'category': True}
        })
        chart_depto.set_title({'name': 'Participação por Departamento'})
        ws_resumo.insert_chart('D20', chart_depto, {'x_scale': 1.1, 'y_scale': 1.1})

        # Chart ticket médio por loja (abreviado 2 casas)
        top10_ticket = lojas.sort_values("faturamento", ascending=False).head(10).reset_index(drop=True)
        start_row = len(lojas) + 3
        ws_lojas.write(start_row, 0, "Loja (Top10 por faturamento)", fmt_header)
        ws_lojas.write(start_row, 1, "Ticket médio", fmt_header)
        for i, row in top10_ticket.iterrows():
            ws_lojas.write(start_row+1+i, 0, row["loja"])
            ws_lojas.write_number(start_row+1+i, 1, float(row["ticket_medio_loja"]), fmt_money)

        label_fmt = choose_label_format(float(top10_ticket["ticket_medio_loja"].max()) if len(top10_ticket)>0 else np.nan)
        chart_ticket = wb.add_chart({'type': 'column'})
        chart_ticket.add_series({
            'name': 'Ticket médio Top10',
            'categories': "='KPIs por Loja'!$A$" + str(start_row+2) + ":$A$" + str(start_row+1+len(top10_ticket)),
            'values':     "='KPIs por Loja'!$B$" + str(start_row+2) + ":$B$" + str(start_row+1+len(top10_ticket)),
            'data_labels': {'value': True, 'num_format': label_fmt}
        })
        chart_ticket.set_title({'name': 'Ticket médio por Loja (Top 10)'})
        chart_ticket.set_x_axis({'name': 'Loja'})
        chart_ticket.set_y_axis({'name': 'R$'})
        ws_resumo.insert_chart('L2', chart_ticket, {'x_scale': 1.1, 'y_scale': 1.1})

        # Chart faturamento por dia
        if ws_dia is not None and len(vendas_dia) > 1:
            chart_dia = wb.add_chart({'type': 'line'})
            chart_dia.add_series({
                'name': 'Faturamento por Dia',
                'categories': "='Vendas por Dia'!$A$2:$A$" + str(len(vendas_dia)+1),
                'values':     "='Vendas por Dia'!$B$2:$B$" + str(len(vendas_dia)+1),
                'data_labels': {'value': False}
            })
            chart_dia.set_title({'name': 'Faturamento Diário'})
            chart_dia.set_x_axis({'name': 'Data'})
            chart_dia.set_y_axis({'name': 'R$'})
            ws_resumo.insert_chart('L20', chart_dia, {'x_scale': 1.2, 'y_scale': 1.1})

    return buf.getvalue()

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Indicadores Drogaria (v3.2)", layout="wide")
st.title("📈 Indicadores de Vendas – Rede de Drogaria (" + APP_VERSION + ")")
st.caption("Ticket médio calculado com base em CUPOM deduplicado por CÓDIGO DE PRODUTO. Evita duplicidades no total do cupom e no somatório por loja/vendedor.")

uploaded = st.file_uploader("Envie seu arquivo (.csv, .xlsx)", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

if uploaded:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        sample_text = uploaded.getvalue().decode("utf-8", errors="ignore")[:5000]
        delim = detect_delimiter(sample_text, default=",")
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=delim, encoding="utf-8", low_memory=False)
    else:
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Selecione a planilha (aba)", options=xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)

    display_cols = list(df.columns)
    normalized = normalize_columns(display_cols)
    norm_to_display = {n: d for n, d in zip(normalized, display_cols)}
    mapping_guess = guess_mapping(normalized)

    st.subheader("🔧 Mapeamento de colunas")
    st.caption("**Obrigatório** mapear: Loja, Valor, Cupom/Transação **e** Código do Produto.")
    def sel(label, key, optional=False):
        options = ["(vazio)"] + normalized
        default_idx = 0
        if mapping_guess.get(key) in normalized:
            default_idx = options.index(mapping_guess[key])
        chosen = st.selectbox(label, options, index=default_idx, help=None)
        return None if chosen == "(vazio)" else chosen

    date_col = sel("Coluna de Data da venda (opcional)", "date", optional=True)
    store_col = sel("Coluna de Loja (OBRIGATÓRIO)", "store")
    seller_col = sel("Coluna de Vendedor (opcional)", "seller", optional=True)
    dept_col = sel("Coluna de Departamento/Categoria (opcional)", "dept", optional=True)
    tx_col = sel("Coluna de Cupom/Transação (OBRIGATÓRIO)", "tx")
    product_col = sel("Coluna de Código do Produto (OBRIGATÓRIO)", "product")
    value_col = sel("Coluna de Valor da Venda (R$) (OBRIGATÓRIO)", "value")
    qty_col = sel("Coluna de Quantidade (opcional)", "qty", optional=True)

    proceed = st.button("Gerar Indicadores e Arquivos (v3.2)")

    if proceed:
        cols_map = {
            "date": norm_to_display.get(date_col) if date_col else None,
            "store": norm_to_display.get(store_col) if store_col else None,
            "seller": norm_to_display.get(seller_col) if seller_col else None,
            "dept": norm_to_display.get(dept_col) if dept_col else None,
            "tx": norm_to_display.get(tx_col) if tx_col else None,
            "product": norm_to_display.get(product_col) if product_col else None,
            "value": norm_to_display.get(value_col) if value_col else None,
            "qty": norm_to_display.get(qty_col) if qty_col else None,
        }

        try:
            ensure_required(df, cols_map)
        except Exception as e:
            st.error("Mapeie **Loja**, **Valor**, **Cupom/Transação** e **Código do Produto** para continuar.")
        else:
            try:
                resumo, lojas, vendedores, departamentos, vendas_dia, total_vendas = compute_indicators(df.copy(), cols_map)
            except Exception as e:
                st.error(f"Erro ao calcular indicadores: {e}")
            else:
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
                col5.metric("Transações (Cupons)", f"{n_cupons}")

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🏬 Top 10 Lojas por Faturamento (por cupom)")
                    st.dataframe(lojas.sort_values("faturamento", ascending=False).head(10), use_container_width=True)
                with c2:
                    st.markdown("#### 🧩 Participação por Departamento")
                    st.dataframe(departamentos.sort_values("faturamento", ascending=False), use_container_width=True)

                st.markdown("#### 👤 KPIs por Vendedor (com nome do vendedor)")
                st.dataframe(vendedores.sort_values("faturamento", ascending=False), use_container_width=True)

                st.markdown("#### 📅 Vendas por Dia (por cupom)")
                st.dataframe(vendas_dia, use_container_width=True)

                st.divider()
                st.markdown("### ⬇️ Downloads")
                excel_bytes = build_excel(df.copy(), resumo, lojas, vendedores, departamentos, vendas_dia)
                st.download_button(
                    label="Baixar Excel com Dashboard (v3.2)",
                    data=excel_bytes,
                    file_name=f"Indicadores_Drogaria_v3_2_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ------------------------------
# Footer note
# ------------------------------
st.caption("© 2025 – " + APP_VERSION + " | Ticket médio por cupom deduplicado por produto; BRL no Excel; rótulos por departamento; deduplicação.") 
