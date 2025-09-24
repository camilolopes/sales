
# app.py
# -*- coding: utf-8 -*-
# v4.8.3 – correções pós-log: remove Plotly, compatível Streamlit 1.50+, sem `use_container_width`, Matplotlib 'Agg'

import io
import re
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

APP_VERSION = "v4.8.3 (produção | correções pós-log)"

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

def guess_column(normalized_cols, keywords):
    for col in normalized_cols:
        if all(k in col for k in keywords):
            return col
    return None

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
    return pd.to_datetime(series, errors="coerce", dayfirst=True)

def fmt_brl(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def brl_formatter():
    return FuncFormatter(lambda x, pos: fmt_brl(x))

def pct_ptbr(x: float) -> str:
    return f"{x:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")

def df_show(data):
    # Streamlit >=1.50
    return st.dataframe(data, width='stretch')

# ==============================
# Cálculos principais (linha a linha, sem deduplicar cupons)
# ==============================
def compute_indicators(df: pd.DataFrame, cols: dict, pct_meta_growth: float):
    vcol = cols["value"]
    qcol = cols["qty"]
    store_col = cols["store_code"]
    seller_code = cols.get("seller_code")
    seller_name = cols.get("seller_name")
    dept_code = cols.get("dept_code")
    dept_name = cols.get("dept_name")
    date_col = cols.get("date_col")
    prod_code = cols.get("prod_code")
    prod_name = cols.get("prod_name")

    # Conversões
    df[vcol] = to_float_series_robust(df[vcol]).fillna(0.0)
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce").fillna(0).astype(float)

    if date_col and date_col in df.columns:
        df["_data_venda"] = parse_date_series(df[date_col])
    else:
        df["_data_venda"] = pd.NaT

    if pd.api.types.is_datetime64_any_dtype(df["_data_venda"]):
        df["_date"] = df["_data_venda"].dt.date
        isocal = df["_data_venda"].dt.isocalendar()
        df["_iso_year"] = isocal.year
        df["_iso_week"] = isocal.week
        df["_weekday"] = df["_data_venda"].dt.weekday
    else:
        df["_date"] = pd.NaT
        df["_iso_year"] = np.nan
        df["_iso_week"] = np.nan
        df["_weekday"] = np.nan

    # 1) Faturamento da rede = soma linha a linha (cupom já consolidado)
    faturamento_rede = float(df[vcol].sum())

    # 5) Total de cupons (soma da coluna quantidade vendida)
    total_cupons = float(df[qcol].sum())

    # 2) Faturamento por loja = soma por código da loja
    fat_loja = df.groupby(store_col, dropna=False)[vcol].sum().reset_index()
    fat_loja.columns = ["codigo_loja", "faturamento"]

    # 3) Cupons por loja (soma quantidade vendida por código da loja)
    cupons_loja = df.groupby(store_col, dropna=False)[qcol].sum().reset_index()
    cupons_loja.columns = ["codigo_loja", "cupons"]

    lojas = fat_loja.merge(cupons_loja, on="codigo_loja", how="outer").fillna(0)
    lojas["ticket_medio_loja"] = np.where(lojas["cupons"]>0, lojas["faturamento"]/lojas["cupons"], np.nan)

    # 4) Vendas por vendedor
    if seller_code and seller_code in df.columns:
        if seller_name and seller_name in df.columns:
            vendedores = df.groupby([seller_code, seller_name], dropna=False)[vcol].sum().reset_index()
            vendedores.columns = ["codigo_vendedor", "nome_vendedor", "faturamento"]
        else:
            vendedores = df.groupby(seller_code, dropna=False)[vcol].sum().reset_index()
            vendedores["nome_vendedor"] = ""
            vendedores.columns = ["codigo_vendedor", "faturamento", "nome_vendedor"]
            vendedores = vendedores[["codigo_vendedor", "nome_vendedor", "faturamento"]]
        vendedores["% meta crescimento"] = pct_meta_growth
        vendedores["meta"] = vendedores["faturamento"] * (1.0 + pct_meta_growth/100.0)
    else:
        vendedores = pd.DataFrame(columns=["codigo_vendedor", "nome_vendedor", "faturamento", "% meta crescimento", "meta"])

    # 7) Participação de departamento (usando código p/ evitar duplicidades)
    if dept_code and dept_code in df.columns:
        dept = df.groupby(dept_code, dropna=False)[vcol].sum().reset_index()
        dept.columns = ["codigo_departamento", "faturamento"]
        if dept_name and dept_name in df.columns:
            names = df[[dept_code, dept_name]].dropna().drop_duplicates(subset=[dept_code])
            names.columns = ["codigo_departamento", "nome_departamento"]
            dept = dept.merge(names, on="codigo_departamento", how="left")
        else:
            dept["nome_departamento"] = ""
        denom = faturamento_rede if faturamento_rede != 0 else np.nan
        dept["participacao_frac"] = np.where(denom>0, dept["faturamento"] / denom, np.nan)
        dept["participacao_%"] = dept["participacao_frac"] * 100.0
        dept = dept[["codigo_departamento","nome_departamento","faturamento","participacao_frac","participacao_%"]]\
                 .sort_values("participacao_%", ascending=False)
        dept_top5 = dept.head(5).copy()
    else:
        dept = pd.DataFrame(columns=["codigo_departamento","nome_departamento","faturamento","participacao_frac","participacao_%"])
        dept_top5 = dept.copy()

    # Vendas diárias
    if pd.api.types.is_datetime64_any_dtype(df["_data_venda"]):
        vendas_diarias = df.groupby(df["_data_venda"].dt.date, dropna=False)[vcol].sum().reset_index()
        vendas_diarias.columns = ["data", "faturamento_dia"]
        vendas_diarias = vendas_diarias.sort_values("data")
    else:
        vendas_diarias = pd.DataFrame(columns=["data","faturamento_dia"])

    # Média diária por vendedor
    if seller_code and seller_code in df.columns and pd.api.types.is_datetime64_any_dtype(df["_data_venda"]):
        vend_dia = df.groupby([seller_code, df["_data_venda"].dt.date], dropna=False)[vcol].sum().reset_index()
        vend_dia.columns = ["codigo_vendedor", "data", "faturamento_dia"]
        media_diaria_vendedor = vend_dia.groupby("codigo_vendedor", dropna=False)["faturamento_dia"].mean().reset_index()
        media_diaria_vendedor.columns = ["codigo_vendedor", "faturamento_medio_diario"]
        if seller_name and seller_name in df.columns:
            names = df[[seller_code, seller_name]].dropna().drop_duplicates(subset=[seller_code])
            names.columns = ["codigo_vendedor", "nome_vendedor"]
            media_diaria_vendedor = media_diaria_vendedor.merge(names, on="codigo_vendedor", how="left")
            media_diaria_vendedor = media_diaria_vendedor[["codigo_vendedor","nome_vendedor","faturamento_medio_diario"]]
    else:
        media_diaria_vendedor = pd.DataFrame(columns=["codigo_vendedor","nome_vendedor","faturamento_medio_diario"])

    # Média semanal por vendedor por unidade (valor e quantidade)
    if seller_code and seller_code in df.columns and store_col in df.columns and pd.api.types.is_datetime64_any_dtype(df["_data_venda"]):
        isocal = df["_data_venda"].dt.isocalendar()
        vend_sem = df.groupby([seller_code, store_col, isocal.year, isocal.week], dropna=False)[[vcol, qcol]].sum().reset_index()
        vend_sem.columns = ["codigo_vendedor","codigo_loja","ano_iso","semana_iso","faturamento_semana","qtd_semana"]
        media_semanal_vendedor_loja = vend_sem.groupby(["codigo_vendedor","codigo_loja"], dropna=False)[["faturamento_semana","qtd_semana"]].mean().reset_index()
        if seller_name and seller_name in df.columns:
            names = df[[seller_code, seller_name]].dropna().drop_duplicates(subset=[seller_code])
            names.columns = ["codigo_vendedor","nome_vendedor"]
            media_semanal_vendedor_loja = media_semanal_vendedor_loja.merge(names, on="codigo_vendedor", how="left")
            media_semanal_vendedor_loja = media_semanal_vendedor_loja[["codigo_vendedor","nome_vendedor","codigo_loja","faturamento_semana","qtd_semana"]]
    else:
        media_semanal_vendedor_loja = pd.DataFrame(columns=["codigo_vendedor","nome_vendedor","codigo_loja","faturamento_semana","qtd_semana"])

    # Média mensal loja e vendedor
    if pd.api.types.is_datetime64_any_dtype(df["_data_venda"]):
        loja_mes = df.groupby([store_col, df["_data_venda"].dt.to_period("M")], dropna=False)[vcol].sum().reset_index()
        loja_mes.columns = ["codigo_loja","mes","faturamento_mes"]
        media_mensal_loja = loja_mes.groupby("codigo_loja", dropna=False)["faturamento_mes"].mean().reset_index()
        media_mensal_loja.columns = ["codigo_loja","faturamento_medio_mensal_loja"]

        if seller_code and seller_code in df.columns:
            vend_mes = df.groupby([seller_code, df["_data_venda"].dt.to_period("M")], dropna=False)[vcol].sum().reset_index()
            vend_mes.columns = ["codigo_vendedor","mes","faturamento_mes"]
            media_mensal_vendedor = vend_mes.groupby("codigo_vendedor", dropna=False)["faturamento_mes"].mean().reset_index()
            media_mensal_vendedor.columns = ["codigo_vendedor","faturamento_medio_mensal_vendedor"]
            if seller_name and seller_name in df.columns:
                names = df[[seller_code, seller_name]].dropna().drop_duplicates(subset=[seller_code])
                names.columns = ["codigo_vendedor","nome_vendedor"]
                media_mensal_vendedor = media_mensal_vendedor.merge(names, on="codigo_vendedor", how="left")
                media_mensal_vendedor = media_mensal_vendedor[["codigo_vendedor","nome_vendedor","faturamento_medio_mensal_vendedor"]]
        else:
            media_mensal_vendedor = pd.DataFrame(columns=["codigo_vendedor","nome_vendedor","faturamento_medio_mensal_vendedor"])
    else:
        media_mensal_loja = pd.DataFrame(columns=["codigo_loja","faturamento_medio_mensal_loja"])
        media_mensal_vendedor = pd.DataFrame(columns=["codigo_vendedor","nome_vendedor","faturamento_medio_mensal_vendedor"])

    # Dia da semana – total e média; heatmap usa a média
    if pd.api.types.is_datetime64_any_dtype(df["_data_venda"]):
        wd_names = ["Segunda","Terça","Quarta","Quinta","Sexta","Sábado","Domingo"]
        wd_sum = df.groupby(df["_data_venda"].dt.weekday, dropna=False)[vcol].sum().reindex(range(7), fill_value=0.0)
        wd_count = df.groupby(df["_data_venda"].dt.weekday, dropna=False)["_date"].nunique().reindex(range(7), fill_value=0)
        wd_avg = np.where(wd_count>0, wd_sum.values / wd_count.values, np.nan)
        wd_table = pd.DataFrame({
            "dia_semana": wd_names,
            "faturamento total período": wd_sum.values,
            "ocorrencias_no_periodo": wd_count.values,
            "faturamento_medio": wd_avg
        })
        melhor_idx = int(np.nanargmax(wd_sum.values)) if len(wd_sum.values)>0 else None
        melhor_dia = wd_names[melhor_idx] if melhor_idx is not None else ""
    else:
        wd_table = pd.DataFrame(columns=["dia_semana","faturamento total período","ocorrencias_no_periodo","faturamento_medio"])
        melhor_dia = ""

    # Top produtos nos 3 dias com maior faturamento (se tivermos produto + data)
    if prod_code and prod_code in df.columns and pd.api.types.is_datetime64_any_dtype(df["_data_venda"]):
        day_map = {0:"Segunda",1:"Terça",2:"Quarta",3:"Quinta",4:"Sexta",5:"Sábado",6:"Domingo"}
        order = np.argsort(-wd_sum.values)[:3]
        frames = []
        for idx in order:
            if np.isfinite(wd_sum.values[idx]) and wd_sum.values[idx] > 0:
                day_name = day_map[idx]
                subset = df[df["_data_venda"].dt.weekday==idx]
                agg = subset.groupby(prod_code, dropna=False).agg(
                    quantidade=(qcol, "sum"),
                    faturamento=(vcol, "sum")
                ).reset_index()
                if prod_name and prod_name in df.columns:
                    names = df[[prod_code, prod_name]].dropna().drop_duplicates(subset=[prod_code])
                    names.columns = ["codigo_produto","nome_produto"]
                    agg = agg.merge(names, how="left", left_on=prod_code, right_on="codigo_produto")
                else:
                    agg["codigo_produto"] = agg[prod_code]
                    agg["nome_produto"] = ""
                agg["dia_semana"] = day_name
                agg = agg.sort_values(["quantidade","faturamento"], ascending=[False, False]).head(50)
                agg = agg[["dia_semana","codigo_produto","nome_produto","quantidade","faturamento"]]
                frames.append(agg)
        top_produtos_top3_dias = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["dia_semana","codigo_produto","nome_produto","quantidade","faturamento"])
    else:
        top_produtos_top3_dias = pd.DataFrame(columns=["dia_semana","codigo_produto","nome_produto","quantidade","faturamento"])

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

    return {
        "resumo": resumo,
        "lojas": lojas,
        "vendedores": vendedores,
        "faturamento_rede": faturamento_rede,
        "cupons_total": total_cupons,
        "dept": dept,
        "dept_top5": dept_top5,
        "vendas_diarias": vendas_diarias,
        "media_diaria_vendedor": media_diaria_vendedor,
        "media_semanal_vendedor_loja": media_semanal_vendedor_loja,
        "media_mensal_loja": media_mensal_loja,
        "media_mensal_vendedor": media_mensal_vendedor,
        "wd_table": wd_table,
        "melhor_dia": melhor_dia,
        "top_produtos_top3_dias": top_produtos_top3_dias
    }

# ==============================
# Excel com dashboard
# ==============================
def build_excel(df, result_dict, pct_meta_growth):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Base", index=False)
        result_dict["resumo"].to_excel(writer, sheet_name="Resumo", index=False)
        result_dict["lojas"].sort_values("faturamento", ascending=False).to_excel(writer, sheet_name="Lojas", index=False)

        vend = result_dict["vendedores"].copy().sort_values("faturamento", ascending=False).reset_index(drop=True)
        vend.to_excel(writer, sheet_name="Vendedores", index=False, startrow=0)
        wb = writer.book
        ws_vend = writer.sheets["Vendedores"]

        fmt_money = wb.add_format({'num_format': 'R$ #,##0.00'})
        fmt_int = wb.add_format({'num_format': '0'})
        fmt_pct = wb.add_format({'num_format': '0.00%'})
        ws_vend.set_column("A:A", 22)
        ws_vend.set_column("B:B", 28)
        ws_vend.set_column("C:C", 18, fmt_money)
        ws_vend.set_column("D:D", 18)
        ws_vend.set_column("E:E", 18, fmt_money)
        nrows = len(vend)
        for i in range(2, nrows+2):
            ws_vend.write_formula(i-1, 4, f"=C{i}*(1 + D{i}/100)")

        result_dict["dept"].to_excel(writer, sheet_name="Departamentos (100%)", index=False)
        result_dict["dept_top5"].to_excel(writer, sheet_name="Dept Top5", index=False)
        result_dict["vendas_diarias"].to_excel(writer, sheet_name="Vendas Diarias", index=False)
        result_dict["media_diaria_vendedor"].to_excel(writer, sheet_name="Vendedor Media Diaria", index=False)
        result_dict["media_semanal_vendedor_loja"].to_excel(writer, sheet_name="Vendedor Media Semanal", index=False)
        result_dict["media_mensal_loja"].to_excel(writer, sheet_name="Loja Media Mensal", index=False)
        result_dict["media_mensal_vendedor"].to_excel(writer, sheet_name="Vendedor Media Mensal", index=False)
        result_dict["wd_table"].to_excel(writer, sheet_name="Dia da Semana", index=False)
        result_dict["top_produtos_top3_dias"].to_excel(writer, sheet_name="Top Produtos Top3 Dias", index=False)

        ws_resumo = writer.sheets["Resumo"]
        ws_lojas = writer.sheets["Lojas"]
        ws_dept5 = writer.sheets["Dept Top5"]
        ws_dias = writer.sheets["Vendas Diarias"]
        ws_wd = writer.sheets["Dia da Semana"]

        ws_resumo.set_column("A:A", 45)
        ws_resumo.set_column("B:B", 22, fmt_money)
        for r in range(0, len(result_dict["resumo"])):
            label = str(result_dict["resumo"].iloc[r, 0]).lower()
            if any(k in label for k in ["cupons", "lojas", "vendedores"]):
                ws_resumo.write_number(r + 1, 1, float(result_dict["resumo"].iloc[r, 1]), fmt_int)

        ws_lojas.set_column("A:A", 18)
        ws_lojas.set_column("B:B", 18, fmt_money)
        ws_lojas.set_column("C:C", 12, fmt_int)
        ws_lojas.set_column("D:D", 20, fmt_money)

        for ws in [writer.sheets["Departamentos (100%)"], ws_dept5]:
            ws.set_column("A:A", 22)
            ws.set_column("B:B", 28)
            ws.set_column("C:C", 18, fmt_money)
            ws.set_column("D:D", 16, fmt_pct)

        ws_dias.set_column("A:A", 14)
        ws_dias.set_column("B:B", 18, fmt_money)

        ws_wd.set_column("A:A", 18)
        ws_wd.set_column("B:B", 18, fmt_money)
        ws_wd.set_column("C:C", 10, fmt_int)
        ws_wd.set_column("D:D", 18, fmt_money)
        last_row_wd = len(result_dict["wd_table"]) + 1
        ws_wd.conditional_format(f"B2:B{last_row_wd}", {'type': '3_color_scale'})

        # Gráficos Excel
        top10 = result_dict["lojas"].sort_values("faturamento", ascending=False).head(10)
        if len(top10) > 0:
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

        if len(result_dict["dept_top5"]) > 0:
            chart_pie = wb.add_chart({'type': 'pie'})
            last_row_dept5 = len(result_dict["dept_top5"]) + 1
            chart_pie.add_series({
                'name': 'Participação Top-5 Departamentos',
                'categories': f"='Dept Top5'!$B$2:$B${last_row_dept5}",
                'values':     f"='Dept Top5'!$D$2:$D${last_row_dept5}",
                'data_labels': {'percentage': True}
            })
            chart_pie.set_title({'name': 'Participação no Faturamento – Top-5 Departamentos'})
            ws_dept5.insert_chart('F2', chart_pie, {'x_scale': 1.2, 'y_scale': 1.2})

        if len(result_dict["vendas_diarias"]) > 0:
            chart_line = wb.add_chart({'type': 'line'})
            last_row_days = len(result_dict["vendas_diarias"]) + 1
            chart_line.add_series({
                'name': 'Faturamento Diário',
                'categories': f"='Vendas Diarias'!$A$2:$A${last_row_days}",
                'values':     f"='Vendas Diarias'!$B$2:$B${last_row_days}",
                'data_labels': {'value': False}
            })
            chart_line.set_title({'name': 'Vendas Diárias da Rede'})
            chart_line.set_y_axis({'name': 'R$'})
            ws_dias.insert_chart('D2', chart_line, {'x_scale': 1.2, 'y_scale': 1.2})

    buffer.seek(0)
    return buffer.getvalue()

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Indicadores Drogaria – v4.8.3 (produção)", layout="wide")
st.title("📈 Indicadores de Vendas – Rede de Drogaria")
st.caption("Versão " + APP_VERSION + " – sem Plotly e sem avisos `use_container_width`.")

uploads = st.file_uploader("Envie seus arquivos (.csv, .xlsx)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
df = None
if uploads:
    dfs = []
    for uploaded in uploads:
        if uploaded.name.lower().endswith(".csv"):
            sample = uploaded.getvalue().decode("utf-8", errors="ignore")[:5000]
            sep = detect_delimiter(sample, default=",")
            dfi = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=sep, encoding="utf-8", low_memory=False)
        else:
            xls = pd.ExcelFile(uploaded)
            aba = xls.sheet_names[0]
            dfi = pd.read_excel(xls, sheet_name=aba)
        dfs.append(dfi)
    try:
        df = pd.concat(dfs, ignore_index=True)
    except Exception:
        cols = set().union(*[set(d.columns) for d in dfs])
        aligned = []
        for d in dfs:
            for c in cols:
                if c not in d.columns:
                    d[c] = np.nan
            aligned.append(d[list(cols)])
        df = pd.concat(aligned, ignore_index=True)

if df is not None:
    st.subheader("🔧 Mapeamento de colunas (auto preenchido – ajuste se necessário)")
    display_cols = list(df.columns)
    normalized = normalize_columns(display_cols)
    norm_to_display = {n: d for n, d in zip(normalized, display_cols)}

    guesses = {
        "store_code": guess_column(normalized, ["codigo", "loja"]) or guess_column(normalized, ["cod", "loja"]),
        "seller_code": guess_column(normalized, ["codigo", "vendedor"]) or guess_column(normalized, ["cod", "vendedor"]),
        "seller_name": guess_column(normalized, ["nome", "vendedor"]),
        "value": guess_column(normalized, ["valor", "total", "venda"]) or guess_column(normalized, ["vr", "total"]) or guess_column(normalized, ["faturamento"]),
        "qty": guess_column(normalized, ["quantidade", "vendida"]) or guess_column(normalized, ["qtd", "vendida"]) or guess_column(normalized, ["cupons"]),
        "dept_code": guess_column(normalized, ["codigo", "departamento"]) or guess_column(normalized, ["cod", "departamento"]),
        "dept_name": guess_column(normalized, ["nome", "departamento"]),
        "date_col": guess_column(normalized, ["data", "venda"]) or guess_column(normalized, ["emissao"]) or guess_column(normalized, ["data"]),
        "prod_code": guess_column(normalized, ["codigo", "produto"]) or guess_column(normalized, ["cod", "produto"]) or guess_column(normalized, ["sku"]),
        "prod_name": guess_column(normalized, ["nome", "produto"]) or guess_column(normalized, ["descricao", "produto"]),
    }

    def sel(label, key, default_norm):
        options = ["(vazio)"] + normalized
        default_idx = 0
        if default_norm and default_norm in normalized:
            default_idx = options.index(default_norm) if default_norm in options else 0
        return st.selectbox(label, options, index=default_idx, key=key)

    store_code_norm = sel("**Código da Loja**", "store_code", guesses["store_code"])
    seller_code_norm = sel("**Código do Vendedor** (opcional)", "seller_code", guesses["seller_code"])
    seller_name_norm = sel("**Nome do Vendedor** (opcional)", "seller_name", guesses["seller_name"])
    value_col_norm = sel("**Valor Total Venda (R$)**", "value_col", guesses["value"])
    qty_col_norm = sel("**Quantidade Vendida**", "qty_col", guesses["qty"])
    dept_code_norm = sel("**Código de Departamento** (opcional)", "dept_code", guesses["dept_code"])
    dept_name_norm = sel("**Nome do Departamento** (opcional)", "dept_name", guesses["dept_name"])
    date_col_norm = sel("**Data da Venda** (opcional)", "date_col", guesses["date_col"])
    prod_code_norm = sel("**Código do Produto** (opcional p/ Top Produtos)", "prod_code", guesses["prod_code"])
    prod_name_norm = sel("**Nome do Produto** (opcional p/ Top Produtos)", "prod_name", guesses["prod_name"])

    pct_meta_growth = st.number_input("📈 % Meta Crescimento dos Vendedores", min_value=0.0, max_value=100.0, value=10.0, step=0.5, help="Meta = faturamento * (1 + %/100).")

    proceed = st.button("Gerar Indicadores (v4.8.3)")

    if proceed:
        mapped = {
            "store_code": norm_to_display.get(store_code_norm) if store_code_norm != "(vazio)" else None,
            "seller_code": norm_to_display.get(seller_code_norm) if seller_code_norm != "(vazio)" else None,
            "seller_name": norm_to_display.get(seller_name_norm) if seller_name_norm != "(vazio)" else None,
            "value": norm_to_display.get(value_col_norm) if value_col_norm != "(vazio)" else None,
            "qty": norm_to_display.get(qty_col_norm) if qty_col_norm != "(vazio)" else None,
            "dept_code": norm_to_display.get(dept_code_norm) if dept_code_norm != "(vazio)" else None,
            "dept_name": norm_to_display.get(dept_name_norm) if dept_name_norm != "(vazio)" else None,
            "date_col": norm_to_display.get(date_col_norm) if date_col_norm != "(vazio)" else None,
            "prod_code": norm_to_display.get(prod_code_norm) if prod_code_norm != "(vazio)" else None,
            "prod_name": norm_to_display.get(prod_name_norm) if prod_name_norm != "(vazio)" else None,
        }
        missing = [k for k in ["store_code","value","qty"] if mapped.get(k) is None]
        if missing:
            st.error("Mapeie as colunas obrigatórias: **Código da Loja**, **Valor Total Venda** e **Quantidade Vendida**.")
        else:
            result = compute_indicators(df.copy(), mapped, pct_meta_growth)

            st.subheader("📌 Resumo do Período (v4.8.3)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faturamento da Rede", fmt_brl(result["faturamento_rede"]))
            col2.metric("Total de Cupons", f"{int(result['cupons_total']):,}".replace(",", "."))
            col3.metric("Lojas Ativas", f"{int(result['lojas']['codigo_loja'].nunique())}")
            col4.metric("Vendedores com Vendas", f"{int(result['vendedores']['codigo_vendedor'].nunique()) if len(result['vendedores'])>0 else 0}")

            st.info(f"📅 Melhor dia da semana (pelo faturamento total): **{result['melhor_dia']}**" if result["melhor_dia"] else "📅 Informe a coluna de Data da Venda para análise por dia da semana.")

            st.markdown("#### 🏬 Faturamento por Loja")
            df_show(result["lojas"].sort_values("faturamento", ascending=False))

            st.markdown("#### 👤 Vendedores (com metas)")
            vend_preview = result["vendedores"].copy()
            if "% meta crescimento" in vend_preview.columns:
                vend_preview["% meta crescimento"] = vend_preview["% meta crescimento"].map(lambda x: pct_ptbr(x) if pd.notna(x) else "")
            vend_preview["meta"] = vend_preview["meta"].map(lambda x: fmt_brl(x) if pd.notna(x) else "")
            vend_preview["faturamento"] = vend_preview["faturamento"].map(lambda x: fmt_brl(x))
            df_show(vend_preview.sort_values("faturamento", ascending=False))

            if len(result["dept_top5"]) > 0:
                st.markdown("#### 🧪 Participação por Departamento – Top-5")
                dept5 = result["dept_top5"].copy()
                dept5["participacao_%"] = dept5["participacao_%"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " %" if pd.notna(x) else "")
                df_show(dept5[["codigo_departamento","nome_departamento","faturamento","participacao_%"]])

                labels = dept5["nome_departamento"].replace("", np.nan).fillna(dept5["codigo_departamento"]).tolist()
                sizes = dept5["participacao_%"].astype(float).tolist()
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct=lambda p: pct_ptbr(p), startangle=140, pctdistance=0.7)
                ax.set_title("Participação no Faturamento – Top-5 Departamentos")
                st.pyplot(fig)

            if len(result["wd_table"]) > 0:
                st.markdown("#### 🔥 Faturamento por Dia da Semana – Total e Média")
                wd_show = result["wd_table"].copy()
                wd_show["faturamento total período"] = wd_show["faturamento total período"].map(fmt_brl)
                wd_show["faturamento_medio"] = wd_show["faturamento_medio"].map(lambda x: fmt_brl(x) if pd.notna(x) else "")
                df_show(wd_show)

                data = np.array([result["wd_table"]["faturamento_medio"].fillna(0).values])
                fig_hm, ax_hm = plt.subplots()
                im = ax_hm.imshow(data, aspect="auto")
                ax_hm.set_yticks([0]); ax_hm.set_yticklabels(["Faturamento Médio"])
                ax_hm.set_xticks(np.arange(len(result["wd_table"])))
                ax_hm.set_xticklabels(result["wd_table"]["dia_semana"].tolist(), rotation=45, ha="right")
                for (j, val) in enumerate(result["wd_table"]["faturamento_medio"].values):
                    ax_hm.text(j, 0, (fmt_brl(val) if pd.notna(val) else "—"), ha="center", va="center", fontsize=9, color="white")
                ax_hm.set_title("Heatmap – Faturamento Médio por Dia da Semana")
                st.pyplot(fig_hm)

            if len(result["vendas_diarias"]) > 0:
                st.markdown("#### 📅 Vendas Diárias da Rede")
                vd = result["vendas_diarias"].copy()
                df_show(vd)

                fig2, ax2 = plt.subplots()
                ax2.plot(vd["data"], vd["faturamento_dia"])
                ax2.set_title("Vendas Diárias da Rede")
                ax2.set_xlabel("Data"); ax2.set_ylabel("R$")
                ax2.yaxis.set_major_formatter(brl_formatter())
                fig2.autofmt_xdate()
                st.pyplot(fig2)

            st.divider()
            st.markdown("### ⬇️ Download do Excel (v4.8.3)")
            excel_bytes = build_excel(df.copy(), result, pct_meta_growth)
            st.download_button(
                label="Baixar Excel (v4.8.3)",
                data=excel_bytes,
                file_name=f"Indicadores_Drogaria_v4_8_3_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
