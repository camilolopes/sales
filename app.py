
# app.py
# -*- coding: utf-8 -*-
# v4.10.4 (estável | base 4.10.0 + pizza % do total | Py3.11 pinado)

import io, re, csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

APP_VERSION = "v4.10.4 (estável | base 4.10.0 + pizza % do total | Py3.11 pinado)"

# ---------------- Utils ----------------
@st.cache_data(show_spinner=False)
def detect_delimiter(sample_text: str, default=","):
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;|\t")
        return dialect.delimiter
    except Exception:
        return default

def normalize_columns(cols):
    import unicodedata as ud
    out = []
    for c in cols:
        c = ud.normalize("NFKD", str(c)).encode("ascii", "ignore").decode("utf-8").lower().strip()
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
    return "R$ " + format(value, ",.2f").replace(",", "X").replace(".", ",").replace("X", ".")

def brl_formatter():
    return FuncFormatter(lambda x, pos: fmt_brl(x))

def pct_ptbr_num(x: float) -> str:
    return format(x, ",.2f").replace(",", "X").replace(".", ",").replace("X", ".")

# ---------------- Cálculos ----------------
def compute(df: pd.DataFrame, cols: dict, pct_meta: float):
    vcol = cols["value"]; qcol = cols["qty"]; store_col = cols["store_code"]
    seller_code = cols.get("seller_code"); seller_name = cols.get("seller_name")
    dept_code = cols.get("dept_code"); dept_name = cols.get("dept_name")
    date_col = cols.get("date_col")

    df[vcol] = to_float_series_robust(df[vcol]).fillna(0.0)
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce").fillna(0).astype(float)

    if date_col and date_col in df.columns:
        df["_data_venda"] = parse_date_series(df[date_col])
    else:
        df["_data_venda"] = pd.NaT

    faturamento_rede = float(df[vcol].sum())
    total_cupons = float(df[qcol].sum())

    fat_loja = df.groupby(store_col, dropna=False)[vcol].sum().reset_index().rename(columns={store_col:"codigo_loja", vcol:"faturamento"})
    cupons_loja = df.groupby(store_col, dropna=False)[qcol].sum().reset_index().rename(columns={store_col:"codigo_loja", qcol:"cupons"})
    lojas = fat_loja.merge(cupons_loja, on="codigo_loja", how="outer").fillna(0)
    lojas["ticket_medio_loja"] = np.where(lojas["cupons"]>0, lojas["faturamento"]/lojas["cupons"], np.nan)

    if seller_code and seller_code in df.columns:
        if seller_name and seller_name in df.columns:
            vendedores = df.groupby([seller_code, seller_name], dropna=False)[vcol].sum().reset_index()
            vendedores.columns = ["codigo_vendedor","nome_vendedor","faturamento"]
        else:
            vendedores = df.groupby(seller_code, dropna=False)[vcol].sum().reset_index()
            vendedores["nome_vendedor"] = ""
            vendedores.columns = ["codigo_vendedor","faturamento","nome_vendedor"]
            vendedores = vendedores[["codigo_vendedor","nome_vendedor","faturamento"]]
        vendedores["% meta crescimento"] = pct_meta
        vendedores["meta"] = vendedores["faturamento"] * (1 + pct_meta/100.0)
    else:
        vendedores = pd.DataFrame(columns=["codigo_vendedor","nome_vendedor","faturamento","% meta crescimento","meta"])

    if dept_code and dept_code in df.columns:
        dept = df.groupby(dept_code, dropna=False)[vcol].sum().reset_index()
        dept.columns = ["codigo_departamento","faturamento"]
        if dept_name and dept_name in df.columns:
            names = df[[dept_code, dept_name]].dropna().drop_duplicates(subset=[dept_code])
            names.columns = ["codigo_departamento","nome_departamento"]
            dept = dept.merge(names, on="codigo_departamento", how="left")
        else:
            dept["nome_departamento"] = ""
        denom = faturamento_rede if faturamento_rede!=0 else np.nan
        dept["participacao_frac"] = np.where(denom>0, dept["faturamento"]/denom, np.nan)
        dept["participacao_%"] = dept["participacao_frac"]*100.0
        dept = dept.sort_values("participacao_%", ascending=False)
        dept_top5 = dept.head(5).copy()
        dept_top5["label_total"] = dept_top5.apply(
            lambda r: "{} – {} %".format(
                (r["nome_departamento"] if (isinstance(r["nome_departamento"], str) and r["nome_departamento"].strip()) else r["codigo_departamento"]),
                pct_ptbr_num(r["participacao_%"]) if pd.notna(r["participacao_%"]) else ""
            ), axis=1
        )
    else:
        dept = pd.DataFrame(columns=["codigo_departamento","nome_departamento","faturamento","participacao_frac","participacao_%"])
        dept_top5 = dept.copy()

    if pd.api.types.is_datetime64_any_dtype(df["_data_venda"]):
        vendas_diarias = df.groupby(df["_data_venda"].dt.date, dropna=False)[vcol].sum().reset_index().rename(columns={0:"faturamento_dia"})
        vendas_diarias.columns = ["data","faturamento_dia"]
    else:
        vendas_diarias = pd.DataFrame(columns=["data","faturamento_dia"])

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

    return dict(
        faturamento_rede=faturamento_rede,
        total_cupons=total_cupons,
        lojas=lojas,
        vendedores=vendedores,
        dept=dept,
        dept_top5=dept_top5,
        vendas_diarias=vendas_diarias,
        resumo=resumo
    )

def build_excel(df, result):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Base", index=False)

        wb = writer.book
        fmt_money = wb.add_format({'num_format': 'R$ #,##0.00'})
        fmt_int = wb.add_format({'num_format': '0'})
        fmt_pct = wb.add_format({'num_format': '0.00%'})

        res = result["resumo"].copy()
        res.to_excel(writer, sheet_name="Resumo", index=False)
        ws_resumo = writer.sheets["Resumo"]
        ws_resumo.set_column("A:A", 50)
        ws_resumo.set_column("B:B", 24, fmt_money)

        lojas = result["lojas"].sort_values("faturamento", ascending=False)
        lojas.to_excel(writer, sheet_name="Lojas", index=False)
        ws_lojas = writer.sheets["Lojas"]
        ws_lojas.set_column("A:A", 18)
        ws_lojas.set_column("B:B", 18, fmt_money)
        ws_lojas.set_column("C:C", 12, fmt_int)
        ws_lojas.set_column("D:D", 20, fmt_money)

        vend = result["vendedores"].copy().sort_values("faturamento", ascending=False).reset_index(drop=True)
        vend.to_excel(writer, sheet_name="Vendedores", index=False)
        ws_vend = writer.sheets["Vendedores"]
        ws_vend.set_column("A:A", 22)
        ws_vend.set_column("B:B", 28)
        ws_vend.set_column("C:C", 18, fmt_money)
        ws_vend.set_column("D:D", 18)
        ws_vend.set_column("E:E", 18, fmt_money)
        for i in range(2, len(vend)+2):
            ws_vend.write_formula(i-1, 4, "=C{row}*(1 + D{row}/100)".format(row=i))

        dept5 = result["dept_top5"].copy()
        dept5.to_excel(writer, sheet_name="Dept Top5", index=False)
        ws_dept5 = writer.sheets["Dept Top5"]
        ws_dept5.set_column("A:A", 22)
        ws_dept5.set_column("B:B", 28)
        ws_dept5.set_column("C:C", 18, fmt_money)
        ws_dept5.set_column("D:D", 16, fmt_pct)
        ws_dept5.set_column("E:E", 30)  # label_total

        # Pizza com rótulo = % do total (usa categoria = label_total)
        if len(dept5) > 0:
            last_row = len(dept5) + 1
            chart_pie = wb.add_chart({'type': 'pie'})
            chart_pie.add_series({
                'name': 'Participação no Faturamento – Top-5 (rótulo = % do total)',
                'categories': "='Dept Top5'!$E$2:$E${}".format(last_row),
                'values':     "='Dept Top5'!$C$2:$C${}".format(last_row),
                'data_labels': {'category': True, 'percentage': False, 'value': False}
            })
            chart_pie.set_title({'name': 'Participação no Faturamento – Top-5 Departamentos'})
            ws_dept5.insert_chart('G2', chart_pie, {'x_scale': 1.2, 'y_scale': 1.2})

        vd = result["vendas_diarias"].copy()
        vd.to_excel(writer, sheet_name="Vendas Diarias", index=False)
        ws_dias = writer.sheets["Vendas Diarias"]
        ws_dias.set_column("A:A", 14)
        ws_dias.set_column("B:B", 18, fmt_money)

    buffer.seek(0)
    return buffer.getvalue()

# ---------------- UI ----------------
st.set_page_config(page_title="Indicadores Drogaria – v4.10.4 (estável | base 4.10.0 + pizza % do total | Py3.11 pinado)", layout="wide")
st.title("📈 Indicadores de Vendas – Rede de Drogaria")
st.caption("Versão " + APP_VERSION)

uploads = st.file_uploader("Envie seus arquivos (.csv, .xlsx)", type=["csv","xlsx","xls"], accept_multiple_files=True)

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
            dfi = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
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
    st.subheader("🔧 Mapeamento de colunas (auto) — ajuste se necessário")
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

    pct_meta = st.number_input("📈 % Meta Crescimento dos Vendedores", min_value=0.0, max_value=100.0, value=10.0, step=0.5)

    if st.button("Gerar Indicadores (v4.10.4)"):
        mapped = {
            "store_code": norm_to_display.get(store_code_norm) if store_code_norm != "(vazio)" else None,
            "seller_code": norm_to_display.get(seller_code_norm) if seller_code_norm != "(vazio)" else None,
            "seller_name": norm_to_display.get(seller_name_norm) if seller_name_norm != "(vazio)" else None,
            "value": norm_to_display.get(value_col_norm) if value_col_norm != "(vazio)" else None,
            "qty": norm_to_display.get(qty_col_norm) if qty_col_norm != "(vazio)" else None,
            "dept_code": norm_to_display.get(dept_code_norm) if dept_code_norm != "(vazio)" else None,
            "dept_name": norm_to_display.get(dept_name_norm) if dept_name_norm != "(vazio)" else None,
            "date_col": norm_to_display.get(date_col_norm) if date_col_norm != "(vazio)" else None,
        }
        missing = [k for k in ["store_code","value","qty"] if mapped.get(k) is None]
        if missing:
            st.error("Mapeie as colunas obrigatórias: **Código da Loja**, **Valor Total Venda** e **Quantidade Vendida**.")
        else:
            result = compute(df.copy(), mapped, pct_meta)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faturamento da Rede", fmt_brl(result["faturamento_rede"]))
            c2.metric("Total de Cupons", "{:,}".format(int(result['total_cupons'])).replace(",", "."))
            c3.metric("Lojas Ativas", str(int(result['lojas']['codigo_loja'].nunique())))
            c4.metric("Vendedores com Vendas", str(int(result['vendedores']['codigo_vendedor'].nunique()) if len(result['vendedores'])>0 else 0))

            st.markdown("#### 🏬 Faturamento por Loja")
            show_lojas = result["lojas"].sort_values("faturamento", ascending=False).copy()
            show_lojas["faturamento"] = show_lojas["faturamento"].map(fmt_brl)
            show_lojas["ticket_medio_loja"] = show_lojas["ticket_medio_loja"].map(lambda x: fmt_brl(x) if np.isfinite(x) else "")
            st.dataframe(show_lojas, width='stretch')

            st.markdown("#### 🧪 Participação por Departamento – Top-5 (rótulo = % do total)")
            dept5 = result["dept_top5"].copy()
            if len(dept5) > 0:
                tbl = dept5[["codigo_departamento","nome_departamento","faturamento","participacao_%"]].copy()
                tbl["participacao_%"] = tbl["participacao_%"].map(lambda x: pct_ptbr_num(x) + " %")
                st.dataframe(tbl, width='stretch')

                labels = dept5["label_total"].tolist()
                sizes = dept5["faturamento"].astype(float).tolist()
                fig, ax = plt.subplots()
                wedges, _ = ax.pie(sizes, startangle=140)
                ax.legend(wedges, labels, title="Departamento – % do total", loc="center left", bbox_to_anchor=(1, 0.5))
                ax.set_title("Participação no Faturamento – Top-5 (rótulo = % do total)")
                st.pyplot(fig)

            if len(result["vendas_diarias"]) > 0:
                st.markdown("#### 📅 Vendas Diárias da Rede")
                vd = result["vendas_diarias"].copy()
                vd_show = vd.copy(); vd_show["faturamento_dia"] = vd_show["faturamento_dia"].map(fmt_brl)
                st.dataframe(vd_show, width='stretch')

            st.divider()
            st.markdown("### ⬇️ Download do Excel (v4.10.4)")
            excel_bytes = build_excel(df.copy(), result)
            st.download_button(
                label="Baixar Excel (v4.10.4)",
                data=excel_bytes,
                file_name="Indicadores_Drogaria_v4_10_2_{}.xlsx".format(datetime.now().strftime('%Y%m%d_%H%M')),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
