# Indicadores – v4.10.3 (produção estável | base 4.10.0 + pizza % total | compat Py3.13)

## O que muda nesta build de estabilização
- Baseada na **v4.10.0** (sua última estável), com a melhoria da **pizza exibindo % do total** (igual à tabela) no app e no Excel.
- Preparada para o Streamlit Cloud que está iniciando com **Python 3.13.7** (conforme log).
  - `requirements.txt` pinado em versões **compatíveis com Py3.13**.
- Mantém `packages.txt` e `.streamlit/config.toml`. Inclui `runtime.txt` (caso o Cloud passe a respeitar).

## Deploy
1) Suba `app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml` e `runtime.txt` (raiz do repo).
2) Em **Manage app → Reboot** e **Clear cache** para rebuild limpo.
3) App deve subir com as versões pinadas (compatíveis com Py3.13).

