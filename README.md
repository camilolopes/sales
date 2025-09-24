# Indicadores – v4.10.2 (estável | base 4.10.0 + pizza % do total | Py3.11 pinado)

Baseada na sua última versão **estável (4.10.0)**, mantendo o comportamento e adicionando a melhoria:
- **Gráfico de pizza mostra o mesmo % da tabela** (participação sobre o total da rede).

## Produção (Streamlit Cloud)
- Use **Python 3.11.9** (forçado por `runtime.txt`).
- Dependências **pinadas** em `requirements.txt` (conjunto validado).
- Inclui `packages.txt` (OpenBLAS/freetype/png) e `.streamlit/config.toml`.

## Rodar local
```bash
pip install -r requirements.txt
streamlit run app.py
```
