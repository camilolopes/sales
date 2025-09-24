# Indicadores – v4.10.0 (produção | compat Py3.13 + config.toml)

## Por que esta versão?
- O Streamlit Cloud estava forçando **Python 3.13**. Algumas versões antigas de libs não tinham wheels compatíveis.
- Nesta build, o `requirements.txt` usa **faixas mínimas** para que o Cloud resolva automaticamente as versões com suporte a Py3.13.
- Mantém `packages.txt` com dependências de sistema (OpenBLAS, freetype, png, etc.).
- Inclui `.streamlit/config.toml` desativando o file watcher (mais estável no Cloud).

## Como rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy no Streamlit Cloud
- Suba `app.py`, `requirements.txt`, `packages.txt` e a pasta `.streamlit/config.toml`.
- Aponte o deploy para `app.py`.
