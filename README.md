# Indicadores – v4.9.6 (produção | packages.txt + deps estáveis)

## Conteúdo
- `app.py` – aplicação Streamlit
- `requirements.txt` – dependências Python (versões estáveis)
- `packages.txt` – dependências de sistema para Streamlit Cloud
- `README.md` – instruções rápidas

## Execução local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy no Streamlit Cloud
Envie **app.py**, **requirements.txt** e **packages.txt**.


## Nota
Esta versão fixa o Python para 3.11.9 via `runtime.txt`, evitando incompatibilidades do Python 3.13 no Streamlit Cloud (pandas/numpy).
