import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


# ============================
# Carregar modelo
# ============================
modelo = joblib.load("modelo_ibovespa.pkl")

st.title("📈 Previsão de Tendências - Ibovespa")
st.write("Selecione o período e clique em 'Atualizar dados' para visualizar as previsões e métricas.")

# ============================
# Carregar dados fixos
# ============================
df = pd.read_csv("dados_ibovespa.csv", sep=",", dayfirst=True)
df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")

# Limpar Vol. e Var%
def converter_volume(valor):
    valor = str(valor).replace(",", ".").strip()
    if valor.endswith("B"):
        return float(valor.replace("B", "")) * 1_000_000_000
    elif valor.endswith("M"):
        return float(valor.replace("M", "")) * 1_000_000
    elif valor.endswith("K"):
        return float(valor.replace("K", "")) * 1_000
    else:
        try:
            return float(valor)
        except:
            return np.nan

def converter_var_percent(valor):
    valor = str(valor).replace("%", "").replace(",", ".").strip()
    try:
        return float(valor)
    except:
        return np.nan

if "Vol." in df.columns:
    df["Vol."] = df["Vol."].apply(converter_volume)
if "Var%" in df.columns:
    df["Var%"] = df["Var%"].apply(converter_var_percent)

# ============================
# Seleção de período
# ============================
st.sidebar.header("Seleção de Período")
data_min = df["Data"].min()
data_max = df["Data"].max()

data_inicio = st.sidebar.date_input("Data inicial", data_min)
data_fim = st.sidebar.date_input("Data final", data_max)

# ============================
# Botão de atualização
# ============================
if st.sidebar.button("Atualizar dados"):

    # ============================
    # Feature engineering (igual ao treino)
    # ============================
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposicao = seasonal_decompose(df["Último"], model="additive", period=30)
    df["residuo"] = decomposicao.resid

    df["Retorno_Diario"] = df["Último"].pct_change()
    df["Retorno_Semanal"] = df["Último"].pct_change(periods=5)
    df["Volatilidade_5d"] = df["Último"].rolling(window=5).std()
    df["Volatilidade_10d"] = df["Último"].rolling(window=10).std()
    df["MM_5d"] = df["Último"].rolling(window=5).mean()
    df["MM_20d"] = df["Último"].rolling(window=20).mean()
    df["MM_60d"] = df["Último"].rolling(window=60).mean()
    df["MM_200d"] = df["Último"].rolling(window=200).mean()
    df["Delta"] = df["Último"].diff().shift(1)
    df["Return"] = df["Último"].pct_change().shift(1)

    for i in range(1, 11):
        df[f"Delta_lag{i}"] = df["Delta"].shift(i)

    df["High"] = df["Máxima"].diff().shift(1)
    df["Low"] = df["Mínima"].diff().shift(1)
    df["Spread"] = df["High"] - df["Low"]
    df["Volatilidade"] = df["Delta"].rolling(window=5).std()
    df["Open"] = df["Abertura"].diff()
    df["MA5"] = df["Delta"].rolling(window=5).mean()
    df["MA22"] = df["Delta"].rolling(window=22).mean()
    df["MA66"] = df["Delta"].rolling(window=66).mean()
    df["MA132"] = df["Delta"].rolling(window=132).mean()
    df["MA252"] = df["Delta"].rolling(window=252).mean()

    # Target para validação
    df["Target"] = (df["Último"].shift(-1) > df["Último"]).astype(int)

    # Remover NaN primeiro
    df_completo = df.dropna().copy()

    # Aplicar filtro de datas depois
    df_model = df_completo[(df_completo["Data"] >= pd.to_datetime(data_inicio)) & 
                           (df_completo["Data"] <= pd.to_datetime(data_fim))].copy()

    st.subheader("📊 Dados filtrados")
    st.write(df_model[["Data", "Último", "Abertura", "Máxima", "Mínima", "Vol.", "Var%"]].head())

    # ============================
    # Predição
    # ============================
    features = modelo.feature_names_in_
    y_pred = modelo.predict(df_model[features])
    y_proba = modelo.predict_proba(df_model[features])[:, 1]

    df_model["Previsão"] = y_pred
    df_model["Probabilidade"] = y_proba

    st.subheader("🔮 Previsões")
    st.write(df_model[["Data", "Último", "Previsão", "Probabilidade"]].head(20))

    # ============================
    # Painel de métricas
    # ============================
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_true = df_model["Target"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    st.subheader("📊 Métricas de Validação do Modelo")
    st.write(f"**Acurácia:** {acc:.2f}")
    st.write(f"**Precisão:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1-score:** {f1:.2f}")
    st.write(f"**AUC:** {auc:.2f}")

    st.markdown("### 🔎 Análise de Performance")
    if acc > 0.7 and auc > 0.7:
        st.success("O modelo apresenta boa capacidade de previsão, com equilíbrio entre precisão e recall.")
    elif acc > 0.6:
        st.warning("O modelo tem desempenho razoável, mas pode estar sofrendo com falsos positivos ou negativos.")
    else:
        st.error("O modelo apresenta baixa performance neste período. Pode ser necessário recalibrar ou treinar novamente.")

    # ============================
    # Gráfico temporal
    # ============================
    st.subheader("📈 Tendência no período selecionado")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_model["Data"], df_model["Último"], color="black", label="Preço de fechamento")
    colors = np.where(df_model["Previsão"] == 1, "tab:blue", "tab:red")
    ax.scatter(df_model["Data"], df_model["Último"], c=colors, s=30, label="Previsão (azul=alta, vermelho=queda)")
    ax.legend()
    st.pyplot(fig)