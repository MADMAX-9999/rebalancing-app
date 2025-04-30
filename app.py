import streamlit as st
import pandas as pd
from datetime import datetime

st.title("Symulator ReBalancingu Portfela Metali Szlachetnych")

# Wczytanie danych LBMA
lbma_data = pd.read_csv("lbma_data.csv", parse_dates=["Date"])
lbma_data.set_index("Date", inplace=True)

# Parametry wejściowe
st.sidebar.header("Parametry symulacji")

initial_amount = st.sidebar.number_input("Kwota początkowej alokacji (EUR)", min_value=1000, value=100000)
monthly_purchase = st.sidebar.number_input("Kwota dokupu (EUR)", min_value=0, value=1000)
initial_date = st.sidebar.date_input("Data pierwszego zakupu", datetime(2005, 1, 3))
rebalance1_enabled = st.sidebar.checkbox("Włącz pierwszy ReBalancing", value=True)
rebalance1_date = st.sidebar.date_input("Data 1. ReBalancingu", datetime(2006, 4, 1))
rebalance2_enabled = st.sidebar.checkbox("Włącz drugi ReBalancing")
rebalance2_date = st.sidebar.date_input("Data 2. ReBalancingu", datetime(2007, 4, 1))
storage_fee = st.sidebar.number_input("Koszt magazynowania (% rocznie)", min_value=0.0, max_value=5.0, value=1.5)

markup = {
    'gold': st.sidebar.number_input("Złoto - marża zakupu (%)", value=12.0) / 100,
    'silver': st.sidebar.number_input("Srebro - marża zakupu (%)", value=15.0) / 100,
    'platinum': st.sidebar.number_input("Platyna - marża zakupu (%)", value=17.0) / 100,
    'palladium': st.sidebar.number_input("Pallad - marża zakupu (%)", value=19.0) / 100,
}
fee = {
    'gold': st.sidebar.number_input("Sprzedaż złota (%)", value=1.5) / 100,
    'silver': st.sidebar.number_input("Sprzedaż srebra (%)", value=3.0) / 100,
    'platinum': st.sidebar.number_input("Sprzedaż platyny (%)", value=3.0) / 100,
    'palladium': st.sidebar.number_input("Sprzedaż palladu (%)", value=3.0) / 100,
}
rebuy = {
    'gold': st.sidebar.number_input("Zakup złota przy ReBalancingu (%)", value=6.5) / 100,
    'silver': st.sidebar.number_input("Zakup srebra przy ReBalancingu (%)", value=6.5) / 100,
    'platinum': st.sidebar.number_input("Zakup platyny przy ReBalancingu (%)", value=6.5) / 100,
    'palladium': st.sidebar.number_input("Zakup palladu przy ReBalancingu (%)", value=6.5) / 100,
}
allocation = {'gold': 0.4, 'silver': 0.2, 'platinum': 0.2, 'palladium': 0.2}

# Inicjalizacja portfela
start_price = lbma_data[lbma_data.index >= pd.to_datetime(initial_date)].iloc[0]
initial_grams = {}
for metal in allocation:
    spot = start_price[f"{metal.capitalize()}_EUR"]
    buy_price = spot * (1 + markup[metal])
    eur_value = initial_amount * allocation[metal]
    initial_grams[metal] = eur_value / buy_price

# Symulacja
current_grams = initial_grams.copy()
portfolio_history = []
storage_cost = storage_fee / 100 * initial_amount

for year in range(initial_date.year, 2026):
    try:
        price_row = lbma_data[lbma_data.index.year == year].iloc[-1]
    except IndexError:
        continue

    if (rebalance1_enabled and year == rebalance1_date.year) or (rebalance2_enabled and year == rebalance2_date.year):
        total_value = sum(current_grams[m] * price_row[f"{m.capitalize()}_EUR"] for m in allocation)
        target_value = {m: total_value * allocation[m] for m in allocation}
        new_grams = {}
        for m in allocation:
            metal_price = price_row[f"{m.capitalize()}_EUR"]
            current_val = current_grams[m] * metal_price
            delta = target_value[m] - current_val
            if delta < 0:
                sell_price = metal_price * (1 - fee[m])
                new_grams[m] = current_grams[m] - abs(delta) / sell_price
            else:
                buy_price = metal_price * (1 + rebuy[m])
                new_grams[m] = current_grams[m] + delta / buy_price
        current_grams = new_grams

    total_value = sum(current_grams[m] * price_row[f"{m.capitalize()}_EUR"] for m in allocation)
    total_value -= storage_cost
    portfolio_history.append({
        'Rok': year,
        'Wartość portfela (EUR)': round(total_value, 2),
        'Złoto (g)': round(current_grams['gold'], 2),
        'Srebro (g)': round(current_grams['silver'], 2),
        'Platyna (g)': round(current_grams['platinum'], 2),
        'Pallad (g)': round(current_grams['palladium'], 2)
    })

st.header("Symulacja z danymi LBMA")
st.dataframe(pd.DataFrame(portfolio_history))
