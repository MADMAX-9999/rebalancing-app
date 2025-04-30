import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.title("Symulator ReBalancingu Portfela Metali Szlachetnych")

# Wczytanie danych LBMA
lbma_data = pd.read_csv("lbma_data.csv", parse_dates=["Date"])
lbma_data.set_index("Date", inplace=True)

# Parametry wejściowe
st.sidebar.header("Parametry symulacji")

initial_amount = st.sidebar.number_input("Kwota początkowej alokacji (EUR)", min_value=1000, value=100000)
periodicity = st.sidebar.selectbox("Periodyczność dokupów", ["Brak", "Tygodniowo", "Miesięcznie", "Kwartalnie"])
monthly_purchase = st.sidebar.number_input("Kwota dokupu (EUR)", min_value=0, value=1000)

if periodicity == "Tygodniowo":
    day_of_week = st.sidebar.selectbox("Dzień tygodnia dokupu", ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek"])
elif periodicity == "Miesięcznie":
    day_of_month = st.sidebar.slider("Dzień miesiąca dokupu", 1, 28, 1)
elif periodicity == "Kwartalnie":
    day_of_quarter = st.sidebar.slider("Dzień kwartału dokupu (licząc od początku kwartału)", 1, 28, 1)

initial_date = st.sidebar.date_input("Data pierwszego zakupu", datetime(2005, 1, 3))
rebalance1_enabled = st.sidebar.checkbox("Włącz pierwszy ReBalancing", value=True)
rebalance1_date = st.sidebar.date_input("Data 1. ReBalancingu", datetime(2006, 4, 1))
rebalance2_enabled = st.sidebar.checkbox("Włącz drugi ReBalancing")
rebalance2_date = st.sidebar.date_input("Data 2. ReBalancingu", datetime(2007, 4, 1))

storage_fee = st.sidebar.number_input("Koszt magazynowania (% rocznie)", min_value=0.0, max_value=5.0, value=1.5)
storage_vat = st.sidebar.number_input("VAT od magazynowania (%)", min_value=0.0, max_value=25.0, value=19.0)
storage_metal_choice = st.sidebar.selectbox("Z jakiego metalu pokrywać koszt magazynu", ["Złoto", "Srebro", "Platyna", "Pallad", "Best this year"])

markup = {
    'gold': st.sidebar.number_input("Złoto - marża zakupu (%)", value=15.6) / 100,
    'silver': st.sidebar.number_input("Srebro - marża zakupu (%)", value=18.36) / 100,
    'platinum': st.sidebar.number_input("Platyna - marża zakupu (%)", value=24.24) / 100,
    'palladium': st.sidebar.number_input("Pallad - marża zakupu (%)", value=22.49) / 100,
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

start_button = st.sidebar.button("▶️ START – uruchom symulację")

# dalszy kod obliczeń i rozliczenia kosztów magazynowania uzupełniam w kolejnym kroku, jeśli chcesz
