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
    'palladium': st.sidebar.number_input("Zakup palladu przy ReBalancingu (%)", value=6.5) / 100
}

# Przycisk uruchamiający symulację (na końcu sidebar)
start_button = st.sidebar.button("▶️ START – uruchom symulację")

if start_button:
    with st.spinner("Trwa przeliczanie strategii..."):
        allocation = {'gold': 0.4, 'silver': 0.2, 'platinum': 0.2, 'palladium': 0.2}
        current_grams = {m: 0.0 for m in allocation}
        total_invested = initial_amount
        portfolio_history = []
        current_date = pd.to_datetime(initial_date)
        end_date = lbma_data.index[-1]

        # Zakup początkowy
        try:
            if current_date not in lbma_data.index:
    st.warning(f"Brak danych dla dnia: {current_date}. Pomijam ten dzień.")
    current_date += timedelta(days=1)
    continue

if current_date not in lbma_data.index:
    st.warning(f"Brak danych dla dnia: {current_date}. Pomijam ten dzień.")
    current_date += timedelta(days=1)
    continue

price_row = lbma_data.loc[current_date]
if not all(col in price_row for col in ["Gold_EUR", "Silver_EUR", "Platinum_EUR", "Palladium_EUR"]):
    st.error(f"Brakuje kolumn cenowych dla dnia {current_date}.")
    st.stop()
if not all(col in price_row for col in ["Gold_EUR", "Silver_EUR", "Platinum_EUR", "Palladium_EUR"]):
    st.error(f"Brakuje kolumn cenowych dla dnia {current_date}.")
    st.stop()
        except IndexError:
            st.error("Brak danych cenowych od daty początkowej.")
            st.stop()

        for m in allocation:
            metal_price_raw = price_row[f"{m.capitalize()}_EUR"]
if metal_price_raw == 0:
    st.warning(f"Cena {m} = 0 dla dnia {current_date}. Pomijam ten metal.")
    continue
metal_price = metal_price_raw * (1 + markup[m])
            current_grams[m] = (initial_amount * allocation[m]) / metal_price

        total_purchase = initial_amount
        years_seen = set()

        while current_date <= end_date:
            # Zakupy cykliczne
            if periodicity == "Tygodniowo" and current_date.strftime('%A') == day_of_week:
                if current_date in lbma_data.index:
                    for m in allocation:
                        price = lbma_data.loc[current_date][f"{m.capitalize()}_EUR"] * (1 + markup[m])
                        value = monthly_purchase * allocation[m]
                        current_grams[m] += value / price
                    total_invested += monthly_purchase
                    total_purchase += monthly_purchase

            elif periodicity == "Miesięcznie" and current_date.day == day_of_month:
                if current_date in lbma_data.index:
                    for m in allocation:
                        price = lbma_data.loc[current_date][f"{m.capitalize()}_EUR"] * (1 + markup[m])
                        value = monthly_purchase * allocation[m]
                        current_grams[m] += value / price
                    total_invested += monthly_purchase
                    total_purchase += monthly_purchase

            elif periodicity == "Kwartalnie" and ((current_date.month - 1) % 3 == 0) and current_date.day == day_of_quarter:
                if current_date in lbma_data.index:
                    for m in allocation:
                        price = lbma_data.loc[current_date][f"{m.capitalize()}_EUR"] * (1 + markup[m])
                        value = monthly_purchase * allocation[m]
                        current_grams[m] += value / price
                    total_invested += monthly_purchase
                    total_purchase += monthly_purchase

            # Koszt magazynowania raz w roku (ostatni dzień roboczy danego roku)
            if current_date.year not in years_seen:
                same_year = lbma_data[lbma_data.index.year == current_date.year]
                if not same_year.empty:
                    last_day = same_year.index[-1]
                    if current_date == last_day:
                        storage_cost = storage_fee / 100 * total_purchase
                        vat_cost = storage_cost * (storage_vat / 100)
                        total_cost = storage_cost + vat_cost

                        # wybór metalu do sprzedaży
                        if storage_metal_choice.lower() == "best this year":
                            year_data = same_year.iloc[[0, -1]]
                            growth = {
                                m: (year_data.iloc[1][f"{m.capitalize()}_EUR"] - year_data.iloc[0][f"{m.capitalize()}_EUR"]) / year_data.iloc[0][f"{m.capitalize()}_EUR"]
                                for m in allocation
                            }
                            metal_to_sell = max(growth, key=growth.get)
                        else:
                            metal_to_sell = storage_metal_choice.lower()

                        price_to_sell = lbma_data.loc[current_date][f"{metal_to_sell.capitalize()}_EUR"] * (1 - fee[metal_to_sell])
                        grams_to_sell = total_cost / price_to_sell
if grams_to_sell > current_grams[metal_to_sell]:
    grams_to_sell = current_grams[metal_to_sell]  # nie sprzedaj więcej niż masz
                        current_grams[metal_to_sell] -= grams_to_sell

                years_seen.add(current_date.year)

            # ReBalancing
            if (rebalance1_enabled and current_date.month == pd.to_datetime(rebalance1_date).month and current_date.day == pd.to_datetime(rebalance1_date).day and current_date >= pd.to_datetime(rebalance1_date)) or \
               (rebalance2_enabled and current_date.month == pd.to_datetime(rebalance2_date).month and current_date.day == pd.to_datetime(rebalance2_date).day and current_date >= pd.to_datetime(rebalance2_date)):
                try:
                    price_row = lbma_data.loc[current_date]
                    total_value = sum(current_grams[m] * price_row[f"{m.capitalize()}_EUR"] for m in allocation)
                    target_value = {m: total_value * allocation[m] for m in allocation}
                    new_grams = {}
                    for m in allocation:
                        metal_price = price_row[f"{m.capitalize()}_EUR"]
                        current_val = current_grams[m] * metal_price
                        delta = target_value[m] - current_val
                        if delta < 0:
                            sell_price = metal_price * (1 - fee[m])
                            grams_needed = abs(delta) / sell_price
new_grams[m] = max(current_grams[m] - grams_needed, 0)
                        else:
                            buy_price = metal_price * (1 + rebuy[m])
                            new_grams[m] = current_grams[m] + delta / buy_price
                    current_grams = new_grams
                except:
                    pass

            # Zapis roczny
            if current_date.month == 12:
                try:
                    year_row = lbma_data[lbma_data.index.year == current_date.year].iloc[-1]
                    total_value = sum(current_grams[m] * year_row[f"{m.capitalize()}_EUR"] for m in allocation)
                    st.write(f"Rok: {current_date.year}, Wartość: {round(total_value, 2)}, Kapitał: {round(total_invested, 2)}")
portfolio_history.append({
                        "Rok": current_date.year,
                        "Wartość portfela (EUR)": round(total_value, 2),
                        "Zaangażowany kapitał (EUR)": round(total_invested, 2),
                    })
                except:
                    pass

            current_date += timedelta(days=1)

        df = pd.DataFrame(portfolio_history)
        st.subheader("📊 Wyniki symulacji z historycznych danych")
        st.dataframe(df)

        st.subheader("📈 Wartość portfela vs. zaangażowany kapitał")
        fig, ax = plt.subplots()
        ax.plot(df['Rok'], df['Wartość portfela (EUR)'], marker='o', label='Wartość portfela')
        ax.plot(df['Rok'], df['Zaangażowany kapitał (EUR)'], marker='x', linestyle='--', label='Kapitał')
        ax.set_xlabel("Rok")
        ax.set_ylabel("EUR")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)




