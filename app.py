# precious_metals_portfolio_simulator.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# =========================================
# CONFIG AND INITIALIZATION
# =========================================

APP_CONFIG = {
    "page_title": "Precious Metals Portfolio Simulator",
    "layout": "wide",
    "icon": "💰",
    "version": "2.0"
}

st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    layout=APP_CONFIG["layout"],
    page_icon=APP_CONFIG["icon"]
)

# Set theme through CSS
st.markdown("""
<style>
    /* Custom theme colors */
    :root {
        --gold-color: #D4AF37;
        --silver-color: #C0C0C0;
        --platinum-color: #E5E4E2;
        --palladium-color: #CED0DD;
        --primary-color: #4F8BF9;
    }
    
    /* Metal-specific styles */
    .gold-text { color: var(--gold-color); }
    .silver-text { color: var(--silver-color); }
    .platinum-text { color: var(--platinum-color); }
    .palladium-text { color: var(--palladium-color); }
    
    /* Custom table formatting */
    .dataframe {
        font-size: 14px;
        width: 100%;
    }
    
    /* Improve metrics appearance */
    .stMetric {
        background-color: rgba(240, 242, 246, 0.5);
        border-radius: 5px;
        padding: 10px !important;
        border: 1px solid #e0e0e0;
    }
    
    /* Better containers */
    .stContainer {
        background-color: rgba(240, 242, 246, 0.2);
        border-radius: 5px;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# LANGUAGE SETTINGS AND TRANSLATIONS
# =========================================

# Initialize language in session state
if "language" not in st.session_state:
    st.session_state.language = "Polski"  # Default language

# Create a sidebar for language selection
with st.sidebar:
    st.header("🌐 Language / Sprache / Język")
    language_choice = st.selectbox(
        "",
        ("🇵🇱 Polski", "🇩🇪 Deutsch", "🇬🇧 English"),
        index=0 if st.session_state.language == "Polski" else 
             (1 if st.session_state.language == "Deutsch" else 2)
    )

# Update language selection
new_language = language_choice.split(" ")[1]
if new_language != st.session_state.language:
    st.session_state.language = new_language
    st.rerun()  # Reload the page after language change

language = st.session_state.language

# Translation dictionary (expanded with English)
translations = {
    "Polski": {
        "app_title": "Symulator ReBalancingu Portfela Metali Szlachetnych",
        "portfolio_value": "Wartość portfela",
        "real_portfolio_value": "Wartość portfela (realna, po inflacji)",
        "invested": "Zainwestowane",
        "storage_cost": "Koszty magazynowania",
        "chart_subtitle": "📈 Rozwój wartości portfela: nominalna i realna",
        "summary_title": "📊 Podsumowanie inwestycji",
        "simulation_settings": "⚙️ Parametry Symulacji",
        "investment_amounts": "💰 Inwestycja: Kwoty i daty",
        "metal_allocation": "⚖️ Alokacja metali szlachetnych (%)",
        "recurring_purchases": "🔁 Zakupy cykliczne",
        "rebalancing": "♻️ ReBalancing",
        "storage_costs": "📦 Koszty magazynowania",
        "margins_fees": "📊 Marże i prowizje",
        "buyback_prices": "💵 Ceny odkupu metali",
        "rebalance_prices": "♻️ Ceny ReBalancingu metali",
        "initial_allocation": "Kwota początkowej alokacji (EUR)",
        "first_purchase_date": "Data pierwszego zakupu",
        "last_purchase_date": "Data ostatniego zakupu",
        "purchase_frequency": "Periodyczność zakupów",
        "none": "Brak",
        "week": "Tydzień",
        "month": "Miesiąc",
        "quarter": "Kwartał",
        "purchase_day_of_week": "Dzień tygodnia zakupu",
        "purchase_day_of_month": "Dzień miesiąca zakupu (1–28)",
        "purchase_day_of_quarter": "Dzień kwartału zakupu (1–28)",
        "purchase_amount": "Kwota dokupu (EUR)",
        "rebalance_1": "ReBalancing 1",
        "rebalance_2": "ReBalancing 2",
        "deviation_condition": "Warunek odchylenia wartości",
        "deviation_threshold": "Próg odchylenia (%)",
        "start_rebalance": "Start ReBalancing",
        "monday": "Poniedziałek",
        "tuesday": "Wtorek",
        "wednesday": "Środa",
        "thursday": "Czwartek",
        "friday": "Piątek",
        "gold": "Złoto (Au)",
        "silver": "Srebro (Ag)",
        "platinum": "Platyna (Pt)",
        "palladium": "Pallad (Pd)",
        "settings_saved": "✅ Ustawienia zostały zapisane!",
        "run_simulation": "🚀 Uruchom symulację",
        "reset_allocation": "🔄 Resetuj do 40/20/20/20",
        "allocation_error": "❗ Suma alokacji: {}% – musi wynosić dokładnie 100%, aby kontynuować.",
        "price_growth": "📊 Wzrost cen metali od startu inwestycji",
        "current_holdings": "⚖️ Aktualnie posiadane ilości metali (g)",
        "capital_allocation": "💶 Alokacja kapitału",
        "sales_valuation": "📦 Wycena sprzedażowa metali",
        "purchase_value": "🛒 Wartość metali po sprzedaży",
        "difference": "📈 Różnica względem wartości portfela",
        "annual_growth": "📈 Średni roczny rozwój cen wszystkich metali razem (ważony alokacją)",
        "annual_growth_weighted": "🌐 Średni roczny wzrost cen (ważony alokacją)",
        "yearly_view": "📅 Mały uproszczony podgląd: Pierwszy dzień każdego roku",
        "storage_summary": "📦 Podsumowanie kosztów magazynowania",
        "annual_storage_cost": "Średnioroczny koszt magazynowy",
        "storage_cost_percent": "Koszt magazynowania (% ostatni rok)",
        "date_range_error": "⚠️ Zakres zakupów: tylko {:.1f} lat. (minimum 7 lat wymagane!)",
        "date_range_success": "✅ Zakres zakupów: {:.1f} lat.",
        "invested_amount": "Zainwestowane (EUR)",
        "portfolio_value_short": "Wartość portfela (EUR)",
        "action": "Akcja",
        "export_data": "💾 Eksportuj dane",
        "export_settings": "⚙️ Eksportuj ustawienia",
        "import_settings": "📥 Importuj ustawienia",
        "save_settings": "💾 Zapisz ustawienia",
        "advanced_options": "🔧 Opcje zaawansowane",
        "inflation_settings": "📈 Ustawienia inflacji",
        "custom_inflation": "Niestandardowa inflacja (%)",
        "year": "Rok",
        "calculations": "📝 Szczegółowe obliczenia",
        "total_return": "Całkowity zwrot",
        "annualized_return": "Zwrot roczny",
        "gold_margin": "Marża Gold (%)",
        "silver_margin": "Marża Silver (%)",
        "platinum_margin": "Marża Platinum (%)",
        "palladium_margin": "Marża Palladium (%)",
        "gold_buyback": "Złoto odk. od SPOT (%)",
        "silver_buyback": "Srebro odk. od SPOT (%)",
        "platinum_buyback": "Platyna odk. od SPOT (%)",
        "palladium_buyback": "Pallad odk. od SPOT (%)",
        "gold_rebalance": "Złoto ReBalancing (%)",
        "silver_rebalance": "Srebro ReBalancing (%)",
        "platinum_rebalance": "Platyna ReBalancing (%)",
        "palladium_rebalance": "Pallad ReBalancing (%)",
        "annual_storage_fee": "Roczny koszt magazynowania (%)",
        "vat": "VAT (%)",
        "storage_metal": "Metal do pokrycia kosztów",
        "best_of_year": "Najlepszy w roku",
        "all_metals": "Wszystkie proporcjonalnie",
    },
    "Deutsch": {
        "app_title": "Edelmetallportfolio ReBalancing Simulator",
        "portfolio_value": "Portfoliowert",
        "real_portfolio_value": "Portfoliowert (real, inflationsbereinigt)",
        "invested": "Investiertes Kapital",
        "storage_cost": "Lagerkosten",
        "chart_subtitle": "📈 Entwicklung des Portfoliowerts: nominal und real",
        "summary_title": "📊 Investitionszusammenfassung",
        "simulation_settings": "⚙️ Simulationseinstellungen",
        "investment_amounts": "💰 Investition: Beträge und Daten",
        "metal_allocation": "⚖️ Aufteilung der Edelmetalle (%)",
        "recurring_purchases": "🔁 Regelmäßige Käufe",
        "rebalancing": "♻️ ReBalancing",
        "storage_costs": "📦 Lagerkosten",
        "margins_fees": "📊 Margen und Gebühren",
        "buyback_prices": "💵 Rückkaufpreise der Metalle",
        "rebalance_prices": "♻️ Preise für ReBalancing der Metalle",
        "initial_allocation": "Anfangsinvestition (EUR)",
        "first_purchase_date": "Kaufstartdatum",
        "last_purchase_date": "Letzter Kauftag",
        "purchase_frequency": "Kaufhäufigkeit",
        "none": "Keine",
        "week": "Woche",
        "month": "Monat",
        "quarter": "Quartal",
        "purchase_day_of_week": "Wochentag für Kauf",
        "purchase_day_of_month": "Kauftag im Monat (1–28)",
        "purchase_day_of_quarter": "Kauftag im Quartal (1–28)",
        "purchase_amount": "Kaufbetrag (EUR)",
        "rebalance_1": "ReBalancing 1",
        "rebalance_2": "ReBalancing 2",
        "deviation_condition": "Abweichungsbedingung",
        "deviation_threshold": "Abweichungsschwelle (%)",
        "start_rebalance": "Start des ReBalancing",
        "monday": "Montag",
        "tuesday": "Dienstag",
        "wednesday": "Mittwoch",
        "thursday": "Donnerstag",
        "friday": "Freitag",
        "gold": "Gold (Au)",
        "silver": "Silber (Ag)",
        "platinum": "Platin (Pt)",
        "palladium": "Palladium (Pd)",
        "settings_saved": "✅ Einstellungen gespeichert!",
        "run_simulation": "🚀 Simulation starten",
        "reset_allocation": "🔄 Zurücksetzen auf 40/20/20/20",
        "allocation_error": "❗ Aufteilungssumme: {}% – muss genau 100% betragen, um fortzufahren.",
        "price_growth": "📊 Preissteigerung der Metalle seit Investitionsbeginn",
        "current_holdings": "⚖️ Aktuelle Metallbestände (g)",
        "capital_allocation": "💶 Kapitalallokation",
        "sales_valuation": "📦 Verkaufsbewertung der Metalle",
        "purchase_value": "🛒 Verkaufswert der Metalle",
        "difference": "📈 Differenz zum Portfoliowert",
        "annual_growth": "📈 Durchschnittliche jährliche Preisentwicklung aller Metalle (gewichtet nach Allokation)",
        "annual_growth_weighted": "🌐 Durchschnittliches jährliches Wachstum (gewichtet nach Allokation)",
        "yearly_view": "📅 Vereinfachte Übersicht: Erster Tag jedes Jahres",
        "storage_summary": "📦 Zusammenfassung der Lagerkosten",
        "annual_storage_cost": "Durchschnittliche jährliche Lagerkosten",
        "storage_cost_percent": "Lagerkosten (% letztes Jahr)",
        "date_range_error": "⚠️ Kaufzeitraum: nur {:.1f} Jahre. (Minimum 7 Jahre erforderlich!)",
        "date_range_success": "✅ Kaufzeitraum: {:.1f} Jahre.",
        "invested_amount": "Investiert (EUR)",
        "portfolio_value_short": "Portfoliowert (EUR)",
        "action": "Aktion",
        "export_data": "💾 Daten exportieren",
        "export_settings": "⚙️ Einstellungen exportieren",
        "import_settings": "📥 Einstellungen importieren",
        "save_settings": "💾 Einstellungen speichern",
        "advanced_options": "🔧 Erweiterte Optionen",
        "inflation_settings": "📈 Inflationseinstellungen",
        "custom_inflation": "Benutzerdefinierte Inflation (%)",
        "year": "Jahr",
        "calculations": "📝 Detaillierte Berechnungen",
        "total_return": "Gesamtrendite",
        "annualized_return": "Jährliche Rendite",
        "gold_margin": "Gold Marge (%)",
        "silver_margin": "Silber Marge (%)",
        "platinum_margin": "Platin Marge (%)",
        "palladium_margin": "Palladium Marge (%)",
        "gold_buyback": "Gold Rückkauf von SPOT (%)",
        "silver_buyback": "Silber Rückkauf von SPOT (%)",
        "platinum_buyback": "Platin Rückkauf von SPOT (%)",
        "palladium_buyback": "Palladium Rückkauf von SPOT (%)",
        "gold_rebalance": "Gold ReBalancing (%)",
        "silver_rebalance": "Silber ReBalancing (%)",
        "platinum_rebalance": "Platin ReBalancing (%)",
        "palladium_rebalance": "Palladium ReBalancing (%)",
        "annual_storage_fee": "Jährliche Lagerkosten (%)",
        "vat": "MwSt (%)",
        "storage_metal": "Metall zur Kostendeckung",
        "best_of_year": "Bestes des Jahres",
        "all_metals": "Alle proportional",
    },
    "English": {
        "app_title": "Precious Metals Portfolio Rebalancing Simulator",
        "portfolio_value": "Portfolio Value",
        "real_portfolio_value": "Portfolio Value (Real, Inflation-Adjusted)",
        "invested": "Invested Capital",
        "storage_cost": "Storage Costs",
        "chart_subtitle": "📈 Portfolio Value Development: Nominal and Real",
        "summary_title": "📊 Investment Summary",
        "simulation_settings": "⚙️ Simulation Settings",
        "investment_amounts": "💰 Investment: Amounts and Dates",
        "metal_allocation": "⚖️ Precious Metals Allocation (%)",
        "recurring_purchases": "🔁 Recurring Purchases",
        "rebalancing": "♻️ Rebalancing",
        "storage_costs": "📦 Storage Costs",
        "margins_fees": "📊 Margins and Fees",
        "buyback_prices": "💵 Metal Buyback Prices",
        "rebalance_prices": "♻️ Metal Rebalancing Prices",
        "initial_allocation": "Initial Allocation Amount (EUR)",
        "first_purchase_date": "First Purchase Date",
        "last_purchase_date": "Last Purchase Date",
        "purchase_frequency": "Purchase Frequency",
        "none": "None",
        "week": "Weekly",
        "month": "Monthly",
        "quarter": "Quarterly",
        "purchase_day_of_week": "Day of Week for Purchase",
        "purchase_day_of_month": "Day of Month for Purchase (1–28)",
        "purchase_day_of_quarter": "Day of Quarter for Purchase (1–28)",
        "purchase_amount": "Purchase Amount (EUR)",
        "rebalance_1": "Rebalancing 1",
        "rebalance_2": "Rebalancing 2",
        "deviation_condition": "Value Deviation Condition",
        "deviation_threshold": "Deviation Threshold (%)",
        "start_rebalance": "Start Rebalancing",
        "monday": "Monday",
        "tuesday": "Tuesday",
        "wednesday": "Wednesday",
        "thursday": "Thursday",
        "friday": "Friday",
        "gold": "Gold (Au)",
        "silver": "Silver (Ag)",
        "platinum": "Platinum (Pt)",
        "palladium": "Palladium (Pd)",
        "settings_saved": "✅ Settings saved!",
        "run_simulation": "🚀 Run Simulation",
        "reset_allocation": "🔄 Reset to 40/20/20/20",
        "allocation_error": "❗ Allocation sum: {}% – must be exactly 100% to continue.",
        "price_growth": "📊 Metal Price Growth Since Investment Start",
        "current_holdings": "⚖️ Current Metal Holdings (g)",
        "capital_allocation": "💶 Capital Allocation",
        "sales_valuation": "📦 Metals Sale Valuation",
        "purchase_value": "🛒 Metals Sale Value",
        "difference": "📈 Difference from Portfolio Value",
        "annual_growth": "📈 Average Annual Price Development of All Metals (Allocation-Weighted)",
        "annual_growth_weighted": "🌐 Average Annual Growth (Allocation-Weighted)",
        "yearly_view": "📅 Simplified Overview: First Day of Each Year",
        "storage_summary": "📦 Storage Costs Summary",
        "annual_storage_cost": "Average Annual Storage Cost",
        "storage_cost_percent": "Storage Cost (% of Last Year)",
        "date_range_error": "⚠️ Purchase range: only {:.1f} years. (minimum 7 years required!)",
        "date_range_success": "✅ Purchase range: {:.1f} years.",
        "invested_amount": "Invested (EUR)",
        "portfolio_value_short": "Portfolio Value (EUR)",
        "action": "Action",
        "export_data": "💾 Export Data",
        "export_settings": "⚙️ Export Settings",
        "import_settings": "📥 Import Settings",
        "save_settings": "💾 Save Settings",
        "advanced_options": "🔧 Advanced Options",
        "inflation_settings": "📈 Inflation Settings",
        "custom_inflation": "Custom Inflation (%)",
        "year": "Year",
        "calculations": "📝 Detailed Calculations",
        "total_return": "Total Return",
        "annualized_return": "Annualized Return",
        "gold_margin": "Gold Margin (%)",
        "silver_margin": "Silver Margin (%)",
        "platinum_margin": "Platinum Margin (%)",
        "palladium_margin": "Palladium Margin (%)",
        "gold_buyback": "Gold Buyback from SPOT (%)",
        "silver_buyback": "Silver Buyback from SPOT (%)",
        "platinum_buyback": "Platinum Buyback from SPOT (%)",
        "palladium_buyback": "Palladium Buyback from SPOT (%)",
        "gold_rebalance": "Gold Rebalancing (%)",
        "silver_rebalance": "Silver Rebalancing (%)",
        "platinum_rebalance": "Platinum Rebalancing (%)",
        "palladium_rebalance": "Palladium Rebalancing (%)",
        "annual_storage_fee": "Annual Storage Fee (%)",
        "vat": "VAT (%)",
        "storage_metal": "Metal for Cost Coverage",
        "best_of_year": "Best of Year",
        "all_metals": "All Proportionally",
    }
}

# Helper function to get translation
def t(key):
    """Get translation for a key in the current language"""
    return translations.get(language, {}).get(key, key)

# =========================================
# DATA LOADING FUNCTIONS
# =========================================

@st.cache_data
def load_data():
    """Load and preprocess the LBMA price data"""
    try:
        df = pd.read_csv("lbma_data.csv", parse_dates=True, index_col=0)
        df = df.sort_index()
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error loading LBMA data: {e}")
        # Return sample data if file not found (for testing)
        return generate_sample_data()

@st.cache_data
def load_inflation_data():
    """Load and preprocess inflation data"""
    try:
        df = pd.read_csv("inflacja.csv", sep=";", encoding="cp1250")
        df = df[["Rok", "Wartość"]].copy()
        df["Wartość"] = df["Wartość"].str.replace(",", ".").astype(float)
        df["Inflacja (%)"] = df["Wartość"] - 100
        return df[["Rok", "Inflacja (%)"]]
    except Exception as e:
        st.error(f"Error loading inflation data: {e}")
        # Return sample data if file not found
        return generate_sample_inflation_data()

def generate_sample_data():
    """Generate sample data for testing purposes"""
    index = pd.date_range(start='2000-01-01', end='2025-01-01', freq='B')
    data = pd.DataFrame(index=index)
    
    # Generate sample price data with realistic trends
    np.random.seed(42)
    
    # Base values
    base_values = {
        'Gold': 400,
        'Silver': 5,
        'Platinum': 350, 
        'Palladium': 200
    }
    
    # Trends and volatility
    for metal, base in base_values.items():
        # Create a growing trend with volatility
        trend = np.linspace(0, 4, len(index))  # Base trend multiplier from 1x to 5x
        noise = np.random.normal(0, 0.2, len(index))  # Daily noise
        seasonal = 0.2 * np.sin(np.linspace(0, 20*np.pi, len(index)))  # Seasonal pattern
        
        # Combine trend, noise and seasonality
        series = base * (1 + trend + noise + seasonal)
        
        # EUR price
        data[f'{metal}_EUR'] = series
        
        # USD price (slightly different)
        data[f'{metal}_USD'] = series * (1 + np.random.normal(0, 0.05, len(index)))
    
    return data

def generate_sample_inflation_data():
    """Generate sample inflation data"""
    years = range(2000, 2026)
    inflation = np.random.normal(2.5, 1.0, len(years))  # Mean 2.5%, std 1%
    
    df = pd.DataFrame({
        "Rok": years,
        "Inflacja (%)": inflation
    })
    return df

# =========================================
# SETTINGS MANAGEMENT
# =========================================

def save_settings_to_file(settings, filename="precious_metals_settings.json"):
    """Save current settings to a JSON file"""
    try:
        settings_path = Path(filename)
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=4, default=str)
        return True
    except Exception as e:
        st.error(f"Error saving settings: {e}")
        return False

def load_settings_from_file(filename="precious_metals_settings.json"):
    """Load settings from a JSON file"""
    try:
        settings_path = Path(filename)
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            return settings
        return None
    except Exception as e:
        st.error(f"Error loading settings: {e}")
        return None

# =========================================
# HELPER FUNCTIONS
# =========================================

def generate_purchase_dates(start_date, freq, day, end_date):
    """
    Generate purchase dates based on frequency and day settings
    
    Args:
        start_date: Initial purchase date
        freq: Frequency (None, 'Week', 'Month', 'Quarter')
        day: Day of week/month/quarter
        end_date: Last possible purchase date
        
    Returns:
        List of purchase dates (datetime objects)
    """
    dates = []
    current = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Convert day names to day numbers if necessary
    if isinstance(day, str):
        day_mapping = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, 
            "Thursday": 3, "Friday": 4
        }
        day = day_mapping.get(day, 0)
    
    if freq == t("week"):
        while current <= end_date:
            # Find next day of the week
            days_ahead = (day - current.weekday()) % 7
            current = current + timedelta(days=days_ahead)
            if current <= end_date:
                dates.append(current)
            current += timedelta(weeks=1)
    
    elif freq == t("month"):
        while current <= end_date:
            # Set to day of month, handling month lengths
            try:
                current = current.replace(day=min(day, 28))
            except ValueError:
                # End of month handling
                last_day = pd.Timestamp(current.year, current.month, 1) + pd.offsets.MonthEnd()
                current = last_day
                
            if current <= end_date:
                dates.append(current)
            current += pd.DateOffset(months=1)
    
    elif freq == t("quarter"):
        while current <= end_date:
            # Set to first month of quarter
            quarter_month = 3 * ((current.month - 1) // 3) + 1
            try:
                current = current.replace(month=quarter_month, day=min(day, 28))
            except ValueError:
                # End of month handling
                last_day = pd.Timestamp(current.year, quarter_month, 1) + pd.offsets.MonthEnd()
                current = last_day
                
            if current <= end_date:
                dates.append(current)
            current += pd.DateOffset(months=3)
    
    # Map dates to nearest available dates in the dataset
    available_dates = []
    for d in dates:
        nearest_idx = data.index.get_indexer([d], method="nearest")
        if len(nearest_idx) > 0 and nearest_idx[0] >= 0:
            available_dates.append(data.index[nearest_idx[0]])
            
    return available_dates

def find_best_metal_of_year(start_date, end_date):
    """Find the metal with the best performance in a given period"""
    start_prices = data.loc[start_date]
    end_prices = data.loc[end_date]
    growth = {}
    
    for metal in ["Gold", "Silver", "Platinum", "Palladium"]:
        start_price = start_prices[f"{metal}_EUR"]
        end_price = end_prices[f"{metal}_EUR"]
        growth[metal] = (end_price / start_price) - 1
    
    return max(growth, key=growth.get)

def format_currency(amount, currency="EUR"):
    """Format amount as currency"""
    return f"{amount:,.2f} {currency}"

# =========================================
# SIMULATION LOGIC
# =========================================

def simulate_portfolio(
    allocation,
    initial_allocation,
    initial_date,
    end_purchase_date,
    purchase_freq,
    purchase_day,
    purchase_amount,
    rebalance_settings,
    storage_settings,
    margins,
    buyback_discounts,
    rebalance_markup,
    inflation_data=None
):
    """
    Simulate precious metals portfolio over time with rebalancing and storage costs
    
    Args:
        allocation: Dictionary of metal allocations (percentage as decimal)
        initial_allocation: Initial investment amount
        initial_date: Start date of investment
        end_purchase_date: End date for purchases
        purchase_freq: Frequency of purchases
        purchase_day: Day for purchases
        purchase_amount: Amount for recurring purchases
        rebalance_settings: Dictionary of rebalancing settings
        storage_settings: Dictionary of storage fee settings
        margins: Dictionary of purchase margins
        buyback_discounts: Dictionary of buyback discounts
        rebalance_markup: Dictionary of rebalancing markups
        inflation_data: Optional custom inflation data
        
    Returns:
        DataFrame with portfolio simulation results
    """
    # Initialize portfolio
    portfolio = {metal: 0.0 for metal in allocation}
    history = []
    invested = 0.0
    
    # Define date range
    all_dates = data.loc[initial_date:end_purchase_date].index
    
    # Generate purchase dates
    purchase_dates = generate_purchase_dates(initial_date, purchase_freq, purchase_day, end_purchase_date)
    
    # Initialize tracking variables
    last_year = None
    last_rebalance_dates = {
        "rebalance_1": None,
        "rebalance_2": None
    }
    
    # Helper function to apply rebalancing
    def apply_rebalance(date, label, condition_enabled, threshold_percent):
        """Apply portfolio rebalancing on given date if conditions are met"""
        # Check minimum time since last rebalance
        min_days_between_rebalances = 30
        last_date = last_rebalance_dates.get(label)
        
        if last_date is not None and (date - last_date).days < min_days_between_rebalances:
            return f"rebalancing_skipped_{label}_too_soon"
        
        # Calculate current portfolio value and allocation
        prices = data.loc[date]
        total_value = sum(prices[m + "_EUR"] * portfolio[m] for m in allocation)
        
        if total_value == 0:
            return f"rebalancing_skipped_{label}_no_value"
        
        # Calculate current allocation percentages
        current_shares = {
            m: (prices[m + "_EUR"] * portfolio[m]) / total_value
            for m in allocation
        }
        
        # Check if deviation exceeds threshold
        rebalance_trigger = False
        for metal in allocation:
            deviation = abs(current_shares[metal] - allocation[metal]) * 100
            if deviation >= threshold_percent:
                rebalance_trigger = True
                break
        
        # Skip if condition enabled but threshold not met
        if condition_enabled and not rebalance_trigger:
            return f"rebalancing_skipped_{label}_no_deviation"
        
        # Calculate target values
        target_value = {m: total_value * allocation[m] for m in allocation}
        
        # Perform rebalancing
        for metal in allocation:
            current_value = prices[metal + "_EUR"] * portfolio[metal]
            diff = current_value - target_value[metal]
            
            if diff > 0:  # Need to sell this metal
                sell_price = prices[metal + "_EUR"] * (1 + buyback_discounts[metal] / 100)
                grams_to_sell = min(diff / sell_price, portfolio[metal])
                portfolio[metal] -= grams_to_sell
                cash = grams_to_sell * sell_price
                
                # Distribute to other metals that need it
                for buy_metal in allocation:
                    needed_value = target_value[buy_metal] - prices[buy_metal + "_EUR"] * portfolio[buy_metal]
                    if needed_value > 0:
                        buy_price = prices[buy_metal + "_EUR"] * (1 + rebalance_markup[buy_metal] / 100)
                        buy_grams = min(cash / buy_price, needed_value / buy_price)
                        portfolio[buy_metal] += buy_grams
                        cash -= buy_grams * buy_price
                        if cash <= 0:
                            break
        
        # Update last rebalance date
        last_rebalance_dates[label] = date
        return label
    
    # Make initial purchase
    initial_ts = data.index[data.index.get_indexer([pd.to_datetime(initial_date)], method="nearest")][0]
    prices = data.loc[initial_ts]
    
    for metal, percent in allocation.items():
        price = prices[metal + "_EUR"] * (1 + margins[metal] / 100)
        grams = (initial_allocation * percent) / price
        portfolio[metal] += grams
    
    invested += initial_allocation
    history.append((initial_ts, invested, dict(portfolio), "initial"))
    
    # Simulate through all dates
    for date in all_dates:
        actions = []
        
        # Handle recurring purchases
        if date in purchase_dates:
            prices = data.loc[date]
            for metal, percent in allocation.items():
                price = prices[metal + "_EUR"] * (1 + margins[metal] / 100)
                grams = (purchase_amount * percent) / price
                portfolio[metal] += grams
            invested += purchase_amount
            actions.append("recurring")
        
        # Check for first rebalancing
        if (rebalance_settings["rebalance_1"] and 
            date >= pd.to_datetime(rebalance_settings["rebalance_1_start"]) and 
            date.month == rebalance_settings["rebalance_1_start"].month and 
            date.day == rebalance_settings["rebalance_1_start"].day):
            
            actions.append(apply_rebalance(
                date, 
                "rebalance_1", 
                rebalance_settings["rebalance_1_condition"], 
                rebalance_settings["rebalance_1_threshold"]
            ))
        
        # Check for second rebalancing
        if (rebalance_settings["rebalance_2"] and 
            date >= pd.to_datetime(rebalance_settings["rebalance_2_start"]) and 
            date.month == rebalance_settings["rebalance_2_start"].month and 
            date.day == rebalance_settings["rebalance_2_start"].day):
            
            actions.append(apply_rebalance(
                date, 
                "rebalance_2", 
                rebalance_settings["rebalance_2_condition"], 
                rebalance_settings["rebalance_2_threshold"]
            ))
        
        # Handle year change (storage fees)
        if last_year is None:
            last_year = date.year
            
        if date.year != last_year:
            last_year_end = data.loc[data.index[data.index.year == last_year]].index[-1]
            storage_cost = invested * (storage_settings["storage_fee"] / 100) * (1 + storage_settings["vat"] / 100)
            prices_end = data.loc[last_year_end]
            
            # Apply storage costs based on selected metal
            if storage_settings["storage_metal"] == "Best of year":
                metal_to_sell = find_best_metal_of_year(
                    data.index[data.index.year == last_year][0],
                    data.index[data.index.year == last_year][-1]
                )
                sell_price = prices_end[metal_to_sell + "_EUR"] * (1 + buyback_discounts[metal_to_sell] / 100)
                grams_needed = storage_cost / sell_price
                grams_needed = min(grams_needed, portfolio[metal_to_sell])
                portfolio[metal_to_sell] -= grams_needed
            
            elif storage_settings["storage_metal"] == "ALL":
                total_value = sum(prices_end[m + "_EUR"] * portfolio[m] for m in allocation)
                for metal in allocation:
                    share = (prices_end[metal + "_EUR"] * portfolio[metal]) / total_value
                    cash_needed = storage_cost * share
                    sell_price = prices_end[metal + "_EUR"] * (1 + buyback_discounts[metal] / 100)
                    grams_needed = cash_needed / sell_price
                    grams_needed = min(grams_needed, portfolio[metal])
                    portfolio[metal] -= grams_needed
            
            else:  # Single metal
                metal = storage_settings["storage_metal"]
                sell_price = prices_end[metal + "_EUR"] * (1 + buyback_discounts[metal] / 100)
                grams_needed = storage_cost / sell_price
                grams_needed = min(grams_needed, portfolio[metal])
                portfolio[metal] -= grams_needed
            
            history.append((last_year_end, invested, dict(portfolio), "storage_fee"))
            last_year = date.year
        
        # Record actions if any occurred
        if actions:
            history.append((date, invested, dict(portfolio), ", ".join(actions)))
    
    # Create result dataframe
    result = pd.DataFrame([
        {
            "Date": h[0],
            "Invested": h[1],
            **{m: h[2][m] for m in allocation},
            "Portfolio Value": sum(
                data.loc[h[0]][m + "_EUR"] * (1 + buyback_discounts[m] / 100) * h[2][m]
                for m in allocation
            ),
            "Action": h[3]
        } for h in history
    ]).set_index("Date")
    
    # Add real (inflation-adjusted) portfolio value
    if inflation_data is not None:
        inflation_dict = dict(zip(inflation_data["Rok"], inflation_data["Inflacja (%)"]))
        start_year = result.index.min().year
        
        real_values = []
        for date in result.index:
            nominal_value = result.loc[date, "Portfolio Value"]
            current_year = date.year
            
            # Calculate cumulative inflation factor
            cumulative_factor = 1.0
            for year in range(start_year, current_year + 1):
                inflation = inflation_dict.get(year, 0.0) / 100
                cumulative_factor *= (1 + inflation)
            
            real_value = nominal_value / cumulative_factor if cumulative_factor != 0 else nominal_value
            real_values.append(real_value)
        
        result["Portfolio Value Real"] = real_values
    
    # Add storage cost column
    result["Storage Cost"] = 0.0
    storage_cost_dates = result[result["Action"] == "storage_fee"].index
    
    for date in storage_cost_dates:
        result.at[date, "Storage Cost"] = result.at[date, "Invested"] * (storage_settings["storage_fee"] / 100) * (1 + storage_settings["vat"] / 100)
    
    return result

# =========================================
# VISUALIZATION FUNCTIONS
# =========================================

def create_portfolio_chart(result_df, language):
    """Create interactive plotly chart for portfolio values"""
    # Prepare data for chart
    chart_data = result_df.copy()
    
    # Ensure numeric values
    for col in ["Portfolio Value", "Portfolio Value Real", "Invested", "Storage Cost"]:
        if col in chart_data.columns:
            chart_data[col] = pd.to_numeric(chart_data[col], errors="coerce").fillna(0)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Portfolio Value"],
        mode='lines',
        name=t("portfolio_value"),
        line=dict(color='#FFD700', width=4)
    ))
    
    # Add real portfolio value line if available
    if "Portfolio Value Real" in chart_data.columns:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data["Portfolio Value Real"],
            mode='lines',
            name=t("real_portfolio_value"),
            line=dict(color='#FF0000', width=1, dash='dash')
        ))
    
    # Add invested capital line
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Invested"],
        mode='lines',
        name=t("invested"),
        line=dict(color='#808080', width=2)
    ))
    
    # Add storage costs if significant
    if chart_data["Storage Cost"].max() > 0:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data["Storage Cost"],
            mode='markers',
            name=t("storage_cost"),
            marker=dict(color='#DC3545', size=8, symbol='triangle-down')
        ))
    
    # Update layout
    fig.update_layout(
        title=t("chart_subtitle"),
        xaxis_title="Date",
        yaxis_title="EUR",
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def create_allocation_pie_chart(portfolio, prices, language):
    """Create pie chart showing current metal allocation"""
    labels = [t("gold"), t("silver"), t("platinum"), t("palladium")]
    metals = ["Gold", "Silver", "Platinum", "Palladium"]
    colors = ['#D4AF37', '#C0C0C0', '#E5E4E2', '#CED0DD']
    
    values = [prices[m + "_EUR"] * portfolio[m] for m in metals]
    total = sum(values)
    percentages = [v/total*100 if total > 0 else 0 for v in values]
    
    labels_with_pct = [f"{label} ({pct:.1f}%)" for label, pct in zip(labels, percentages)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels_with_pct,
        values=values,
        hole=.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title=t("current_holdings"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    
    return fig

# =========================================
# MAIN APP LAYOUT
# =========================================

# Load data
data = load_data()
inflation_real = load_inflation_data()

# App header
st.title(t("app_title"))
st.markdown(f"v{APP_CONFIG['version']}")
st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header(t("simulation_settings"))
    
    # Investment amounts and dates
    st.subheader(t("investment_amounts"))
    
    initial_allocation = st.number_input(
        t("initial_allocation"),
        value=100000.0,
        step=100.0
    )
    
    today = datetime.today()
    default_initial_date = today.replace(year=today.year - 20)
    
    initial_date = st.date_input(
        t("first_purchase_date"),
        value=default_initial_date.date(),
        min_value=data.index.min().date(),
        max_value=data.index.max().date()
    )
    
    # Calculate minimum end date (initial_date + 7 years)
    min_end_date = (pd.to_datetime(initial_date) + pd.DateOffset(years=7)).date()
    if min_end_date > data.index.max().date():
        min_end_date = data.index.max().date()
    
    end_purchase_date = st.date_input(
        t("last_purchase_date"),
        value=data.index.max().date(),
        min_value=min_end_date,
        max_value=data.index.max().date()
    )
    
    # Calculate investment period
    days_difference = (pd.to_datetime(end_purchase_date) - pd.to_datetime(initial_date)).days
    years_difference = days_difference / 365.25
    
    # Validate date range
    if years_difference >= 7:
        st.success(t("date_range_success").format(years_difference))
        dates_valid = True
    else:
        st.error(t("date_range_error").format(years_difference))
        dates_valid = False
    
    # Metal allocation
    st.subheader(t("metal_allocation"))
    
    # Initialize allocation in session state
    for metal, default in {"Gold": 40, "Silver": 20, "Platinum": 20, "Palladium": 20}.items():
        if f"alloc_{metal}" not in st.session_state:
            st.session_state[f"alloc_{metal}"] = default
    
    # Reset button
    if st.button(t("reset_allocation")):
        st.session_state["alloc_Gold"] = 40
        st.session_state["alloc_Silver"] = 20
        st.session_state["alloc_Platinum"] = 20
        st.session_state["alloc_Palladium"] = 20
        st.rerun()
    
    # Allocation sliders
    allocation_gold = st.slider(t("gold"), 0, 100, key="alloc_Gold")
    allocation_silver = st.slider(t("silver"), 0, 100, key="alloc_Silver")
    allocation_platinum = st.slider(t("platinum"), 0, 100, key="alloc_Platinum")
    allocation_palladium = st.slider(t("palladium"), 0, 100, key="alloc_Palladium")
    
    # Calculate total allocation
    total_allocation = allocation_gold + allocation_silver + allocation_platinum + allocation_palladium
    
    # Allocation dictionary
    allocation = {
        "Gold": allocation_gold / 100,
        "Silver": allocation_silver / 100,
        "Platinum": allocation_platinum / 100,
        "Palladium": allocation_palladium / 100
    }
    
    # Recurring purchases
    st.subheader(t("recurring_purchases"))
    
    purchase_freq = st.selectbox(
        t("purchase_frequency"),
        [t("none"), t("week"), t("month"), t("quarter")],
        index=1
    )
    
    if purchase_freq == t("week"):
        days_of_week = [t("monday"), t("tuesday"), t("wednesday"), t("thursday"), t("friday")]
        selected_day = st.selectbox(t("purchase_day_of_week"), days_of_week, index=0)
        purchase_day = selected_day
        default_purchase_amount = 250.0
    elif purchase_freq == t("month"):
        purchase_day = st.number_input(t("purchase_day_of_month"), min_value=1, max_value=28, value=1)
        default_purchase_amount = 1000.0
    elif purchase_freq == t("quarter"):
        purchase_day = st.number_input(t("purchase_day_of_quarter"), min_value=1, max_value=28, value=1)
        default_purchase_amount = 3250.0
    else:
        purchase_day = None
        default_purchase_amount = 0.0
    
    purchase_amount = st.number_input(t("purchase_amount"), value=default_purchase_amount, step=50.0)
    
    # Rebalancing settings
    st.subheader(t("rebalancing"))
    
    # Default rebalancing dates based on initial date
    rebalance_base_year = initial_date.year + 1
    rebalance_1_default = datetime(rebalance_base_year, 4, 1)
    rebalance_2_default = datetime(rebalance_base_year, 10, 1)
    
    # First rebalancing
    rebalance_1 = st.checkbox(t("rebalance_1"), value=True)
    rebalance_1_condition = st.checkbox(t("deviation_condition") + " 1", value=False)
    rebalance_1_threshold = st.number_input(
        t("deviation_threshold") + " 1",
        min_value=0.0,
        max_value=100.0,
        value=12.0,
        step=0.5
    )
    
    rebalance_1_start = st.date_input(
        t("start_rebalance") + " 1",
        value=rebalance_1_default.date(),
        min_value=data.index.min().date(),
        max_value=data.index.max().date()
    )
    
    # Second rebalancing
    rebalance_2 = st.checkbox(t("rebalance_2"), value=False)
    rebalance_2_condition = st.checkbox(t("deviation_condition") + " 2", value=False)
    rebalance_2_threshold = st.number_input(
        t("deviation_threshold") + " 2",
        min_value=0.0,
        max_value=100.0,
        value=12.0,
        step=0.5
    )
    
    rebalance_2_start = st.date_input(
        t("start_rebalance") + " 2",
        value=rebalance_2_default.date(),
        min_value=data.index.min().date(),
        max_value=data.index.max().date()
    )
    
    # Rebalancing settings dictionary
    rebalance_settings = {
        "rebalance_1": rebalance_1,
        "rebalance_1_condition": rebalance_1_condition,
        "rebalance_1_threshold": rebalance_1_threshold,
        "rebalance_1_start": rebalance_1_start,
        "rebalance_2": rebalance_2,
        "rebalance_2_condition": rebalance_2_condition,
        "rebalance_2_threshold": rebalance_2_threshold,
        "rebalance_2_start": rebalance_2_start
    }
    
    # Storage costs
    st.subheader(t("storage_costs"))
    
    storage_fee = st.number_input(t("annual_storage_fee"), value=1.5)
    vat = st.number_input(t("vat"), value=19.0)
    storage_metal = st.selectbox(
        t("storage_metal"),
        ["Gold", "Silver", "Platinum", "Palladium", t("best_of_year"), "ALL"]
    )
    
    # Storage settings dictionary
    storage_settings = {
        "storage_fee": storage_fee,
        "vat": vat,
        "storage_metal": storage_metal
    }
    
    # Margins and fees
    st.subheader(t("margins_fees"))
    
    with st.expander(t("margins_fees"), expanded=False):
        margins = {
            "Gold": st.number_input(t("gold_margin"), value=15.6),
            "Silver": st.number_input(t("silver_margin"), value=18.36),
            "Platinum": st.number_input(t("platinum_margin"), value=24.24),
            "Palladium": st.number_input(t("palladium_margin"), value=22.49)
        }
    
    # Buyback prices
    st.subheader(t("buyback_prices"))
    
    with st.expander(t("buyback_prices"), expanded=False):
        buyback_discounts = {
            "Gold": st.number_input(t("gold_buyback"), value=-1.5, step=0.1),
            "Silver": st.number_input(t("silver_buyback"), value=-3.0, step=0.1),
            "Platinum": st.number_input(t("platinum_buyback"), value=-3.0, step=0.1),
            "Palladium": st.number_input(t("palladium_buyback"), value=-3.0, step=0.1)
        }
    
    # Rebalancing prices
    st.subheader(t("rebalance_prices"))
    
    with st.expander(t("rebalance_prices"), expanded=False):
        rebalance_markup = {
            "Gold": st.number_input(t("gold_rebalance"), value=6.5, step=0.1),
            "Silver": st.number_input(t("silver_rebalance"), value=6.5, step=0.1),
            "Platinum": st.number_input(t("platinum_rebalance"), value=6.5, step=0.1),
            "Palladium": st.number_input(t("palladium_rebalance"), value=6.5, step=0.1)
        }
    
    # Advanced options
    st.subheader(t("advanced_options"))
    
    with st.expander(t("inflation_settings"), expanded=False):
        use_custom_inflation = st.checkbox("Use custom inflation", value=False)
        if use_custom_inflation:
            custom_inflation = {}
            with st.container():
                for year in range(initial_date.year, end_purchase_date.year + 1):
                    custom_inflation[year] = st.number_input(
                        f"{t('year')} {year} - {t('custom_inflation')}",
                        value=2.5,
                        step=0.1
                    )
    
    # Settings export/import
    with st.expander(t("export_settings"), expanded=False):
        # Collect all settings
        all_settings = {
            "allocation": allocation,
            "initial_allocation": initial_allocation,
            "purchase_freq": purchase_freq,
            "purchase_day": purchase_day,
            "purchase_amount": purchase_amount,
            "rebalance_settings": rebalance_settings,
            "storage_settings": storage_settings,
            "margins": margins,
            "buyback_discounts": buyback_discounts,
            "rebalance_markup": rebalance_markup
        }
        
        if st.button(t("save_settings")):
            if save_settings_to_file(all_settings):
                st.success(t("settings_saved"))
    
    with st.expander(t("import_settings"), expanded=False):
        if st.button(t("import_settings")):
            imported_settings = load_settings_from_file()
            if imported_settings:
                # Update session state with imported settings
                st.success("Settings imported successfully!")
                st.rerun()
    
    # Simulation button
    start_simulation = st.button(
        t("run_simulation"),
        disabled=not dates_valid or total_allocation != 100
    )

# Main content area
if total_allocation != 100:
    st.error(t("allocation_error").format(total_allocation))
    st.stop()

# Run simulation if button clicked or parameters changed
if start_simulation or 'result' not in st.session_state:
    with st.spinner("Running simulation..."):
        result = simulate_portfolio(
            allocation=allocation,
            initial_allocation=initial_allocation,
            initial_date=initial_date,
            end_purchase_date=end_purchase_date,
            purchase_freq=purchase_freq,
            purchase_day=purchase_day,
            purchase_amount=purchase_amount,
            rebalance_settings=rebalance_settings,
            storage_settings=storage_settings,
            margins=margins,
            buyback_discounts=buyback_discounts,
            rebalance_markup=rebalance_markup,
            inflation_data=inflation_real
        )
        st.session_state.result = result

# Use cached result if available
result = st.session_state.result

# Portfolio value chart
fig = create_portfolio_chart(result, language)
st.plotly_chart(fig, use_container_width=True)

# Results Summary
st.subheader(t("summary_title"))

# Get dates and calculate investment period
start_date = result.index.min()
end_date = result.index.max()
years = (end_date - start_date).days / 365.25

# Calculate performance metrics
capital_invested = result["Invested"].max()
portfolio_value = result["Portfolio Value"].iloc[-1]

if capital_invested > 0 and years > 0:
    annual_return = (portfolio_value / capital_invested) ** (1 / years) - 1
else:
    annual_return = 0.0

# Metal price growth section
st.subheader(t("price_growth"))

start_prices = data.loc[start_date]
end_prices = data.loc[end_date]

growth_metrics = {}
for metal in ["Gold", "Silver", "Platinum", "Palladium"]:
    start_price = start_prices[metal + "_EUR"]
    end_price = end_prices[metal + "_EUR"]
    growth_pct = (end_price / start_price - 1) * 100
    growth_metrics[metal] = growth_pct

# Display growth metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(t("gold"), f"{growth_metrics['Gold']:.2f}%", delta=f"{growth_metrics['Gold']:.1f}%")
with col2:
    st.metric(t("silver"), f"{growth_metrics['Silver']:.2f}%", delta=f"{growth_metrics['Silver']:.1f}%")
with col3:
    st.metric(t("platinum"), f"{growth_metrics['Platinum']:.2f}%", delta=f"{growth_metrics['Platinum']:.1f}%")
with col4:
    st.metric(t("palladium"), f"{growth_metrics['Palladium']:.2f}%", delta=f"{growth_metrics['Palladium']:.1f}%")

# Current metal holdings
st.subheader(t("current_holdings"))

# Get final holdings
final_holdings = {
    "Gold": result.iloc[-1]["Gold"],
    "Silver": result.iloc[-1]["Silver"],
    "Platinum": result.iloc[-1]["Platinum"],
    "Palladium": result.iloc[-1]["Palladium"]
}

# Define metal colors
metal_colors = {
    "Gold": "#D4AF37",
    "Silver": "#C0C0C0",
    "Platinum": "#E5E4E2",
    "Palladium": "#CED0DD"
}

# Two-column layout: holdings metrics and pie chart
col1, col2 = st.columns([3, 2])

with col1:
    # Display metals in a grid
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<h4 style='color:{metal_colors['Gold']}; text-align: center;'>{t('gold')}</h4>", unsafe_allow_html=True)
        st.metric(label="", value=f"{final_holdings['Gold']:.2f} g")
        
        st.markdown(f"<h4 style='color:{metal_colors['Platinum']}; text-align: center;'>{t('platinum')}</h4>", unsafe_allow_html=True)
        st.metric(label="", value=f"{final_holdings['Platinum']:.2f} g")
    
    with c2:
        st.markdown(f"<h4 style='color:{metal_colors['Silver']}; text-align: center;'>{t('silver')}</h4>", unsafe_allow_html=True)
        st.metric(label="", value=f"{final_holdings['Silver']:.2f} g")
        
        st.markdown(f"<h4 style='color:{metal_colors['Palladium']}; text-align: center;'>{t('palladium')}</h4>", unsafe_allow_html=True)
        st.metric(label="", value=f"{final_holdings['Palladium']:.2f} g")

with col2:
    # Create pie chart of current portfolio value distribution
    pie_fig = create_allocation_pie_chart(final_holdings, data.loc[end_date], language)
    st.plotly_chart(pie_fig, use_container_width=True)

# 📊 Podsumowanie inwestycji – wersja uproszczona

st.subheader(t("summary_title"))

# Obliczenia podstawowe
start_date = result.index.min()
end_date = result.index.max()
years = (end_date - start_date).days / 365.25

capital_invested = result["Invested"].max()
portfolio_value = result["Portfolio Value"].iloc[-1]

# Oblicz średni roczny wzrost cen (ważony alokacją)
weighted_start_price = sum(
    allocation[metal] * data.loc[start_date][metal + "_EUR"]
    for metal in ["Gold", "Silver", "Platinum", "Palladium"]
)

weighted_end_price = sum(
    allocation[metal] * data.loc[end_date][metal + "_EUR"]
    for metal in ["Gold", "Silver", "Platinum", "Palladium"]
)

if weighted_start_price > 0 and years > 0:
    weighted_avg_annual_growth = (weighted_end_price / weighted_start_price) ** (1 / years) - 1
else:
    weighted_avg_annual_growth = 0.0

# Wyświetlanie w 1 kolumnie
col1 = st.container()

with col1:
    st.metric(t("capital_allocation"), format_currency(capital_invested))
    st.metric(t("purchase_value"), format_currency(portfolio_value))
    st.metric(
    t("annual_growth_weighted"), 
    f"{weighted_avg_annual_growth * 100:.2f}%",
    delta=f"{weighted_avg_annual_growth * 100:.1f}%"
    )

# Show yearly summary table
st.subheader(t("yearly_view"))

# Group by year and take first day
result_yearly = result.groupby(result.index.year).first()

# Create a simple table with selected columns
yearly_table = pd.DataFrame({
    t("invested_amount"): result_yearly["Invested"].round(0),
    t("portfolio_value_short"): result_yearly["Portfolio Value"].round(0),
    t("gold"): result_yearly["Gold"].round(2),
    t("silver"): result_yearly["Silver"].round(2),
    t("platinum"): result_yearly["Platinum"].round(2),
    t("palladium"): result_yearly["Palladium"].round(2),
    t("action"): result_yearly["Action"]
})

# Format currency columns
yearly_table[t("invested_amount")] = yearly_table[t("invested_amount")].map(lambda x: f"{x:,.0f} EUR")
yearly_table[t("portfolio_value_short")] = yearly_table[t("portfolio_value_short")].map(lambda x: f"{x:,.0f} EUR")

# Display table with alternating row colors using custom CSS
st.dataframe(yearly_table, use_container_width=True)

# Storage costs summary
st.subheader(t("storage_summary"))

# Filter for storage fee entries
storage_costs = result[result["Action"] == "storage_fee"]

# Calculate total storage cost
total_storage_cost = result["Storage Cost"].sum()

# Calculate percentage of current portfolio value
if portfolio_value > 0:
    storage_pct = (avg_annual_storage_cost / portfolio_value) * 100
else:
    storage_pct = 0.0

# Display in columns
col1, col2 = st.columns(2)
with col1:
    st.metric(t("annual_storage_cost"), format_currency(avg_annual_storage_cost))
with col2:
    st.metric(t("storage_cost_percent"), f"{storage_pct:.2f}%")

# Export data option
with st.expander(t("export_data"), expanded=False):
    # Create download button for CSV
    csv = result.to_csv()
    st.download_button(
        label="CSV Export",
        data=csv,
        file_name="precious_metals_simulation.csv",
        mime="text/csv"
    )
    
    # Display raw data table with pagination
    st.dataframe(result)

# Footer with disclaimer
st.markdown("---")
st.caption("Disclaimer: This simulation is for educational purposes only. Past performance does not guarantee future results.")

# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":
    # App is already running through Streamlit's script execution
    pass
