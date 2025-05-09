# precious_metals_portfolio_simulator.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
import json
import os
import base64
from datetime import datetime, timedelta
from pathlib import Path


# Definicja presetów
PRESETS = {
    "SSW": {
        "name": {
            "Polski": "SSW",
            "Deutsch": "SSW",
            "English": "SSW"
        },
        "description": {
            "Polski": "Standardowe ustawienia symulatora",
            "Deutsch": "Standardeinstellungen des Simulators",
            "English": "Standard simulator settings"
        },
        # Nie zmieniamy ustawień domyślnych - zachowując obecne wartości
        "allocation": {
            "Gold": 40,
            "Silver": 20,
            "Platinum": 20,
            "Palladium": 20
        },
        "margins": {
            "Gold": 15.6,
            "Silver": 18.36,
            "Platinum": 24.24,
            "Palladium": 22.49
        },
        "buyback_discounts": {
            "Gold": -1.5,
            "Silver": -3.0,
            "Platinum": -3.0,
            "Palladium": -3.0
        },
        "storage_settings": {
            "storage_fee": 1.5,
            "vat": 19.0,
            "storage_metal": "Gold",
            "storage_frequency": "Annual"
        }
    },
    "Auvesta": {
        "name": {
            "Polski": "Auvesta",
            "Deutsch": "Auvesta",
            "English": "Auvesta"
        },
        "description": {
            "Polski": "Ustawienia bazujące na ofercie Auvesta",
            "Deutsch": "Einstellungen basierend auf dem Auvesta-Angebot",
            "English": "Settings based on Auvesta offer"
        },
        "allocation": {
            "Gold": 40,
            "Silver": 30,
            "Platinum": 15,
            "Palladium": 15
        },
        "margins": {
            "Gold": 11.0,
            "Silver": 16.0,
            "Platinum": 17.0,
            "Palladium": 21.0
        },
        "buyback_discounts": {
            "Gold": 0.0,
            "Silver": 0.0,
            "Platinum": 0.0,
            "Palladium": 0.0
        },
        "storage_settings": {
            "storage_fee": 0.96,
            "vat": 19.0,
            "storage_metal": "ALL",
            "storage_frequency": "Monthly"
        }
    },
    "custom": {
        "name": {
            "Polski": "Własne ustawienia",
            "Deutsch": "Benutzerdefinierte Einstellungen",
            "English": "Custom settings"
        },
        "description": {
            "Polski": "Twoje własne ustawienia",
            "Deutsch": "Ihre eigenen Einstellungen",
            "English": "Your own custom settings"
        },
        "allocation": None,
        "margins": None,
        "buyback_discounts": None,
        "storage_settings": None
    }
}

# Funkcja do zastosowania presetu
def apply_preset(preset_key):
    """Apply selected preset to session state"""
    if preset_key not in PRESETS or preset_key == "custom":
        return False
    
    preset = PRESETS[preset_key]
    
    # Apply allocation
    if preset["allocation"]:
        for metal, value in preset["allocation"].items():
            st.session_state[f"alloc_{metal}"] = value
    
    # Apply margins
    if preset["margins"]:
        for metal, value in preset["margins"].items():
            key = f"margin_{metal.lower()}"
            if key in st.session_state:
                st.session_state[key] = value
    
    # Apply buyback discounts
    if preset["buyback_discounts"]:
        for metal, value in preset["buyback_discounts"].items():
            key = f"buyback_{metal.lower()}"
            if key in st.session_state:
                st.session_state[key] = value
    
    # Apply storage settings
    if preset["storage_settings"]:
        if "storage_fee" in preset["storage_settings"]:
            st.session_state["storage_fee"] = preset["storage_settings"]["storage_fee"]
        if "vat" in preset["storage_settings"]:
            st.session_state["vat"] = preset["storage_settings"]["vat"]
        if "storage_metal" in preset["storage_settings"]:
            st.session_state["storage_metal"] = preset["storage_settings"]["storage_metal"]
        if "storage_frequency" in preset["storage_settings"]:
            frequency_map = {"Annual": 0, "Quarterly": 1, "Monthly": 2}
            if preset["storage_settings"]["storage_frequency"] in frequency_map:
                st.session_state["storage_frequency"] = frequency_map[preset["storage_settings"]["storage_frequency"]]
    
    return True







# =========================================
# CONFIG AND INITIALIZATION
# =========================================

APP_CONFIG = {
    "page_title": "Precious Metals Portfolio Simulator",
    "layout": "wide",
    "icon": "💰",
    "version": "2.1"  # Increased version number
}

st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    layout=APP_CONFIG["layout"],
    page_icon=APP_CONFIG["icon"]
)

# Set theme through CSS with improved styling
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
    
    /* Improved table formatting */
    .dataframe {
        font-size: 14px;
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: #f0f2f6;
        padding: 8px;
        border-bottom: 2px solid #ddd;
    }
    
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    /* Improve metrics appearance */
    .stMetric {
        background-color: rgba(240, 242, 246, 0.5);
        border-radius: 5px;
        padding: 10px !important;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    
    .stMetric:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Better containers */
    .stContainer {
        background-color: rgba(240, 242, 246, 0.2);
        border-radius: 5px;
        padding: 10px !important;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# LANGUAGE SETTINGS AND TRANSLATIONS
# =========================================

# Improved: Load translations from external JSON file if available
def load_translations():
    try:
        with open("translations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to hardcoded translations if file not found or invalid
        return {
            "Polski": {
                "app_title": "Symulator ReBalancingu Portfela Metali Szlachetnych",
                "portfolio_value": "Wartość portfela",
                # ... other translations
            },
            "Deutsch": {
                "app_title": "Edelmetallportfolio ReBalancing Simulator",
                # ... other translations
            },
            "English": {
                "app_title": "Precious Metals Portfolio Rebalancing Simulator",
                # ... other translations
            }
        }

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

# Load translations
translations = load_translations()

# Helper function to get translation
def t(key):
    """Get translation for a key in the current language"""
    return translations.get(language, {}).get(key, key)

# =========================================
# DATA LOADING FUNCTIONS WITH IMPROVED ERROR HANDLING
# =========================================

@st.cache_data
def load_data():
    """Load and preprocess the LBMA price data with improved error handling"""
    try:
        # First try to load from the data directory
        data_paths = ["lbma_data.csv", "data/lbma_data.csv", "../data/lbma_data.csv"]
        
        for path in data_paths:
            try:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
                df = df.sort_index()
                df = df.dropna()
                return df
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue
                
        # If all paths fail, generate sample data
        st.warning("LBMA data file not found. Using generated sample data.")
        return generate_sample_data()
        
    except Exception as e:
        st.error(f"Error loading LBMA data: {e}")
        # Return sample data if file not found (for testing)
        return generate_sample_data()

@st.cache_data
def load_inflation_data():
    """Load and preprocess inflation data with improved error handling"""
    try:
        # First try to load from the data directory
        data_paths = ["inflacja.csv", "data/inflacja.csv", "../data/inflacja.csv"]
        
        for path in data_paths:
            try:
                df = pd.read_csv(path, sep=";", encoding="cp1250")
                df = df[["Rok", "Wartość"]].copy()
                df["Wartość"] = df["Wartość"].str.replace(",", ".").astype(float)
                df["Inflacja (%)"] = df["Wartość"] - 100
                return df[["Rok", "Inflacja (%)"]]
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue
        
        # If all paths fail, generate sample data
        st.warning("Inflation data file not found. Using generated sample data.")
        return generate_sample_inflation_data()
        
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
        with open(settings_path, 'w', encoding='utf-8') as f:
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
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            return settings
        return None
    except Exception as e:
        st.error(f"Error loading settings: {e}")
        return None

# New function to save simulation results to PDF
def create_pdf_download_link(fig, portfolio_data, filename="precious_metals_report.pdf"):
    """Create a download link for a PDF report of the simulation"""
    try:
        import io
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        import plotly.io as pio
        
        # Convert plotly figure to image
        img_bytes = pio.to_image(fig, format="png")
        
        # Create PDF buffer
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Precious Metals Portfolio Simulation Report")
        
        # Add date
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Add chart
        c.drawImage(io.BytesIO(img_bytes), 50, height - 400, width=500, height=300)
        
        # Add portfolio summary
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 430, "Portfolio Summary")
        
        c.setFont("Helvetica", 10)
        y_pos = height - 450
        for i, (key, value) in enumerate(portfolio_data.items()):
            c.drawString(50, y_pos - (i * 20), f"{key}: {value}")
        
        c.save()
        buffer.seek(0)
        
        # Convert to download link
        b64 = base64.b64encode(buffer.read()).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    except ImportError:
        return "PDF generation requires ReportLab package. Install with: pip install reportlab"
    except Exception as e:
        return f"Error generating PDF: {e}"

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
    try:
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
    except Exception as e:
        st.error(f"Error generating purchase dates: {e}")
        return []

def find_best_metal_of_year(start_date, end_date):
    """Find the metal with the best performance in a given period"""
    try:
        start_prices = data.loc[start_date]
        end_prices = data.loc[end_date]
        growth = {}
        
        for metal in ["Gold", "Silver", "Platinum", "Palladium"]:
            start_price = start_prices[f"{metal}_EUR"]
            end_price = end_prices[f"{metal}_EUR"]
            growth[metal] = (end_price / start_price) - 1
        
        return max(growth, key=growth.get)
    except Exception as e:
        st.warning(f"Could not determine best metal: {e}")
        return "Gold"  # Default to Gold on error

def format_currency(amount, currency="EUR"):
    """Format amount as currency"""
    return f"{amount:,.2f} {currency}"

# =========================================
# SIMULATION LOGIC - REFACTORED FOR BETTER READABILITY
# =========================================

def process_initial_purchase(initial_date, allocation, initial_allocation, margins):
    """Process the initial purchase and return the portfolio and invested amount"""
    portfolio = {metal: 0.0 for metal in allocation}
    
    initial_ts = data.index[data.index.get_indexer([pd.to_datetime(initial_date)], method="nearest")][0]
    prices = data.loc[initial_ts]
    
    for metal, percent in allocation.items():
        price = prices[metal + "_EUR"] * (1 + margins[metal] / 100)
        grams = (initial_allocation * percent) / price
        portfolio[metal] += grams
    
    return portfolio, initial_allocation, initial_ts

def process_recurring_purchase(date, portfolio, allocation, margins, purchase_amount):
    """Process a recurring purchase and return the updated portfolio and amount invested"""
    prices = data.loc[date]
    additional_investment = 0
    
    for metal, percent in allocation.items():
        price = prices[metal + "_EUR"] * (1 + margins[metal] / 100)
        grams = (purchase_amount * percent) / price
        portfolio[metal] += grams
        additional_investment += purchase_amount * percent
    
    return portfolio, additional_investment

def apply_rebalance(date, portfolio, allocation, prices, label, condition_enabled, 
                   threshold_percent, buyback_discounts, rebalance_markup, last_rebalance_dates):
    """Apply portfolio rebalancing on given date if conditions are met"""
    # Check minimum time since last rebalance
    min_days_between_rebalances = 30
    last_date = last_rebalance_dates.get(label)
    
    if last_date is not None and (date - last_date).days < min_days_between_rebalances:
        return portfolio, f"rebalancing_skipped_{label}_too_soon"
    
    # Calculate current portfolio value and allocation
    total_value = sum(prices[m + "_EUR"] * portfolio[m] for m in allocation)
    
    if total_value == 0:
        return portfolio, f"rebalancing_skipped_{label}_no_value"
    
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
        return portfolio, f"rebalancing_skipped_{label}_no_deviation"
    
    # Calculate target values
    target_value = {m: total_value * allocation[m] for m in allocation}
    
    # Perform rebalancing
    cash = 0
    for metal in allocation:
        current_value = prices[metal + "_EUR"] * portfolio[metal]
        diff = current_value - target_value[metal]
        
        if diff > 0:  # Need to sell this metal
            sell_price = prices[metal + "_EUR"] * (1 + buyback_discounts[metal] / 100)
            grams_to_sell = min(diff / sell_price, portfolio[metal])
            portfolio[metal] -= grams_to_sell
            cash += grams_to_sell * sell_price
    
    # Distribute cash to metals that need it
    for metal in allocation:
        current_value = prices[metal + "_EUR"] * portfolio[metal]
        diff = target_value[metal] - current_value
        
        if diff > 0 and cash > 0:  # Need to buy this metal
            buy_price = prices[metal + "_EUR"] * (1 + rebalance_markup[metal] / 100)
            buy_grams = min(cash / buy_price, diff / buy_price)
            portfolio[metal] += buy_grams
            cash -= buy_grams * buy_price
    
    # Update last rebalance date in the caller's dictionary
    last_rebalance_dates[label] = date
    return portfolio, label

def apply_storage_costs(date, portfolio, invested, storage_settings, buyback_discounts, last_year):
    """Apply annual storage costs by selling metals"""
    last_year_end = data.loc[data.index[data.index.year == last_year]].index[-1]
    storage_cost = invested * (storage_settings["storage_fee"] / 100) * (1 + storage_settings["vat"] / 100)
    prices_end = data.loc[last_year_end]
    
    # Apply storage costs based on selected metal
    if storage_settings["storage_metal"] == t("best_of_year"):
        metal_to_sell = find_best_metal_of_year(
            data.index[data.index.year == last_year][0],
            data.index[data.index.year == last_year][-1]
        )
        sell_price = prices_end[metal_to_sell + "_EUR"] * (1 + buyback_discounts[metal_to_sell] / 100)
        grams_needed = storage_cost / sell_price
        grams_needed = min(grams_needed, portfolio[metal_to_sell])
        portfolio[metal_to_sell] -= grams_needed
    
    elif storage_settings["storage_metal"] == "ALL":
        total_value = sum(prices_end[m + "_EUR"] * portfolio[m] for m in portfolio)
        if total_value > 0:
            for metal in portfolio:
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
    
    return portfolio, storage_cost, last_year_end

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
    inflation_data=None,
    storage_frequency="Annual"  # New parameter
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
        storage_frequency: Frequency of storage fee application
        
    Returns:
        DataFrame with portfolio simulation results
    """
    try:
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
        last_month = None
        last_quarter = None
        
        last_rebalance_dates = {
            "rebalance_1": None,
            "rebalance_2": None
        }
        
        # Make initial purchase
        portfolio, initial_invested, initial_ts = process_initial_purchase(
            initial_date, allocation, initial_allocation, margins
        )
        
        invested += initial_invested
        history.append((initial_ts, invested, dict(portfolio), "initial"))
        
        # Determine relevant dates for storage fee application
        storage_dates = set()
        if storage_frequency == "Annual":
            storage_dates = {d for d in all_dates if d.month == 12 and d.day >= 28}
        elif storage_frequency == "Quarterly":
            storage_dates = {d for d in all_dates if d.month in [3, 6, 9, 12] and d.day >= 28}
        elif storage_frequency == "Monthly":
            storage_dates = {d for d in all_dates if d.day >= 28}
        
        # Simulate through all dates
        for date in all_dates:
            actions = []
            
            # Handle recurring purchases
            if date in purchase_dates:
                portfolio, additional = process_recurring_purchase(
                    date, portfolio, allocation, margins, purchase_amount
                )
                invested += additional
                actions.append("recurring")
            
            # Check for second rebalancing
            if (rebalance_settings["rebalance_2"] and 
                date >= pd.to_datetime(rebalance_settings["rebalance_2_start"]) and 
                date.month == rebalance_settings["rebalance_2_start"].month and 
                date.day == rebalance_settings["rebalance_2_start"].day):
                
                portfolio, rebalance_action = apply_rebalance(
                    date, 
                    portfolio,
                    allocation,
                    data.loc[date],
                    "rebalance_2", 
                    rebalance_settings["rebalance_2_condition"], 
                    rebalance_settings["rebalance_2_threshold"],
                    buyback_discounts,
                    rebalance_markup,
                    last_rebalance_dates
                )
                
                actions.append(rebalance_action)
            
            # Handle storage costs based on the frequency
            # For Annual frequency, apply costs at year end
            if storage_frequency == "Annual" and last_year is None:
                last_year = date.year
            elif storage_frequency == "Annual" and date.year != last_year:
                portfolio, storage_cost, fee_date = apply_storage_costs(
                    date, portfolio, invested, storage_settings, buyback_discounts, last_year
                )
                history.append((fee_date, invested, dict(portfolio), "storage_fee"))
                last_year = date.year
                
            # For Quarterly frequency, apply costs at quarter end
            elif storage_frequency == "Quarterly":
                current_quarter = (date.month - 1) // 3
                if last_quarter is None:
                    last_quarter = current_quarter
                elif current_quarter != last_quarter:
                    # Apply quarterly fee (1/4 of annual fee)
                    quarterly_settings = dict(storage_settings)
                    quarterly_settings["storage_fee"] = storage_settings["storage_fee"] / 4
                    portfolio, storage_cost, fee_date = apply_storage_costs(
                        date, portfolio, invested, quarterly_settings, buyback_discounts, date.year
                    )
                    history.append((fee_date, invested, dict(portfolio), "quarterly_fee"))
                    last_quarter = current_quarter
                    
            # For Monthly frequency, apply costs at month end
            elif storage_frequency == "Monthly":
                if last_month is None:
                    last_month = date.month
                elif date.month != last_month:
                    # Apply monthly fee (1/12 of annual fee)
                    monthly_settings = dict(storage_settings)
                    monthly_settings["storage_fee"] = storage_settings["storage_fee"] / 12
                    portfolio, storage_cost, fee_date = apply_storage_costs(
                        date, portfolio, invested, monthly_settings, buyback_discounts, date.year
                    )
                    history.append((fee_date, invested, dict(portfolio), "monthly_fee"))
                    last_month = date.month
            
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
        storage_cost_dates = result[result["Action"].str.contains("fee")].index
        
        for date in storage_cost_dates:
            if storage_frequency == "Annual":
                result.at[date, "Storage Cost"] = result.at[date, "Invested"] * (storage_settings["storage_fee"] / 100) * (1 + storage_settings["vat"] / 100)
            elif storage_frequency == "Quarterly":
                result.at[date, "Storage Cost"] = result.at[date, "Invested"] * (storage_settings["storage_fee"] / 400) * (1 + storage_settings["vat"] / 100)
            elif storage_frequency == "Monthly":
                result.at[date, "Storage Cost"] = result.at[date, "Invested"] * (storage_settings["storage_fee"] / 1200) * (1 + storage_settings["vat"] / 100)
        
        return result
    
    except Exception as e:
        st.error(f"Error in portfolio simulation: {e}")
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=["Date", "Invested", *allocation.keys(), "Portfolio Value", "Action", "Storage Cost"])

# =========================================
# VISUALIZATION FUNCTIONS WITH IMPROVED ANNOTATIONS
# =========================================

def create_portfolio_chart(result_df, language):
    """Create interactive plotly chart for portfolio values with improved annotations"""
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
    
    # Add rebalancing event annotations
    rebalance_events = chart_data[chart_data["Action"].str.contains("rebalance_[12]$", regex=True)]
    
    for date, row in rebalance_events.iterrows():
        fig.add_annotation(
            x=date,
            y=row["Portfolio Value"],
            text="♻️ Rebalance",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#4CAF50",
            bgcolor="#4CAF50",
            bordercolor="#4CAF50",
            font=dict(color="white", size=10),
            borderwidth=1,
            borderpad=2,
            opacity=0.8
        )
    
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
        marker_colors=colors,
        textinfo='percent',
        hoverinfo='label+value+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title=t("current_holdings"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    
    return fig

# New function to create metal performance comparison chart
def create_metal_performance_chart(start_date, end_date, allocation):
    """Create a comparison chart of metal price performance"""
    start_prices = data.loc[start_date]
    
    # Create normalized series for each metal (starting at 100)
    performance_data = []
    for date in data.loc[start_date:end_date].index:
        prices = data.loc[date]
        row = {"date": date}
        
        for metal in ["Gold", "Silver", "Platinum", "Palladium"]:
            normalized_price = (prices[f"{metal}_EUR"] / start_prices[f"{metal}_EUR"]) * 100
            row[metal] = normalized_price
            
        performance_data.append(row)
    
    perf_df = pd.DataFrame(performance_data)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add lines for each metal
    colors = {'Gold': '#D4AF37', 'Silver': '#C0C0C0', 'Platinum': '#E5E4E2', 'Palladium': '#CED0DD'}
    for metal in ["Gold", "Silver", "Platinum", "Palladium"]:
        fig.add_trace(go.Scatter(
            x=perf_df["date"],
            y=perf_df[metal],
            mode='lines',
            name=t(metal.lower()),
            line=dict(color=colors[metal], width=3)
        ))
    
    # Add reference line at 100
    fig.add_shape(
        type="line",
        x0=start_date,
        y0=100,
        x1=end_date,
        y1=100,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    # Update layout
    fig.update_layout(
        title="Metal Price Performance (indexed to 100)",
        xaxis_title="Date",
        yaxis_title="Price Index (Start = 100)",
        height=500,
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
    
    return fig

# =========================================
# MAIN APP LAYOUT WITH IMPROVED UX
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
    # Sekcja presetów
    st.header("🔍 " + t("presets"))
    
    # Tworzymy listę nazw presetów dla aktualnego języka
    preset_options = [(key, PRESETS[key]["name"][language]) for key in PRESETS.keys()]
    
    # Wybór presetu
    selected_preset = st.selectbox(
        t("select_preset"),
        options=[key for key, _ in preset_options],
        format_func=lambda key: next((name for k, name in preset_options if k == key), key),
        index=list(PRESETS.keys()).index("custom")  # Domyślnie "custom"
    )
    
    # Wyświetlanie opisu presetu
    st.info(PRESETS[selected_preset]["description"][language])
    
    # Przycisk do zastosowania presetu
    if selected_preset != "custom":
        if st.button(t("apply_preset")):
            success = apply_preset(selected_preset)
            if success:
                st.success(t("preset_applied"))
                # Wymuszamy odświeżenie strony, aby zastosować nowe ustawienia
                st.rerun()
    
    st.markdown("---")



    
    st.header(t("simulation_settings"))
    
    # Investment amounts and dates
    st.subheader(t("investment_amounts"))
    
    initial_allocation = st.number_input(
        t("initial_allocation"),
        value=100000.0,
        step=100.0,
        help="The initial amount to invest across all metals"
    )
    
    today = datetime.today()
    default_initial_date = today.replace(year=today.year - 20)
    
    initial_date = st.date_input(
        t("first_purchase_date"),
        value=default_initial_date.date(),
        min_value=data.index.min().date(),
        max_value=data.index.max().date(),
        help="When your investment starts"
    )
    
    # Calculate minimum end date (initial_date + 7 years)
    min_end_date = (pd.to_datetime(initial_date) + pd.DateOffset(years=7)).date()
    if min_end_date > data.index.max().date():
        min_end_date = data.index.max().date()
    
    end_purchase_date = st.date_input(
        t("last_purchase_date"),
        value=data.index.max().date(),
        min_value=min_end_date,
        max_value=data.index.max().date(),
        help="When your investment period ends"
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
    if st.button(t("reset_allocation"), help="Reset to default 40/20/20/20 allocation"):
        st.session_state["alloc_Gold"] = 40
        st.session_state["alloc_Silver"] = 20
        st.session_state["alloc_Platinum"] = 20
        st.session_state["alloc_Palladium"] = 20
        st.rerun()
    
    # Allocation sliders
    allocation_gold = st.slider(
        t("gold"), 0, 100, key="alloc_Gold", 
        help="Percentage allocated to Gold"
    )
    allocation_silver = st.slider(
        t("silver"), 0, 100, key="alloc_Silver", 
        help="Percentage allocated to Silver"
    )
    allocation_platinum = st.slider(
        t("platinum"), 0, 100, key="alloc_Platinum", 
        help="Percentage allocated to Platinum"
    )
    allocation_palladium = st.slider(
        t("palladium"), 0, 100, key="alloc_Palladium", 
        help="Percentage allocated to Palladium"
    )
    
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
        index=1,
        help="How often to make additional purchases"
    )
    
    if purchase_freq == t("week"):
        days_of_week = [t("monday"), t("tuesday"), t("wednesday"), t("thursday"), t("friday")]
        selected_day = st.selectbox(
            t("purchase_day_of_week"), 
            days_of_week, 
            index=0,
            help="Which day of the week to make purchases"
        )
        purchase_day = selected_day
        default_purchase_amount = 250.0
    elif purchase_freq == t("month"):
        purchase_day = st.number_input(
            t("purchase_day_of_month"), 
            min_value=1, 
            max_value=28, 
            value=1,
            help="Which day of the month to make purchases (1-28)"
        )
        default_purchase_amount = 1000.0
    elif purchase_freq == t("quarter"):
        purchase_day = st.number_input(
            t("purchase_day_of_quarter"), 
            min_value=1, 
            max_value=28, 
            value=1,
            help="Which day of the first month of each quarter to make purchases (1-28)"
        )
        default_purchase_amount = 3250.0
    else:
        purchase_day = None
        default_purchase_amount = 0.0
    
    purchase_amount = st.number_input(
        t("purchase_amount"), 
        value=default_purchase_amount, 
        step=50.0,
        help="Amount to invest in each recurring purchase"
    )

    # Storage costs
    st.subheader(t("storage_costs"))
    
    with st.expander(t("storage_costs"), expanded=False):
        storage_fee = st.number_input(
            t("annual_storage_fee"), 
            value=1.5,
            help="Annual percentage fee for storing metals"
        )
        
        storage_frequency = st.selectbox(
            "Storage Fee Frequency",
            ["Annual", "Quarterly", "Monthly"],
            index=0,
            help="How often storage fees are charged"
        )
        
        vat = st.number_input(
            t("vat"), 
            value=19.0,
            help="VAT percentage charged on storage fees"
        )
        
        storage_metal = st.selectbox(
            t("storage_metal"),
            ["Gold", "Silver", "Platinum", "Palladium", t("best_of_year"), "ALL"],
            help="Which metal(s) to sell to cover storage costs"
        )
    
    # Storage settings dictionary
    storage_settings = {
        "storage_fee": storage_fee,
        "vat": vat,
        "storage_metal": storage_metal
    }

    
    
    # Rebalancing settings
    st.subheader(t("rebalancing"))
    
    with st.expander(t("rebalancing"), expanded=False):
        # Default rebalancing dates based on initial date
        rebalance_base_year = initial_date.year + 1
        rebalance_1_default = datetime(rebalance_base_year, 4, 1)
        rebalance_2_default = datetime(rebalance_base_year, 10, 1)
        
        # First rebalancing
        rebalance_1 = st.checkbox(
            t("rebalance_1"), 
            value=False,
            help="Enable the first annual rebalancing event"
        )
        rebalance_1_condition = st.checkbox(
            t("deviation_condition") + " 1", 
            value=False,
            help="Only rebalance if allocation deviates beyond threshold"
        )
        rebalance_1_threshold = st.number_input(
            t("deviation_threshold") + " 1",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=0.5,
            help="Percentage deviation that triggers rebalancing"
        )
        
        rebalance_1_start = st.date_input(
            t("start_rebalance") + " 1",
            value=rebalance_1_default.date(),
            min_value=data.index.min().date(),
            max_value=data.index.max().date(),
            help="Date when first rebalancing starts (recurs annually)"
        )
        
        # Second rebalancing
        rebalance_2 = st.checkbox(
            t("rebalance_2"), 
            value=False,
            help="Enable a second annual rebalancing event"
        )
        rebalance_2_condition = st.checkbox(
            t("deviation_condition") + " 2", 
            value=False,
            help="Only rebalance if allocation deviates beyond threshold"
        )
        rebalance_2_threshold = st.number_input(
            t("deviation_threshold") + " 2",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=0.5,
            help="Percentage deviation that triggers rebalancing"
        )
        
        rebalance_2_start = st.date_input(
            t("start_rebalance") + " 2",
            value=rebalance_2_default.date(),
            min_value=data.index.min().date(),
            max_value=data.index.max().date(),
            help="Date when second rebalancing starts (recurs annually)"
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
    

    
    # Margins and fees
    st.subheader(t("margins_fees"))
    
    with st.expander(t("margins_fees"), expanded=False):
    # Inicjalizacja session_state dla marż
        if "margin_gold" not in st.session_state:
            st.session_state["margin_gold"] = 15.6
        if "margin_silver" not in st.session_state:
            st.session_state["margin_silver"] = 18.36
        if "margin_platinum" not in st.session_state:
            st.session_state["margin_platinum"] = 24.24
        if "margin_palladium" not in st.session_state:
            st.session_state["margin_palladium"] = 22.49
        
        margins = {
            "Gold": st.number_input(t("gold_margin"), value=st.session_state["margin_gold"], key="margin_gold"),
            "Silver": st.number_input(t("silver_margin"), value=st.session_state["margin_silver"], key="margin_silver"),
            "Platinum": st.number_input(t("platinum_margin"), value=st.session_state["margin_platinum"], key="margin_platinum"),
            "Palladium": st.number_input(t("palladium_margin"), value=st.session_state["margin_palladium"], key="margin_palladium")
        }
    
    # Buyback prices
    st.subheader(t("buyback_prices"))
    
    with st.expander(t("buyback_prices"), expanded=False):
        buyback_discounts = {
            "Gold": st.number_input(
                t("gold_buyback"), 
                value=-1.5, 
                step=0.1,
                help="Percentage difference from spot price when selling gold"
            ),
            "Silver": st.number_input(
                t("silver_buyback"), 
                value=-3.0, 
                step=0.1,
                help="Percentage difference from spot price when selling silver"
            ),
            "Platinum": st.number_input(
                t("platinum_buyback"), 
                value=-3.0, 
                step=0.1,
                help="Percentage difference from spot price when selling platinum"
            ),
            "Palladium": st.number_input(
                t("palladium_buyback"), 
                value=-3.0, 
                step=0.1,
                help="Percentage difference from spot price when selling palladium"
            )
        }
    
    # Rebalancing prices
    st.subheader(t("rebalance_prices"))
    
    with st.expander(t("rebalance_prices"), expanded=False):
        rebalance_markup = {
            "Gold": st.number_input(
                t("gold_rebalance"), 
                value=6.5, 
                step=0.1,
                help="Percentage markup when buying gold during rebalancing"
            ),
            "Silver": st.number_input(
                t("silver_rebalance"), 
                value=6.5, 
                step=0.1,
                help="Percentage markup when buying silver during rebalancing"
            ),
            "Platinum": st.number_input(
                t("platinum_rebalance"), 
                value=6.5, 
                step=0.1,
                help="Percentage markup when buying platinum during rebalancing"
            ),
            "Palladium": st.number_input(
                t("palladium_rebalance"), 
                value=6.5, 
                step=0.1,
                help="Percentage markup when buying palladium during rebalancing"
            )
        }
    
    # Advanced options
    st.subheader(t("advanced_options"))
    
    with st.expander(t("inflation_settings"), expanded=False):
        use_custom_inflation = st.checkbox(
            "Use custom inflation", 
            value=False,
            help="Override the default inflation data"
        )
        if use_custom_inflation:
            custom_inflation = {}
            with st.container():
                for year in range(initial_date.year, end_purchase_date.year + 1):
                    custom_inflation[year] = st.number_input(
                        f"{t('year')} {year} - {t('custom_inflation')}",
                        value=2.5,
                        step=0.1,
                        help=f"Custom inflation rate for {year}"
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
            "storage_frequency": storage_frequency,
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
        disabled=not dates_valid or total_allocation != 100,
        help="Run the simulation with current settings"
    )

# Main content area
if total_allocation != 100:
    st.error(t("allocation_error").format(total_allocation))
    st.stop()

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Overview", "Detailed Analysis", "Metal Performance", "Data Export"])

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
            inflation_data=inflation_real,
            storage_frequency=storage_frequency
        )
        st.session_state.result = result
        
        # Calculate additional metrics
        if not result.empty:
            start_date = result.index.min()
            end_date = result.index.max()
            years = (end_date - start_date).days / 365.25
            
            capital_invested = result["Invested"].max()
            portfolio_value = result["Portfolio Value"].iloc[-1]
            
            # Calculate annual return
            if capital_invested > 0 and years > 0:
                annual_return = (portfolio_value / capital_invested) ** (1 / years) - 1
            else:
                annual_return = 0.0
            
            st.session_state.metrics = {
                "years": years,
                "capital_invested": capital_invested,
                "portfolio_value": portfolio_value,
                "annual_return": annual_return,
                "start_date": start_date,
                "end_date": end_date
            }

# Use cached result if available
result = st.session_state.result
metrics = st.session_state.get("metrics", {})

# Tab 1: Portfolio Overview
with tab1:
    # Portfolio value chart
    fig = create_portfolio_chart(result, language)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Metrics
    st.subheader("Key Metrics")
    
    # Two-column layout for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Investment",
            format_currency(metrics.get("capital_invested", 0)),
            help="Total amount invested over time"
        )
        
        # Add total return metric
        total_return = ((metrics.get("portfolio_value", 0) / metrics.get("capital_invested", 1)) - 1) * 100
        st.metric(
            "Total Return",
            f"{total_return:.2f}%",
            delta=f"{total_return:.1f}%",
            help="Total percentage return on investment"
        )
    
    with col2:
        st.metric(
            "Current Portfolio Value",
            format_currency(metrics.get("portfolio_value", 0)),
            help="Current value of your precious metals portfolio"
        )
        
        # Add investment period
        st.metric(
            "Investment Period",
            f"{metrics.get('years', 0):.1f} years",
            help="Total investment period in years"
        )
        
    with col3:
        # Add annualized return
        annual_return = metrics.get("annual_return", 0) * 100
        st.metric(
            "Annualized Return",
            f"{annual_return:.2f}%",
            delta=f"{annual_return:.1f}%",
            help="Average yearly return on investment"
        )
        
        # Add storage costs summary
        total_storage = result["Storage Cost"].sum()
        st.metric(
            "Total Storage Costs",
            format_currency(total_storage),
            help="Total storage costs over investment period"
        )
    
    # Current holdings section
    st.subheader(t("current_holdings"))
    
    # Get final holdings
    if not result.empty:
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
            if not result.empty:
                end_date = result.index.max()
                pie_fig = create_allocation_pie_chart(final_holdings, data.loc[end_date], language)
                st.plotly_chart(pie_fig, use_container_width=True)

# Tab 2: Detailed Analysis
with tab2:
    st.subheader("Detailed Portfolio Analysis")
    
    # Calculate metal price growth
    if not result.empty:
        start_date = result.index.min()
        end_date = result.index.max()
        
        start_prices = data.loc[start_date]
        end_prices = data.loc[end_date]
        
        growth_metrics = {}
        for metal in ["Gold", "Silver", "Platinum", "Palladium"]:
            start_price = start_prices[metal + "_EUR"]
            end_price = end_prices[metal + "_EUR"]
            growth_pct = (end_price / start_price - 1) * 100
            growth_metrics[metal] = growth_pct
        
        # Metal price growth section
        st.subheader(t("price_growth"))
        
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
    
    # Calculate value comparison
    if not result.empty:
        # Get final holdings
        final_holdings = {
            "Gold": result.iloc[-1]["Gold"],
            "Silver": result.iloc[-1]["Silver"],
            "Platinum": result.iloc[-1]["Platinum"],
            "Palladium": result.iloc[-1]["Palladium"]
        }
        
        end_date = result.index.max()
        end_prices = data.loc[end_date]
        
        capital_invested = result["Invested"].max()
        portfolio_value = result["Portfolio Value"].iloc[-1]
        
        # Calculate replacement value
        purchase_replacement_value = 0.0
        for metal in ["Gold", "Silver", "Platinum", "Palladium"]:
            grams = final_holdings[metal]
            spot_price = end_prices[metal + "_EUR"]
            margin_percent = margins[metal] / 100
            buy_price = spot_price * (1 + margin_percent)
            purchase_replacement_value += grams * buy_price
        
        # Calculate weighted average growth
        weighted_start_price = sum(
            allocation[metal] * data.loc[start_date][metal + "_EUR"]
            for metal in ["Gold", "Silver", "Platinum", "Palladium"]
        )
        
        weighted_end_price = sum(
            allocation[metal] * data.loc[end_date][metal + "_EUR"]
            for metal in ["Gold", "Silver", "Platinum", "Palladium"]
        )
        
        years = (end_date - start_date).days / 365.25
        
        if weighted_start_price > 0 and years > 0:
            weighted_avg_annual_growth = (weighted_end_price / weighted_start_price) ** (1 / years) - 1
        else:
            weighted_avg_annual_growth = 0.0
        
        # Display value comparison
        st.subheader("Value Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Investment vs Portfolio Value
            st.markdown(f"""
            <div style="background-color: #E6F4EA; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <h4 style="color: green;">🛒 {t('purchase_value')}</h4>
                <h2 style="color: green;">{format_currency(portfolio_value)}</h2>
                <p>vs. Investment: {format_currency(capital_invested)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Replacement Value
            st.markdown(f"""
            <div style="background-color: #FDECEA; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <h4 style="color: red;">💎 Replacement Value</h4>
                <h2 style="color: red;">{format_currency(purchase_replacement_value)}</h2>
                <p>Cost to rebuy current holdings at retail prices</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Average annual growth metric
        st.metric(
            t("annual_growth_weighted"),
            f"{weighted_avg_annual_growth * 100:.2f}%",
            delta=f"{weighted_avg_annual_growth * 100:.1f}%"
        )
    
    # Show yearly summary table with improved formatting
    st.subheader(t("yearly_view"))
    
    if not result.empty:
        # Group by year and take first day
        result_yearly = result.groupby(result.index.year).first()
        
        # Create a simple table with selected columns
        yearly_table = pd.DataFrame({
            t("year"): result_yearly.index,
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
        
        # Display table with improved styling
        st.dataframe(
            yearly_table,
            use_container_width=True,
            column_config={
                t("year"): st.column_config.NumberColumn(format="%d"),
                t("invested_amount"): st.column_config.TextColumn("Invested (EUR)"),
                t("portfolio_value_short"): st.column_config.TextColumn("Portfolio Value (EUR)"),
                t("gold"): st.column_config.NumberColumn("Gold (g)", format="%.2f g"),
                t("silver"): st.column_config.NumberColumn("Silver (g)", format="%.2f g"),
                t("platinum"): st.column_config.NumberColumn("Platinum (g)", format="%.2f g"),
                t("palladium"): st.column_config.NumberColumn("Palladium (g)", format="%.2f g"),
                t("action"): st.column_config.TextColumn("Actions")
            },
            hide_index=True
        )
    
    # Storage costs summary with improved visualization
    st.subheader(t("storage_summary"))
    
    if not result.empty:
        # Filter for storage fee entries
        storage_costs = result[result["Action"].str.contains("fee")]
        
        # Calculate total storage cost
        total_storage_cost = result["Storage Cost"].sum()
        
        # Calculate average annual storage cost
        years = (result.index.max() - result.index.min()).days / 365.25
        
        if years > 0:
            avg_annual_storage_cost = total_storage_cost / years
        else:
            avg_annual_storage_cost = 0.0
        
        # Calculate percentage of current portfolio value
        portfolio_value = result["Portfolio Value"].iloc[-1]
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
        
        # Add storage cost visualization if there are storage costs
        if total_storage_cost > 0:
            # Create a bar chart of storage costs by year
            storage_by_year = storage_costs.groupby(storage_costs.index.year)["Storage Cost"].sum()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=storage_by_year.index,
                    y=storage_by_year.values,
                    marker_color='indianred'
                )
            ])
            
            fig.update_layout(
                title="Storage Costs by Year",
                xaxis_title="Year",
                yaxis_title="Storage Cost (EUR)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Metal Performance Comparison
with tab3:
    st.subheader("Metal Performance Comparison")
    
    if not result.empty:
        start_date = result.index.min()
        end_date = result.index.max()
        
        # Create and display the performance chart
        perf_fig = create_metal_performance_chart(start_date, end_date, allocation)
        st.plotly_chart(perf_fig, use_container_width=True)
        
        # Add correlation matrix
        st.subheader("Metal Price Correlation Matrix")
        
        # Get price data for the period
        price_data = data.loc[start_date:end_date, ["Gold_EUR", "Silver_EUR", "Platinum_EUR", "Palladium_EUR"]]
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=["Gold", "Silver", "Platinum", "Palladium"],
            y=["Gold", "Silver", "Platinum", "Palladium"],
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size":14}
        ))
        
        fig.update_layout(
            title="Correlation of Metal Price Returns",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical perspective
        st.subheader("Historical Price Ranges")
        
        # Calculate historical stats
        historic_stats = {}
        for metal in ["Gold", "Silver", "Platinum", "Palladium"]:
            prices = data[f"{metal}_EUR"]
            historic_stats[metal] = {
                "min": prices.min(),
                "max": prices.max(),
                "mean": prices.mean(),
                "start": prices.loc[start_date],
                "end": prices.loc[end_date],
                "volatility": prices.pct_change().std() * (252 ** 0.5) * 100  # Annualized volatility
            }
        
        # Display historical stats
        stats_df = pd.DataFrame(index=["Gold", "Silver", "Platinum", "Palladium"])
        stats_df["Min Price (EUR)"] = [historic_stats[m]["min"] for m in stats_df.index]
        stats_df["Max Price (EUR)"] = [historic_stats[m]["max"] for m in stats_df.index]
        stats_df["Mean Price (EUR)"] = [historic_stats[m]["mean"] for m in stats_df.index]
        stats_df["Start Price (EUR)"] = [historic_stats[m]["start"] for m in stats_df.index]
        stats_df["End Price (EUR)"] = [historic_stats[m]["end"] for m in stats_df.index]
        stats_df["Volatility (%)"] = [historic_stats[m]["volatility"] for m in stats_df.index]
        
        st.dataframe(
            stats_df.round(2),
            use_container_width=True,
            column_config={
                "Min Price (EUR)": st.column_config.NumberColumn(format="%.2f €"),
                "Max Price (EUR)": st.column_config.NumberColumn(format="%.2f €"),
                "Mean Price (EUR)": st.column_config.NumberColumn(format="%.2f €"),
                "Start Price (EUR)": st.column_config.NumberColumn(format="%.2f €"),
                "End Price (EUR)": st.column_config.NumberColumn(format="%.2f €"),
                "Volatility (%)": st.column_config.NumberColumn(format="%.2f%%")
            }
        )

# Tab 4: Data Export
with tab4:
    st.subheader(t("export_data"))
    
    if not result.empty:
        # Create tabs for different export formats
        export_tab1, export_tab2, export_tab3 = st.tabs(["CSV Export", "PDF Report", "Raw Data"])
        
        with export_tab1:
            # Create download button for CSV
            csv = result.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="precious_metals_simulation.csv",
                mime="text/csv",
                help="Download the full simulation data as CSV"
            )
            
            # Add option to export yearly summary
            yearly_csv = result.groupby(result.index.year).first().to_csv()
            st.download_button(
                label="Download Yearly Summary CSV",
                data=yearly_csv,
                file_name="precious_metals_yearly_summary.csv",
                mime="text/csv",
                help="Download yearly summary data as CSV"
            )
        
        with export_tab2:
            # Create PDF report
            if not result.empty:
                start_date = result.index.min()
                end_date = result.index.max()
                
                # Get portfolio value chart for PDF
                pdf_fig = create_portfolio_chart(result, language)
                
                # Create portfolio summary data
                portfolio_summary = {
                    "Investment Period": f"{(end_date - start_date).days / 365.25:.2f} years",
                    "Total Invested": format_currency(result["Invested"].max()),
                    "Final Portfolio Value": format_currency(result["Portfolio Value"].iloc[-1]),
                    "Total Return": f"{((result['Portfolio Value'].iloc[-1] / result['Invested'].max()) - 1) * 100:.2f}%",
                    "Gold Holdings": f"{result.iloc[-1]['Gold']:.2f} g",
                    "Silver Holdings": f"{result.iloc[-1]['Silver']:.2f} g",
                    "Platinum Holdings": f"{result.iloc[-1]['Platinum']:.2f} g",
                    "Palladium Holdings": f"{result.iloc[-1]['Palladium']:.2f} g"
                }
                
                # Generate PDF download link
                pdf_html = create_pdf_download_link(pdf_fig, portfolio_summary)
                st.markdown(pdf_html, unsafe_allow_html=True)
        
        with export_tab3:
            # Display raw data table with pagination
            st.subheader("Complete Simulation Data")
            
            # Add filtering options
            action_filter = st.multiselect(
                "Filter by Action",
                options=result["Action"].unique().tolist(),
                default=[],
                help="Select specific actions to filter the data"
            )
            
            if action_filter:
                filtered_result = result[result["Action"].isin(action_filter)]
            else:
                filtered_result = result
            
            # Display filtered data
            st.dataframe(
                filtered_result,
                use_container_width=True,
                height=500
            )

# Footer with disclaimer and additional help
st.markdown("---")
with st.expander("Help & Information"):
    st.markdown("""
    ## About This Simulator
    
    This Precious Metals Portfolio Simulator allows you to model investment strategies for gold, silver, platinum, and palladium over time. Key features include:
    
    - **Initial and recurring investments** with customizable frequencies
    - **Portfolio rebalancing** to maintain your target allocation
    - **Storage cost simulation** with different fee schedules
    - **Real price data** for accurate historical modeling
    - **Inflation adjustment** to see your real returns
    
    ## How to Use
    
    1. Set your investment parameters in the sidebar
    2. Ensure your metal allocation adds up to 100%
    3. Click "Run Simulation" to see the results
    4. Explore different tabs for various analyses
    5. Export your results for further analysis
    
    ## Data Sources
    
    The simulator uses LBMA (London Bullion Market Association) price data for the precious metals and country-specific inflation data.
    
    ## Disclaimer
    
    This simulation is for educational purposes only. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.
    """)

st.caption("Disclaimer: This simulation is for educational purposes only. Past performance does not guarantee future results.")
st.caption(f"App Version: {APP_CONFIG['version']} | Last Updated: May 2025")
if (rebalance_settings["rebalance_1"] and 
        date >= pd.to_datetime(rebalance_settings["rebalance_1_start"]) and 
        date.month == rebalance_settings["rebalance_1_start"].month and 
        date.day == rebalance_settings["rebalance_1_start"].day):
                
        portfolio, rebalance_action = apply_rebalance(
            date, 
            portfolio,
            allocation,
            data.loc[date],
            "rebalance_1", 
            rebalance_settings["rebalance_1_condition"], 
            rebalance_settings["rebalance_1_threshold"],
            buyback_discounts,
            rebalance_markup,
            last_rebalance_dates
        )
                
        actions.append(rebalance_action)
            
        # Check for
