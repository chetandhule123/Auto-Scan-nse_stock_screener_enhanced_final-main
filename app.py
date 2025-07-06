import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import queue
import os
import pytz
import time
import requests
import streamlit.components.v1 as components

# Import custom modules
from scanners.macd_scanner import MACDScanner
from scanners.macd_scanner_original import MACDScannerOriginal
from scanners.range_breakout_scanner import RangeBreakoutScanner
from scanners.resistance_breakout_scanner import ResistanceBreakoutScanner
from scanners.support_level_scanner import SupportLevelScanner
from utils.market_indices import MarketIndices
from utils.data_fetcher import DataFetcher
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="🚀 NSE Stock Screener Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}
if 'auto_scan_enabled' not in st.session_state:
    st.session_state.auto_scan_enabled = True
if 'scan_interval' not in st.session_state:
    st.session_state.scan_interval = 15  # minutes - FIXED: Default to 15 minutes
if 'active_scanners' not in st.session_state:
    st.session_state.active_scanners = {
        "MACD 15min": True,
        "MACD 4h": True, 
        "MACD 1d": True,
        "Range Breakout 4h": True,
        "Resistance Breakout 4h": True,
        "Support Level 4h": True
    }

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(IST)

def check_market_hours_ist():
    """Check if NSE market is open in IST"""
    now = get_ist_time()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
    return is_weekday and market_open <= now <= market_close

def show_auto_refresh_timer():
    components.html("""
        <div style="text-align: center; padding: 10px;">
            <h4 style="color: #2a5298;">🔄 Auto Refresh In: <span id="timer">15:00</span></h4>
        </div>
        <script>
        let totalSeconds = 900;
        function updateTimer() {
            let minutes = Math.floor(totalSeconds / 60);
            let seconds = totalSeconds % 60;
            document.getElementById('timer').innerHTML = 
                (minutes < 10 ? "0" + minutes : minutes) + ":" + 
                (seconds < 10 ? "0" + seconds : seconds);
            if (totalSeconds > 0) {
                totalSeconds--;
                setTimeout(updateTimer, 1000);
            }
        }
        setTimeout(function() {
            window.location.reload();
        }, 900000);  // Reload after 15 minutes
        updateTimer();  // Start countdown
        </script>
    """, height=80)




def main():
    # Fresh modern UI header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
            🚀 NSE Stock Screener Pro
        </h1>
        <p style="color: #E8F4FD; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Advanced Technical Analysis & Real-time Market Intelligence
        </p>
        <p style="color: #B8D4EA; text-align: center; margin: 0.5rem 0 0 0;">
            IST: {current_time} | Market Status: {market_status}
        </p>
    </div>
    """.format(
        current_time=get_ist_time().strftime('%Y-%m-%d %H:%M:%S'),
        market_status="🟢 OPEN" if check_market_hours_ist() else "🔴 CLOSED"
    ), unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Scanner Configuration")
        
        # Auto-scan settings
        st.markdown("#### 🔄 Auto-Scan Settings")
        
        auto_scan = st.checkbox("Enable Auto-Scan (15min intervals)", value=st.session_state.auto_scan_enabled, key="auto_scan_checkbox")

        
        
        # FIXED: Force scan interval to 15 minutes as per requirements
        scan_interval = st.selectbox(
            "Scan Interval (minutes)",
            [15, 30, 60],  # Removed 5 and 10 minute options
            index=0,  # Default to 15 minutes
            key="scan_interval_select"
        )
        
        if auto_scan != st.session_state.auto_scan_enabled:
            st.session_state.auto_scan_enabled = auto_scan
            st.session_state.scan_interval = scan_interval
        
        # Manual scan button
        if st.button("🔍 Run Manual Scan", type="primary", use_container_width=True):
            run_all_scanners()
            st.rerun()
        
        
        st.session_state.notification_enabled = st.checkbox("Enable Telegram Notifications", value=st.session_state.get('notification_enabled', True),key="telegram_notifications")
        
        # Scanner selection - PRESERVE EXISTING MACD LOGIC
        st.markdown("#### 📊 Active Scanners")
        
        # MACD Scanners (existing logic preserved)
        st.markdown("**MACD Scanners:**")
        st.session_state.active_scanners["MACD 15min"] = st.checkbox("MACD 15-minute", value=st.session_state.active_scanners["MACD 15min"])
        st.session_state.active_scanners["MACD 4h"] = st.checkbox("MACD 4-hour", value=st.session_state.active_scanners["MACD 4h"])
        st.session_state.active_scanners["MACD 1d"] = st.checkbox("MACD 1-day", value=st.session_state.active_scanners["MACD 1d"])
        
        # New scanners
        st.markdown("**New Advanced Scanners:**")
        st.session_state.active_scanners["Range Breakout 4h"] = st.checkbox("Range Breakout (4h)", value=st.session_state.active_scanners["Range Breakout 4h"])
        st.session_state.active_scanners["Resistance Breakout 4h"] = st.checkbox("Resistance Breakout (4h)", value=st.session_state.active_scanners["Resistance Breakout 4h"])
        st.session_state.active_scanners["Support Level 4h"] = st.checkbox("Support Level (4h)", value=st.session_state.active_scanners["Support Level 4h"])
        
        # Export options
        st.markdown("#### 📊 Export Options")
        if st.button("📥 Export Results", use_container_width=True):
            export_results()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Market indices display
        display_market_indices()
        
        # Scanner results tabs
        display_scanner_results()
    
    with col2:
        # Status and info panel - get the containers that need updating
        time_since_container, countdown_container = display_status_panel()
    
    # Auto-scan logic
    if st.session_state.auto_scan_enabled:
        handle_auto_scan()
        show_auto_refresh_timer()

    
    # Update the counters in real-time
    if time_since_container or countdown_container:
        update_counters(time_since_container, countdown_container)

    # 🔁 Force full page reload every 15 minutes even if browser is inactive
    st.markdown("""
        <meta http-equiv="refresh" content="900">
    """, unsafe_allow_html=True)

    # 🕒 Optional: Show visible countdown
    show_auto_refresh_timer()
    










def display_market_indices():
    """Display real-time market indices with fresh UI"""
    st.markdown("### 📊 Live Market Indices")
    
    try:
        market_indices = MarketIndices()
        indices_data = market_indices.get_live_indices()
        
        if not indices_data.empty:
            # Create responsive columns
            cols = st.columns(min(len(indices_data), 4))
            
            for i, (index, row) in enumerate(indices_data.iterrows()):
                with cols[i % 4]:
                    # Fix the LSP error by converting to string
                    name = str(row['Name'])
                    price = float(row['Price'])
                    change = float(row['Change'])
                    change_pct = float(row['Change%'])
                    
                    st.metric(
                        label=name,
                        value=f"₹{price:,.2f}",
                        delta=f"{change:+.2f} ({change_pct:+.2f}%)"
                    )
        else:
            st.warning("📊 Market indices data temporarily unavailable")
    except Exception as e:
        st.error(f"⚠️ Error fetching market indices: {str(e)}")

def display_scanner_results():
    """Display results from all active scanners with fresh UI"""
    st.markdown("### 🎯 Technical Scanner Results")
    
    # Get active scanners from session state
    active_scanners = [name for name, active in st.session_state.active_scanners.items() if active]
    
    if active_scanners:
        tabs = st.tabs(active_scanners)
        
        for i, scanner_name in enumerate(active_scanners):
            with tabs[i]:
                display_individual_scanner_results(scanner_name)
    else:
        st.info("💡 No scanners selected. Please enable scanners from the sidebar.")

def display_individual_scanner_results(scanner_name):
    """Display results for a specific scanner"""
    if scanner_name in st.session_state.scan_results:
        results = st.session_state.scan_results[scanner_name]
        
        if isinstance(results, pd.DataFrame) and not results.empty:
            # Add sorting and filtering options
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    options=results.columns.tolist(),
                    key=f"sort_{scanner_name}"
                )
            
            with col2:
                sort_order = st.selectbox(
                    "Order",
                    ["Descending", "Ascending"],
                    key=f"order_{scanner_name}"
                )
            
            with col3:
                max_results = st.number_input(
                    "Max Results",
                    min_value=10,
                    max_value=100,
                    value=50,
                    key=f"max_{scanner_name}"
                )
            
            # Sort and limit results
            ascending = sort_order == "Ascending"
            sorted_results = results.sort_values(by=sort_by, ascending=ascending).head(max_results)
            
            # Display results table
            st.dataframe(
                sorted_results,
                use_container_width=True,
                hide_index=True
            )
            
            # Display summary stats
            st.write(f"**Total signals found:** {len(results)}")
            st.write(f"**Showing top:** {len(sorted_results)} results")
            
        else:
            st.info(f"No signals found for {scanner_name}")
    else:
        st.info(f"No data available for {scanner_name}. Run a scan to see results.")


 


def display_status_panel():
    """Display status and information panel with IST times"""
    st.markdown("### 📋 Control Panel")
    
    current_time = get_ist_time()
    
    # Last scan information
    st.markdown("#### ⏱️ Scan Status")
    if st.session_state.last_scan_time:
        time_since = current_time - st.session_state.last_scan_time
        minutes_ago = int(time_since.total_seconds() / 60)
        seconds_ago = int(time_since.total_seconds() % 60)
        
        # Create a container for the time since counter
        time_since_container = st.empty()
        time_since_container.write(f"**Last Scan:** {st.session_state.last_scan_time.strftime('%H:%M:%S IST')}")
        time_since_container.write(f"**Time Since:** {minutes_ago}m {seconds_ago}s")
    else:
        st.write("**Last Scan:** Never")
    
    # Auto-scan status
    if st.session_state.auto_scan_enabled:
        st.success("✅ Auto-scan ENABLED")
        st.write(f"**Interval:** {st.session_state.scan_interval} minutes")
        
        # Next scan countdown
        if st.session_state.last_scan_time:
            # FIXED: Use minimum 15-minute intervals
            actual_interval = max(st.session_state.scan_interval, 15)
            next_scan = st.session_state.last_scan_time + timedelta(minutes=actual_interval)
            time_to_next = next_scan - current_time
            
            # Create a container for the countdown
            countdown_container = st.empty()
            
            if time_to_next.total_seconds() > 0:
                minutes_left = int(time_to_next.total_seconds() / 60)
                seconds_left = int(time_to_next.total_seconds() % 60)
                countdown_container.write(f"**Next Scan:** {minutes_left}m {seconds_left}s")
            else:
                countdown_container.write("**Next Scan:** ⏰ Due now")
    else:
        st.info("⏸️ Auto-scan DISABLED")
    
    # Rest of the function remains the same...
    # Market status with IST
    st.markdown("#### 🏛️ NSE Market Status")
    if check_market_hours_ist():
        st.success("🟢 MARKET OPEN")
        st.write(f"**Current Time:** {current_time.strftime('%H:%M:%S IST')}")
    else:
        st.error("🔴 MARKET CLOSED")
        st.write(f"**Current Time:** {current_time.strftime('%H:%M:%S IST')}")
        st.write("**Market Hours:** 09:15 - 15:30 IST")
    
    # Scanner statistics
    st.markdown("#### 📈 Live Statistics")
    total_signals = sum(len(results) if isinstance(results, pd.DataFrame) else 0 
                       for results in st.session_state.scan_results.values())
    st.metric("🎯 Total Active Signals", total_signals)
    
    # Active scanners count
    active_count = sum(1 for active in st.session_state.active_scanners.values() if active)
    st.metric("🔧 Active Scanners", f"{active_count}/6")
    
    # Return the containers that need to be updated
    if st.session_state.auto_scan_enabled and st.session_state.last_scan_time:
        return time_since_container, countdown_container
    elif st.session_state.last_scan_time:
        return time_since_container, None
    else:
        return None, None



def update_counters(time_since_container, countdown_container):
    """Continuously update the time counters"""
    current_time = get_ist_time()
    
    if time_since_container and st.session_state.last_scan_time:
        time_since = current_time - st.session_state.last_scan_time
        minutes_ago = int(time_since.total_seconds() / 60)
        seconds_ago = int(time_since.total_seconds() % 60)
        time_since_container.write(f"**Time Since:** {minutes_ago}m {seconds_ago}s")
    
    if countdown_container and st.session_state.last_scan_time and st.session_state.auto_scan_enabled:
        actual_interval = max(st.session_state.scan_interval, 15)
        next_scan = st.session_state.last_scan_time + timedelta(minutes=actual_interval)
        time_to_next = next_scan - current_time
        
        if time_to_next.total_seconds() > 0:
            minutes_left = int(time_to_next.total_seconds() / 60)
            seconds_left = int(time_to_next.total_seconds() % 60)
            countdown_container.write(f"**Next Scan:** {minutes_left}m {seconds_left}s")
        else:
            countdown_container.write("**Next Scan:** ⏰ Due now")
    
    # Schedule the next update in 1 second
    time.sleep(1)
    st.rerun()







def run_all_scanners():
    """Run all enabled scanners and send Telegram notifications if enabled"""
    with st.spinner("🔄 Running active scanners..."):
        try:
            # Initialize scanners
            macd_scanner_original = MACDScannerOriginal()
            range_scanner = RangeBreakoutScanner()
            resistance_scanner = ResistanceBreakoutScanner()
            support_scanner = SupportLevelScanner()
            
            # Run scanners based on active selections
            scan_results = {}
            
            if st.session_state.active_scanners["MACD 15min"]:
                scan_results["MACD 15min"] = macd_scanner_original.scan(timeframe="15m")
            
            if st.session_state.active_scanners["MACD 4h"]:
                scan_results["MACD 4h"] = macd_scanner_original.scan(timeframe="4h")
            
            if st.session_state.active_scanners["MACD 1d"]:
                scan_results["MACD 1d"] = macd_scanner_original.scan(timeframe="1d")
            
            if st.session_state.active_scanners["Range Breakout 4h"]:
                scan_results["Range Breakout 4h"] = range_scanner.scan(timeframe="4h")
            
            if st.session_state.active_scanners["Resistance Breakout 4h"]:
                scan_results["Resistance Breakout 4h"] = resistance_scanner.scan(timeframe="4h")
            
            if st.session_state.active_scanners["Support Level 4h"]:
                scan_results["Support Level 4h"] = support_scanner.scan(timeframe="4h")
            
            # Update session state with results
            st.session_state.scan_results = scan_results
            #for name, df in scan_results.items():
                #if isinstance(df, pd.DataFrame):
                    #st.write(f"🔍 Scanner: {name}")
                    #st.dataframe(df.head())

            st.session_state.last_scan_time = get_ist_time()
            
            # Send Telegram notification if enabled and there are results
            if (st.session_state.notification_enabled and 
                any(isinstance(df, pd.DataFrame) and not df.empty 
                for df in scan_results.values())):
                
                send_telegram_notification(scan_results)
            
            st.success("✅ All active scanners completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error running scanners: {str(e)}")


def send_telegram_notification(scan_results):
    """Send formatted scan results to Telegram"""
    try:
        BOT_TOKEN = st.secrets["BOT_TOKEN"]
        CHAT_ID = st.secrets["CHAT_ID"]

        if not BOT_TOKEN or not CHAT_ID:
            st.warning("Telegram credentials not configured")
            return False

        now = get_ist_time().strftime('%d %b %Y, %I:%M %p IST')
        message = f"📊 *Market Scanner Report*\n🕒 *Scanned at:* {now}\n"

        def format_section(title, df):
            # Handle symbol column case-insensitively
            symbol_col = None
            for col in df.columns:
                if col.lower() == "symbol":
                    symbol_col = col
                    break
            if symbol_col is None:
                return "", []

            df = df[df[symbol_col].notna()]
            df = df[df[symbol_col].astype(str).str.strip() != ""]
            if df.empty:
                return "", []

            lines = [f"\n*{title}:*"]
            buttons = []
            for _, row in df.iterrows():
                symbol = str(row[symbol_col]).strip()
                if symbol and symbol.lower() != "nan" and symbol.upper() != "N/A":
                    symbol = symbol.replace(".NS", "")  # Optional: clean .NS for cleaner view
                    lines.append(f"• {symbol} [🔗 Chart](https://www.tradingview.com/chart/?symbol=NSE:{symbol})")
                    buttons.append({
                        "text": f"{symbol}",
                        "url": f"https://www.tradingview.com/chart/?symbol=NSE:{symbol}"
                    })
            return "\n".join(lines), buttons

        sections = []
        all_buttons = []

        # MACD 4H
        if "MACD 4h" in scan_results:
            df = scan_results["MACD 4h"]
            if isinstance(df, pd.DataFrame) and not df.empty:
                sec, btns = format_section("MACD 4H Crossover", df)
                if sec: sections.append(sec)
                all_buttons.extend(btns)

        # MACD 1D
        if "MACD 1d" in scan_results:
            df = scan_results["MACD 1d"]
            if isinstance(df, pd.DataFrame) and not df.empty:
                sec, btns = format_section("MACD 1D Crossover", df)
                if sec: sections.append(sec)
                all_buttons.extend(btns)

        # Range Breakout 4H
        if "Range Breakout 4h" in scan_results:
            df = scan_results["Range Breakout 4h"]
            if isinstance(df, pd.DataFrame) and not df.empty:
                sec, btns = format_section("Range Breakout 4H", df)
                if sec: sections.append(sec)
                all_buttons.extend(btns)

        # Resistance Breakout 4h
        if "Resistance Breakout 4h" in scan_results:
            df = scan_results["Resistance Breakout 4h"]
            if isinstance(df, pd.DataFrame) and not df.empty and "Distance_to_Resistance_%" in df.columns:
                filtered_df = df[df["Distance_to_Resistance_%"] < 2]
                if "Signal_Type" in filtered_df.columns:
                    retrace_df = filtered_df[filtered_df["Signal_Type"].str.contains("retracement", case=False, na=False)]
                    fresh_df = filtered_df[filtered_df["Signal_Type"].str.contains("fresh", case=False, na=False)]
                else:
                    retrace_df = filtered_df
                    fresh_df = pd.DataFrame()
                if not retrace_df.empty:
                    sec, btns = format_section("Resistance Breakout (Retracement <2%)", retrace_df)
                    if sec: sections.append(sec)
                    all_buttons.extend(btns)
                if not fresh_df.empty:
                    sec, btns = format_section("Resistance Breakout (Fresh Entry <2%)", fresh_df)
                    if sec: sections.append(sec)
                    all_buttons.extend(btns)

        # Support Level 4h
        if "Support Level 4h" in scan_results:
            df = scan_results["Support Level 4h"]
            if isinstance(df, pd.DataFrame) and not df.empty and "Distance_to_Support_%" in df.columns:
                near_df = df[df["Distance_to_Support_%"] < 2]
                sec, btns = format_section("Support Level 4H (Near Support <2%)", near_df)
                if sec: sections.append(sec)
                all_buttons.extend(btns)

        # Final message
        message += "\n".join(sections) if sections else "\n_No signals found._"

        # Payload
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }

        # Buttons (2 per row)
        if all_buttons:
            inline_keyboard = []
            for i in range(0, len(all_buttons), 2):
                inline_keyboard.append(all_buttons[i:i+2])
            payload["reply_markup"] = {"inline_keyboard": inline_keyboard}

        # Send
        response = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json=payload
        )

        if response.status_code != 200:
            st.error(f"Telegram API error: {response.text}")
            return False

        return True

    except Exception as e:
        st.error(f"Failed to send Telegram notification: {str(e)}")
        return False






def handle_auto_scan():
    """Handle automatic scanning with immediate first scan"""
    current_time = get_ist_time()
    
    # First run - scan immediately if no previous scan time
    if st.session_state.last_scan_time is None:
        run_all_scanners()
        st.session_state.last_scan_time = current_time
        time.sleep(1)  # Small delay before refresh
        st.rerun()
    else:
        # Subsequent runs - check if 15 minutes have passed
        next_scan = st.session_state.last_scan_time + timedelta(minutes=15)
        if current_time >= next_scan:
            run_all_scanners()
            st.session_state.last_scan_time = current_time
            time.sleep(1)  # Small delay before refresh
            st.rerun()
    
    # Remove the visual indicator from here since we're handling it in update_counters
    time.sleep(1)


def export_results():
    """Export scan results to CSV"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for scanner_name, results in st.session_state.scan_results.items():
            if isinstance(results, pd.DataFrame) and not results.empty:
                filename = f"{scanner_name.replace(' ', '_')}_{timestamp}.csv"
                results.to_csv(filename, index=False)
                
                # Provide download link
                with open(filename, 'rb') as f:
                    st.download_button(
                        label=f"Download {scanner_name} Results",
                        data=f.read(),
                        file_name=filename,
                        mime='text/csv'
                    )
        
        st.success("✅ Export completed!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")



if __name__ == "__main__":
    main()
