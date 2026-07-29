import streamlit as st
import streamlit.components.v1 as components
import ccxt
import pandas as pd
import numpy as np
from ta.trend import ema_indicator
from ta.volatility import average_true_range
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# --- تنظیمات ظاهر صفحه ---
st.set_page_config(page_title="AI Breakout Pro Dashboard", page_icon="🤖", layout="wide")
st.title("🚀 داشبورد پیشرفته تحلیل شکست با هوش مصنوعی (نسخه کامل و جامع)")
st.markdown("مجهز به ماشین‌حساب زنده سفارشات، ساعت جهانی صرافی و قابلیت پشتیبانی از تمام ارزهای بازار اسپات.")

# --- تابع استخراج خودکار تمام جفت ارزهای بازار اسپات صرافی ---
@st.cache_data(ttl=86400) # ذخیره لیست ارزها در حافظه موقت برای 24 ساعت تا سایت کند نشود
def get_all_pairs():
    try:
        exchange_temp = ccxt.bybit()
        markets = exchange_temp.load_markets()
        usdt_pairs = [market['symbol'] for market in markets.values() if market['spot'] and market['quote'] == 'USDT']
        return sorted(usdt_pairs)
    except:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "BNB/USDT", "LINK/USDT"]

available_pairs = get_all_pairs()
default_idx = available_pairs.index("BTC/USDT") if "BTC/USDT" in available_pairs else 0

# --- بخش سایدبار (تنظیمات کاربر و ساعت زنده) ---
st.sidebar.header("🌍 زمان زنده بازار (تریدینگ‌ویو)")
components.html(
    """
    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #f39c12; color: #f39c12; font-family: sans-serif; margin-bottom: 10px;">
        <span style="font-size: 14px; color: #ffffff;">ساعت رسمی UTC</span><br>
        <span id="utc_clock" style="font-size: 26px; font-weight: bold; letter-spacing: 2px;"></span>
    </div>
    <script>
        function updateTime() {
            const now = new Date();
            const hours = String(now.getUTCHours()).padStart(2, '0');
            const minutes = String(now.getUTCMinutes()).padStart(2, '0');
            const seconds = String(now.getUTCSeconds()).padStart(2, '0');
            document.getElementById('utc_clock').innerText = hours + ':' + minutes + ':' + seconds;
        }
        setInterval(updateTime, 1000);
        updateTime();
    </script>
    """,
    height=100
)

st.sidebar.header("⚙️ تنظیمات اصلی ربات")
# استفاده از لیست کامل ارزها بجای محدودیت ۳ تایی
symbol = st.sidebar.selectbox("انتخاب جفت ارز", available_pairs, index=default_idx)
timeframe = st.sidebar.selectbox("تایم‌فریم", ["1h", "30m", "15m"], index=0)
target_limit = st.sidebar.slider("تعداد کندل تاریخی برای آموزش", min_value=1000, max_value=5000, step=1000, value=3000)

st.sidebar.markdown("---")
st.sidebar.header("⏰ تنظیمات استراتژی زمانی")
use_time_filter = st.sidebar.checkbox("فعال‌سازی فیلتر زمان معامله", value=False)
target_hour = st.sidebar.number_input("ساعت مجاز معامله (بر اساس UTC صرافی)", min_value=0, max_value=23, value=21, step=1) if use_time_filter else None

st.sidebar.markdown("---")
st.sidebar.header("💰 تنظیمات بک‌تست مالی")
initial_capital = st.sidebar.number_input("سرمایه اولیه (دلار)", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)
risk_per_trade_pct = st.sidebar.number_input("ریسک در هر معامله (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
fee_rate = st.sidebar.number_input("کارمزد صرافی (درصد)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100.0

run_btn = st.sidebar.button("🔄 اجرای تحلیل کامل و بک‌تست")

if run_btn:
    with st.spinner(f"در حال دریافت اطلاعات و تحلیل ارز {symbol}..."):
        
        # 1. دریافت دیتا
        exchange = ccxt.bybit()
        
        all_bars = []
        since = None
        while len(all_bars) < target_limit:
            limit = min(1000, target_limit - len(all_bars))
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not bars: break
            if since == bars[0][0]: break
            since = bars[-1][0] + 1
            all_bars.extend(bars)
            if len(bars) < limit: break
                
        df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # 2. مهندسی ویژگی‌ها
        df['ema_200'] = ema_indicator(df['close'], window=200)
        df['atr_14'] = average_true_range(df['high'], df['low'], df['close'], window=14)
        
        df['ref_high'] = df['high'].shift(1)
        df['ref_low'] = df['low'].shift(1)
        df['ref_close'] = df['close'].shift(1)
        df['candle_len'] = df['ref_high'] - df['ref_low']
        df['prev_atr'] = df['atr_14'].shift(1)
        df['prev_ema'] = df['ema_200'].shift(1)
        
        df['candle_len_ratio'] = df['candle_len'] / df['prev_atr'] 
        df['dist_ema'] = (df['ref_close'] - df['prev_ema']) / df['prev_ema'] 
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'] 
        df['uptrend'] = np.where(df['ref_close'] > df['prev_ema'], 1, 0)
        df['downtrend'] = np.where(df['ref_close'] < df['prev_ema'], 1, 0)
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['valid_size'] = np.where((df['candle_len_ratio'] > 0.5) & (df['candle_len_ratio'] < 2.0), 1, 0)
        df.dropna(inplace=True)
        
        # 3. برچسب‌گذاری (ریسک به ریوارد 1:2)
        labels = []
        df_reset = df.reset_index(drop=True)
        max_holding_bars = 24
        
        for i in range(len(df_reset) - max_holding_bars):
            row = df_reset.iloc[i]
            target = 0 
            if row['valid_size'] == 1:
                buffer = row['prev_atr'] * 0.1
                if row['uptrend'] == 1:
                    entry_l = row['ref_high'] + buffer
                    sl_l = row['ref_low'] - buffer
                    tp_l = entry_l + ((entry_l - sl_l) * 2.0)
                    for j in range(i + 1, i + max_holding_bars):
                        f_row = df_reset.iloc[j]
                        if f_row['high'] >= entry_l:
                            if f_row['high'] >= tp_l and f_row['low'] > sl_l: target = 1; break
                            elif f_row['low'] <= sl_l: target = -1; break
                elif row['downtrend'] == 1:
                    entry_s = row['ref_low'] - buffer
                    sl_s = row['ref_high'] + buffer
                    tp_s = entry_s - ((sl_s - entry_s) * 2.0)
                    for j in range(i + 1, i + max_holding_bars):
                        f_row = df_reset.iloc[j]
                        if f_row['low'] <= entry_s:
                            if f_row['low'] <= tp_s and f_row['high'] < sl_s: target = 1; break
                            elif f_row['high'] >= sl_s: target = -1; break
            labels.append(target)
            
        while len(labels) < len(df_reset): labels.append(0)
        df_reset['ml_target'] = labels
        
        # 4. آموزش مدل
        df_model = df_reset[df_reset['ml_target'] != 0].copy()
        df_model['target_bin'] = np.where(df_model['ml_target'] == 1, 1, 0)
        
        feature_cols = ['candle_len_ratio', 'dist_ema', 'volume_ratio', 'uptrend', 'hour_of_day']
        X = df_model[feature_cols]
        y = df_model['target_bin']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # --- موتور بک‌تست ---
        test_indices = X_test.index
        df_test_period = df_reset.loc[test_indices].sort_values('timestamp')
        df_test_period['ai_signal'] = model.predict(df_test_period[feature_cols])
        
        current_capital = initial_capital
        equity_curve = [{'timestamp': df_test_period['timestamp'].iloc[0], 'equity': current_capital}]
        trades_won = 0
        trades_lost = 0
        total_fees_paid = 0
        
        for index, row in df_test_period.iterrows():
            time_condition_met = True
            if use_time_filter and row['hour_of_day'] != target_hour:
                time_condition_met = False
                
            if row['ai_signal'] == 1 and time_condition_met and row['ml_target'] != 0:
                buffer = row['prev_atr'] * 0.1
                if row['uptrend'] == 1:
                    entry = row['ref_high'] + buffer
                    sl = row['ref_low'] - buffer
                    tp = entry + ((entry - sl) * 2.0)
                else:
                    entry = row['ref_low'] - buffer
                    sl = row['ref_high'] + buffer
                    tp = entry - ((sl - entry) * 2.0)
                
                distance_sl = abs(entry - sl)
                risk_amount = current_capital * (risk_per_trade_pct / 100.0)
                
                ideal_position_usd = (risk_amount * entry) / distance_sl if distance_sl > 0 else 0
                actual_position_usd = min(current_capital, ideal_position_usd)
                actual_position_crypto = actual_position_usd / entry if entry > 0 else 0
                
                entry_fee_usd = actual_position_usd * fee_rate
                
                if row['ml_target'] == 1:
                    exit_fee_usd = (actual_position_crypto * tp) * fee_rate
                    gross_profit = actual_position_crypto * abs(tp - entry)
                    net_profit = gross_profit - entry_fee_usd - exit_fee_usd
                    current_capital += net_profit
                    trades_won += 1
                    total_fees_paid += (entry_fee_usd + exit_fee_usd)
                else:
                    exit_fee_usd = (actual_position_crypto * sl) * fee_rate
                    gross_loss = actual_position_crypto * abs(entry - sl)
                    net_loss = gross_loss + entry_fee_usd + exit_fee_usd
                    current_capital -= net_loss
                    trades_lost += 1
                    total_fees_paid += (entry_fee_usd + exit_fee_usd)
                
                equity_curve.append({'timestamp': row['timestamp'], 'equity': current_capital})
        
        # 5. تحلیل زنده بازار
        latest_row = df_reset.iloc[[-2]].copy()
        X_live = latest_row[feature_cols]
        prediction = model.predict(X_live)[0]
        probability = model.predict_proba(X_live)[0]
        current_hour = latest_row['hour_of_day'].values[0]
        
        signal_allowed = True
        if use_time_filter and current_hour != target_hour:
            signal_allowed = False
        
        # ==========================================
        # رندر داشبورد گرافیکی
        # ==========================================
        st.success(f"✅ پردازش هوش مصنوعی برای ارز {symbol} با موفقیت به پایان رسید.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="وضعیت روند فعلی (EMA)", value="صعودی" if latest_row['uptrend'].values[0] == 1 else "نزولی")
        with col2:
            conf_val = round(probability[1] * 100, 1) if prediction == 1 else round(probability[0] * 100, 1)
            st.metric(label="اطمینان هوش مصنوعی", value=f"{conf_val}%")
        with col3:
            if not signal_allowed: signal_text = f"🟡 منتظر ساعت {target_hour}:00"
            elif prediction == 1 and latest_row['uptrend'].values[0] == 1: signal_text = "🟢 سیگنال LONG"
            elif prediction == 1 and latest_row['uptrend'].values[0] == 0: signal_text = "🔴 سیگنال SHORT"
            else: signal_text = "⚪ صبر کنید"
            st.metric(label="دستور لایو سیستم", value=signal_text)
            
        # کادر نمایش اعداد دقیق سفارشات
        if signal_allowed and prediction == 1:
            buffer = latest_row['prev_atr'].values[0] * 0.1
            if latest_row['uptrend'].values[0] == 1:
                entry = latest_row['ref_high'].values[0] + buffer
                sl = latest_row['ref_low'].values[0] - buffer
                tp = entry + ((entry - sl) * 2.0)
            else:
                entry = latest_row['ref_low'].values[0] - buffer
                sl = latest_row['ref_high'].values[0] + buffer
                tp = entry - ((sl - entry) * 2.0)
                
            st.info(f"🎯 **مشخصات سفارش‌گذاری در صرافی برای {symbol}:** \n\n"
                    f"🛒 **نقطه ورود (Entry Order):** `{round(entry, 4)}` \n\n"
                    f"🛑 **حد ضرر (Stop Loss):** `{round(sl, 4)}` \n\n"
                    f"🏆 **حد سود (Take Profit):** `{round(tp, 4)}`")
            
            # --- کدهای جدید: رسم نمودار گرافیکی سیگنال ---
            st.markdown("### 📊 نمودار زنده سیگنال (نقاط ورود و خروج)")
            
            # استخراج ۵۰ کندل آخر برای نمایش بهتر
            df_plot = df_reset.tail(50)
            
            fig = go.Figure(data=[go.Candlestick(x=df_plot['timestamp'],
                            open=df_plot['open'],
                            high=df_plot['high'],
                            low=df_plot['low'],
                            close=df_plot['close'],
                            name="نوسان قیمت")])
            
            # رسم خط آبی (نقطه ورود)
            fig.add_hline(y=entry, line_dash="dash", line_color="blue", annotation_text=f"ورود (Entry): {round(entry, 4)}", annotation_position="top right")
            
            # رسم خط سبز (حد سود)
            fig.add_hline(y=tp, line_dash="solid", line_color="green", annotation_text=f"حد سود (TP): {round(tp, 4)}", annotation_position="top right")
            
            # رسم خط قرمز (حد ضرر)
            fig.add_hline(y=sl, line_dash="solid", line_color="red", annotation_text=f"حد ضرر (SL): {round(sl, 4)}", annotation_position="bottom right")
            
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            # ----------------------------------------------
            
        st.markdown("---")
        
        # --- نمایش گزارش بک‌تست ---
        st.header("📈 گزارش شبیه‌ساز مالی و وضعیت استراتژی")
        
        total_trades = trades_won + trades_lost
        win_rate = round((trades_won / total_trades * 100), 1) if total_trades > 0 else 0
        net_profit_total = current_capital - initial_capital
        roi_pct = round((net_profit_total / initial_capital) * 100, 2)
        
        df_equity = pd.DataFrame(equity_curve)
        if len(df_equity) > 1:
            df_equity['peak'] = df_equity['equity'].cummax()
            df_equity['drawdown'] = (df_equity['peak'] - df_equity['equity']) / df_equity['peak']
            max_dd = round(df_equity['drawdown'].max() * 100, 2)
        else: max_dd = 0

        bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
        bcol1.metric("موجودی نهایی", f"${round(current_capital, 2)}")
        bcol2.metric("سود/ضرر خالص (ROI)", f"{roi_pct}%")
        bcol3.metric("تعداد معاملات", f"{total_trades} ترید")
        bcol4.metric("وین‌ریت", f"{win_rate}%")
        bcol5.metric("حداکثر افت (Drawdown)", f"{max_dd}%")
        
        if len(df_equity) > 1:
            st.line_chart(df_equity.set_index('timestamp')['equity'], use_container_width=True)

else:
    st.info("👈 ارز مورد نظر و پارامترهای خود را تنظیم کرده و روی دکمه اجرای تحلیل کلیک کنید.")
