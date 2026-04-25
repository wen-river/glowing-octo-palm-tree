import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

import akshare as ak
import streamlit as st

# ================== 页面设置 ==================
st.set_page_config(
    page_title="量化选股Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main .block-container {padding: 1rem 0.5rem;}
    .stButton>button {
        width:100%; background:#f63366; color:white; font-weight:bold;
        border-radius:8px; padding:0.6rem;
    }
    .stButton>button:hover {background:#e62e5c;}
</style>
""", unsafe_allow_html=True)

# ================== 默认策略配置 ==================
DEFAULT_CONFIG = {
    "trend": {
        "ema": {
            "enabled": True,
            "fast": 8, "slow": 21,
            "score_ema_fast_above_slow": 0.4,
            "score_price_above_fast": 0.3,
            "score_macd_hist_positive": 0.3
        }
    },
    "momentum": {
        "rsi": {
            "enabled": True,
            "lower": 40, "upper": 65,
            "score_in_range_min": 0.3,
            "score_in_range_max": 0.6,
            "score_30_40": 0.2
        },
        "kdj": {
            "enabled": True,
            "score_k_above_d": 0.2,
            "score_j_above_k": 0.1
        }
    },
    "volume": {
        "enabled": True,
        "ratio_2_0": 1.0,
        "ratio_1_5": 0.8,
        "ratio_1_2": 0.6,
        "ratio_1_0": 0.3,
        "ratio_below": 0.1
    },
    "risk": {
        "atr": {
            "enabled": True,
            "penalty_5pct": 0.3,
            "penalty_3pct": 0.15
        }
    },
    "extra": {
        "wr": {"enabled": False, "threshold": -80, "score": 0.15},
        "bollinger": {"enabled": False, "score": 0.2},
        "ema55_bullish": {"enabled": False, "score": 0.1}
    },
    "weights": {
        "trend": 0.35, "momentum": 0.30,
        "volume": 0.25, "risk": 0.10
    },
    "buy_threshold": 0.65
}

# ================== 技术指标纯自实现（无任何第三方库依赖） ==================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    vol = df['volume'].astype(float)

    # EMA
    df['ema8'] = close.ewm(span=8, adjust=False).mean()
    df['ema21'] = close.ewm(span=21, adjust=False).mean()
    df['ema55'] = close.ewm(span=55, adjust=False).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # KDJ
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-9) * 100.0
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = 3.0 * k - 2.0 * d

    # ATR
    tr = np.maximum(high - low,
                    np.maximum(abs(high - close.shift(1)),
                               abs(low - close.shift(1))))
    df['atr'] = tr.rolling(14).mean()

    # 量比
    ma20_vol = vol.rolling(20).mean()
    df['volume_ratio'] = vol / ma20_vol

    # 威廉 WR
    h14 = high.rolling(14).max()
    l14 = low.rolling(14).min()
    df['wr'] = (h14 - close) / (h14 - l14 + 1e-9) * (-100.0)

    # 布林带 (20,2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BOLU'] = ma20 + 2 * std20
    df['BOLD'] = ma20 - 2 * std20

    return df.dropna()

# ================== 信号评分 ==================
class SignalEngine:
    @staticmethod
    def score(row, config):
        # 趋势得分
        t = 0.0
        if config['trend']['ema']['enabled']:
            c = config['trend']['ema']
            if row['ema8'] > row['ema21']:
                t += c['score_ema_fast_above_slow']
            if row['close'] > row['ema8']:
                t += c['score_price_above_fast']
            if row['macd_hist'] > 0:
                t += c['score_macd_hist_positive']
        t = min(t, 1.0)

        # 动量得分
        m = 0.0
        rsi_cfg = config['momentum']['rsi']
        if rsi_cfg['enabled']:
            rsi = row['rsi']
            if rsi_cfg['lower'] <= rsi <= rsi_cfg['upper']:
                pos = (rsi - rsi_cfg['lower']) / (rsi_cfg['upper'] - rsi_cfg['lower'])
                m += rsi_cfg['score_in_range_min'] + pos * (rsi_cfg['score_in_range_max'] - rsi_cfg['score_in_range_min'])
            elif 30 <= rsi < rsi_cfg['lower']:
                m += rsi_cfg['score_30_40']
        if config['momentum']['kdj']['enabled']:
            if row['kdj_k'] > row['kdj_d']:
                m += config['momentum']['kdj']['score_k_above_d']
                if row['kdj_j'] > row['kdj_k']:
                    m += config['momentum']['kdj']['score_j_above_k']
        m = min(m, 1.0)

        # 量能得分
        v = 0.0
        if config['volume']['enabled']:
            vr = row['volume_ratio']
            c = config['volume']
            if vr >= 2.0:   v = c['ratio_2_0']
            elif vr >= 1.5: v = c['ratio_1_5']
            elif vr >= 1.2: v = c['ratio_1_2']
            elif vr >= 1.0: v = c['ratio_1_0']
            else:           v = c['ratio_below']

        # 风险惩罚
        p = 0.0
        if config['risk']['atr']['enabled']:
            atr_pct = row['atr'] / row['close'] if row['close'] else 0
            if atr_pct > 0.05:
                p += config['risk']['atr']['penalty_5pct']
            elif atr_pct > 0.03:
                p += config['risk']['atr']['penalty_3pct']
        p = min(p, 1.0)

        # 额外加分
        extra = 0.0
        ex = config.get('extra', {})
        if ex.get('wr', {}).get('enabled') and row['wr'] <= ex['wr']['threshold']:
            extra += ex['wr']['score']
        if ex.get('bollinger', {}).get('enabled'):
            if row['close'] > row['BOLU']:
                extra += ex['bollinger']['score']
        if ex.get('ema55_bullish', {}).get('enabled') and row['ema8'] > row['ema21'] > row['ema55']:
            extra += ex['ema55_bullish']['score']

        w = config['weights']
        tfbi = (w['trend'] * t + w['momentum'] * m +
                w['volume'] * v - w['risk'] * p) + extra
        tfbi = round(max(0.0, min(tfbi, 1.0)), 3)

        return {
            'TFBI': tfbi,
            'trend': round(t, 2), 'momentum': round(m, 2),
            'volume': round(v, 2), 'risk': round(p, 2),
            'extra': round(extra, 2),
            'buy': tfbi >= config['buy_threshold'],
            'close': row['close']
        }

# ================== 数据获取 ==================
@st.cache_data(ttl=600)
def get_hot_stocks(n=100):
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.sort_values('总市值', ascending=False).head(n)
        return list(zip(df['代码'], df['名称']))
    except:
        return [('000001','平安银行'), ('000858','五粮液'), ('300750','宁德时代'),
                ('600519','贵州茅台'), ('601318','中国平安'), ('000333','美的集团')]

@st.cache_data(ttl=3600)
def get_stock_name(code):
    try:
        df = ak.stock_info_a_code_name()
        match = df[df['code'] == code]
        return match.iloc[0]['name'] if not match.empty else ""
    except:
        return ""

@st.cache_data(ttl=600)
def fetch_indicators(code):
    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df.empty: return None
        df = df.rename(columns={'日期':'date','开盘':'open','收盘':'close',
                                '最高':'high','最低':'low','成交量':'volume'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = calc_indicators(df)
        return df.iloc[-1] if len(df) >= 30 else None
    except:
        return None

# ================== Session State ==================
if 'custom_stocks' not in st.session_state:
    st.session_state.custom_stocks = []
if 'config' not in st.session_state:
    st.session_state.config = DEFAULT_CONFIG.copy()
if 'saved_configs' not in st.session_state:
    st.session_state.saved_configs = {"默认": DEFAULT_CONFIG.copy()}

# ================== 主界面 ==================
st.title("📈 量化选股Pro")
st.caption("趋势共振突破 · Python 3.14 兼容版")

tab1, tab2, tab3, tab4 = st.tabs(["📋 股票池", "⚡ 扫描", "🔍 单股", "⚙️ 策略"])

with tab1:
    st.subheader("自选股票池")
    with st.form("add_form", clear_on_submit=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            new_code = st.text_input("股票代码", placeholder="如 600519")
        with c2:
            submitted = st.form_submit_button("➕ 添加")
        if submitted and new_code:
            code = new_code.strip().zfill(6)
            if code in [c for c,n in st.session_state.custom_stocks]:
                st.warning("已在池中")
            else:
                name = get_stock_name(code)
                if not name:
                    st.error("未获取到名称")
                else:
                    st.session_state.custom_stocks.append((code, name))
                    st.success(f"已添加 {code} {name}")
    if st.session_state.custom_stocks:
        st.subheader(f"共 {len(st.session_state.custom_stocks)} 只")
        for i, (code, name) in enumerate(st.session_state.custom_stocks):
            ca, cb, cc = st.columns([2, 3, 1])
            ca.write(f"**{code}**")
            cb.write(name)
            if cc.button("🗑", key=f"del_{i}"):
                st.session_state.custom_stocks.pop(i)
                st.rerun()
        if st.button("🗑 清空全部"):
            st.session_state.custom_stocks.clear()
            st.rerun()
    else:
        st.info("股票池为空")

with tab2:
    st.subheader("扫描池")
    src = st.radio("来源", ["📁 我的股票池", "🔥 热门股 Top N"], horizontal=True)
    targets, label = [], ""
    if src == "🔥 热门股 Top N":
        n = st.slider("数量", 20, 200, 80, 10)
        targets = get_hot_stocks(n)
        label = f"热门股 Top{len(targets)}"
    else:
        targets = st.session_state.custom_stocks.copy()
        label = f"自选池({len(targets)}只)"
        if not targets:
            st.warning("自选池为空")

    if st.button("🔍 开始扫描", use_container_width=True):
        if not targets:
            st.error("无股票")
        else:
            progress_cont = st.empty()
            bar = progress_cont.progress(0)
            status = progress_cont.empty()
            results = []
            total = len(targets)
            cfg = st.session_state.config
            for i, (code, name) in enumerate(targets):
                status.text(f"({i+1}/{total}) {code} {name}")
                row = fetch_indicators(code)
                if row is not None:
                    res = SignalEngine.score(row, cfg)
                    if res['buy']:
                        results.append((code, name, res))
                bar.progress((i+1)/total)
            progress_cont.empty()
            if results:
                st.success(f"✅ 发现 {len(results)} 个信号")
                results.sort(key=lambda x: x[2]['TFBI'], reverse=True)
                for code, name, r in results:
                    with st.expander(f"✅ {code} {name}"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("TFBI", r['TFBI'])
                        c2.metric("趋势", f"{r['trend']:.0%}")
                        c3.metric("动量", f"{r['momentum']:.0%}")
                        st.write(f"量能 {r['volume']:.0%} | 风险 {r['risk']:.0%} | 额外 +{r['extra']:.0%}")
                        st.caption(f"最新价 {r['close']:.2f}")
            else:
                st.info("无买入信号")

    if st.button("🧹 清除缓存"):
        st.cache_data.clear()
        st.success("缓存已清除")

with tab3:
    st.subheader("单股诊断")
    code = st.text_input("代码", "000001")
    if st.button("检测", use_container_width=True):
        row = fetch_indicators(code.strip().zfill(6))
        if row is None:
            st.error("数据获取失败")
        else:
            res = SignalEngine.score(row, st.session_state.config)
            st.subheader(f"诊断：{code}")
            c1, c2 = st.columns(2)
            c1.metric("TFBI", res['TFBI'])
            c2.metric("价格", f"{res['close']:.2f}")
            tcol, mcol, vcol, rcol = st.columns(4)
            tcol.metric("趋势", f"{res['trend']:.0%}")
            mcol.metric("动量", f"{res['momentum']:.0%}")
            vcol.metric("量能", f"{res['volume']:.0%}")
            rcol.metric("风险", f"{res['risk']:.0%}", delta="-", delta_color="inverse")
            if res['buy']:
                st.success("🎯 买入信号！")
            else:
                st.info("未触发信号")

with tab4:
    st.subheader("策略配置")
    saved = list(st.session_state.saved_configs.keys())
    sel = st.selectbox("加载方案", saved, index=saved.index("默认"))
    if st.button("加载"):
        st.session_state.config = st.session_state.saved_configs[sel]
        st.rerun()

    cfg = st.session_state.config
    with st.expander("趋势"):
        cfg['trend']['ema']['enabled'] = st.checkbox("启用", value=cfg['trend']['ema']['enabled'])
        if cfg['trend']['ema']['enabled']:
            cfg['trend']['ema']['fast'] = st.slider("快线", 3,20, cfg['trend']['ema']['fast'])
            cfg['trend']['ema']['slow'] = st.slider("慢线", 5,50, cfg['trend']['ema']['slow'])
    with st.expander("动量"):
        cfg['momentum']['rsi']['enabled'] = st.checkbox("RSI", value=cfg['momentum']['rsi']['enabled'])
        if cfg['momentum']['rsi']['enabled']:
            cfg['momentum']['rsi']['lower'] = st.slider("下限", 10,50, cfg['momentum']['rsi']['lower'])
            cfg['momentum']['rsi']['upper'] = st.slider("上限", 50,90, cfg['momentum']['rsi']['upper'])
        cfg['momentum']['kdj']['enabled'] = st.checkbox("KDJ", value=cfg['momentum']['kdj']['enabled'])
    with st.expander("量能 & 风控"):
        cfg['volume']['enabled'] = st.checkbox("成交量", value=cfg['volume']['enabled'])
        cfg['risk']['atr']['enabled'] = st.checkbox("波动率惩罚", value=cfg['risk']['atr']['enabled'])
    with st.expander("权重 & 阈值"):
        cfg['weights']['trend'] = st.slider("趋势权重", 0.0,1.0, cfg['weights']['trend'], 0.05)
        cfg['weights']['momentum'] = st.slider("动量权重", 0.0,1.0, cfg['weights']['momentum'], 0.05)
        cfg['weights']['volume'] = st.slider("量能权重", 0.0,1.0, cfg['weights']['volume'], 0.05)
        cfg['weights']['risk'] = st.slider("风控权重", 0.0,1.0, cfg['weights']['risk'], 0.05)
        cfg['buy_threshold'] = st.slider("买入阈值", 0.3,0.9, cfg['buy_threshold'], 0.01)

    col_save, col_reset = st.columns(2)
    with col_save:
        new_name = st.text_input("方案名", "自定义")
        if st.button("💾 保存"):
            st.session_state.saved_configs[new_name] = cfg.copy()
            st.success(f"方案“{new_name}”已保存")
    with col_reset:
        if st.button("🔄 重置"):
            st.session_state.config = DEFAULT_CONFIG.copy()
            st.session_state.saved_configs["默认"] = DEFAULT_CONFIG.copy()
            st.rerun()

st.divider()
st.caption("⚠️ 本App仅供学习研究，不构成投资建议。")
st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")