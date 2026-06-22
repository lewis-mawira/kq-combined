import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="KQ Wellness Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f4f7f9; }

    .kpi-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border-bottom: 4px solid #D71920;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-icon { font-size: 2rem; margin-bottom: 6px; }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: #002147; margin: 4px 0; }
    .kpi-label { font-size: 0.78rem; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-year { font-size: 0.7rem; color: #888; font-weight: 600; margin-top: 0; }

    .section-header {
        background: linear-gradient(90deg, #002147, #00306e);
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        margin: 28px 0 14px 0;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }

    .year-badge-2022 { background:#002147; color:white; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:700; }
    .year-badge-2025 { background:#D71920; color:white; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:700; }
    .year-badge-2026 { background:#1a7a4a; color:white; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:700; }

    .delta-up { color: #1a7a4a; font-weight: 700; }
    .delta-down { color: #D71920; font-weight: 700; }

    .feedback-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 12px;
        border-left: 6px solid #D71920;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.07);
    }
    .new-badge {
        background: #e8f5e9; color: #1a7a4a;
        border: 1px solid #1a7a4a;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.72rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# COLOUR PALETTE
# =========================
COL_2022 = "#002147"
COL_2025 = "#D71920"
COL_2026 = "#1a7a4a"
YEAR_COLORS = {'2022': COL_2022, '2025': COL_2025, '2026': COL_2026}

# =========================
# HELPERS
# =========================
def parse_sleep(val):
    if pd.isna(val): return np.nan
    val = str(val).lower()
    if 'less than 4' in val: return 3.5
    if '4 - 6' in val or '4-6' in val: return 5.0
    if '7 - 9' in val or '7-9' in val: return 8.0
    if '9+' in val: return 10.0
    return np.nan

JUNK = {'no response', 'na', 'n/a', 'none', 'nil', '-', 'not applicable', '', 'nan'}

def clean(series):
    return series[~series.astype(str).str.lower().str.strip().isin(JUNK)].dropna()

def pct(series, values):
    s = clean(series)
    if len(s) == 0: return 0
    return (s.isin(values)).sum() / len(s) * 100

def delta_html(old, new, unit="%", higher_is_better=True):
    diff = new - old
    arrow = "▲" if diff >= 0 else "▼"
    css = "delta-up" if (diff >= 0) == higher_is_better else "delta-down"
    sign = "+" if diff >= 0 else ""
    return f'<span class="{css}">{arrow} {sign}{diff:.1f}{unit}</span>'

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    # --- 2022 ---
    df22 = pd.read_csv('KQ Mental Wellness 2022.csv')
    df22.columns = df22.columns.str.strip()
    df22 = df22[df22['Please select your gender'].notna()]

    # --- 2025 ---
    df25 = pd.read_csv('KQ 2025.csv')
    df25.columns = df25.columns.str.replace('  ', ' ').str.strip()
    consent_col = [c for c in df25.columns if "consenting to us" in c.lower()][0]
    # NOTE: must match "I Agree" exactly — "Agree" alone also matches "I Disagree"
    df25 = df25[df25[consent_col].str.strip().str.lower() == "i agree"]
    df25 = df25.rename(columns={
        'Mental Wellness Issues Would Like Addressed - Emotional intelligence': 'MW_Issue_Emotional_Intelligence',
        'Mental Wellness Issues Would Like Addressed - Financial constraints': 'MW_Issue_Financial_Constraints',
        'Mental Wellness Issues Would Like Addressed - Trauma': 'MW_Issue_Trauma',
        'Mental Wellness Issues Would Like Addressed - Substance and Drug Dependence': 'MW_Issue_Substance_Drug_Dependence',
        'Mental Wellness Issues Would Like Addressed - Relationship Difficulties': 'MW_Issue_Relationship_Difficulties',
        'Mental Wellness Issues Would Like Addressed - Unresolved Childhood Difficulties': 'MW_Issue_Unresolved_Childhood_Difficulties',
        'Mental Wellness Issues Would Like Addressed - Anxiety and Depression': 'MW_Issue_Anxiety_Depression',
        'Mental Wellness Issues Would Like Addressed - Other (please specify)': 'MW_Issue_Other',
        'Coping Mechanisms - Other (please specify)': 'Coping_Mechanisms_Other',
        'Preferred Method of Receiving Mental Health Information - Other (please specify)': 'MH_Info_Preference_Other',
        'EAP Awareness & Usage - If yes to Question 29, have you ever accessed the service?': 'If yes to Question 28, have you ever accessed the service?',
        'EAP Awareness & Usage - If no, please state the reason.': 'If no, please state the reason.',
        'During the past two weeks, how often has your mental health affected your relationships?': 'During the past 2 weeks, how often has your mental health affected your relationships?',
        'What about the amount of social support you receive from your family, friends, etc? When you have the need to talk to someone or go on outings with friends and/or relatives, do you feel there is someone who fulfills these needs?':
            'To what extent do you feel you have people you can rely on for emotional or practical support when you need it? (This may include family, friends, colleagues, or community members, e.g., church, mosque, club, etc)',
    })

    # --- 2026 ---
    df26 = pd.read_csv('KQ_2026.csv')

    # Defensive re-check: KQ_2026.csv is expected to already be consent-filtered
    # by clean_2026.py, but re-verify here in case a future raw export is loaded
    # directly without going through that script.
    consent_col_26 = [c for c in df26.columns if "consenting to us" in c.lower()]
    if consent_col_26:
        df26 = df26[df26[consent_col_26[0]].astype(str).str.contains("Yes", na=False)]

    # IMPORTANT: 2025 and 2026 reuse the SAME question text "How would you rate the
    # current state of your mental well-being?" for two DIFFERENT questions:
    #   2025: categorical state (I am OK and coping well / I'm just there / I am not
    #         OK / I need support)
    #   2026: a Yes/No "Do you feel the need to be supported?" question
    # These are not comparable — split 2026's into its own column so it never gets
    # merged into the 2025/2022 "current state" chart.
    df26 = df26.rename(columns={
        'How would you rate the current state of your mental well-being?': 'Need Support in Current State (2026)'
    })

    # Normalise demographics NaN across all three
    for col in ['Please select your gender', 'Kindly select your age bracket',
                'Please select your department from the list below',
                'How would you rate the current state of your mental well-being?']:
        for df in [df22, df25, df26]:
            if col in df.columns:
                df[col] = df[col].replace({'No Response': np.nan}).fillna('Not Specified')

    # Normalise typographic punctuation that varies between survey export years
    # (e.g. one year's export uses a straight apostrophe "I'm just there", another
    # uses a curly one "I’m just there" for the SAME answer option). Without this,
    # pandas/Plotly treat them as two different categories, silently splitting a
    # single answer into two bars on any shared multi-year chart — this is what
    # caused "I'm just there" to appear twice (once empty) on the Current State
    # of Well-Being chart. Applied to every text column in every year so it
    # can't recur elsewhere as new questions get added.
    QUOTE_MAP = {
        '\u2019': "'",  # curly apostrophe -> straight
        '\u2018': "'",  # curly opening single quote -> straight
        '\u201c': '"',  # curly opening double quote -> straight
        '\u201d': '"',  # curly closing double quote -> straight
    }
    def normalize_quotes(v):
        if isinstance(v, str):
            return ''.join(QUOTE_MAP.get(ch, ch) for ch in v)
        return v

    for df in [df22, df25, df26]:
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            df[col] = df[col].apply(normalize_quotes)

    # Normalise remaining wording variants between years that describe the SAME
    # answer option but differ in more than just punctuation (e.g. "Few days
    # ago" vs "A few days ago"). NOTE: only safe when the options genuinely
    # mean the same thing — see freq_order handling below for a case where the
    # scales themselves differ and are intentionally kept separate instead of
    # merged.
    TIMING_ALIASES = {
        'Few days ago': 'A few days ago',
        'Few weeks ago': 'A few weeks ago',
        'Few months ago': 'A few months ago',
        'Few years ago': 'A few years ago',
    }
    WORDING_ALIASES = {
        'When was the last time you were really happy?': TIMING_ALIASES,
        'When was the last time you felt good about yourself?': TIMING_ALIASES,
        'When was the last time you had a positive outlook on life?': TIMING_ALIASES,
        'How often do you feel positive about your life?': {
            'About half the time': 'About half of the time',
        },
    }

    for col, mapping in WORDING_ALIASES.items():
        for df in [df22, df25, df26]:
            if col in df.columns:
                df[col] = df[col].replace(mapping)

    return df22, df25, df26

df22_raw, df25_raw, df26_raw = load_data()

# =========================
# SIDEBAR
# =========================
try:
    st.sidebar.image("logo.png", use_container_width=True)
except Exception:
    pass
st.sidebar.markdown("## Centre for Innovation and Analytics (CIA)")

view_mode = st.sidebar.radio(
    "📅 View Mode",
    ["All Years (2022 / 2025 / 2026)", "Comparison (2022 vs 2025)", "2022 Only", "2025 Only", "2026 Only"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔽 Filters")

def safe_categories(*series_list):
    """Build a clean, de-duplicated, NaN-free set of categories from one or more
    series, normalising blanks/whitespace so a quirk in any single year's raw
    data (e.g. a stray blank string) can't silently exclude an otherwise valid
    category from the default filter selection."""
    vals = set()
    for s in series_list:
        cleaned = s.dropna().astype(str).str.strip()
        cleaned = cleaned[cleaned != '']
        vals |= set(cleaned.unique())
    vals.discard('nan')
    vals.add('Not Specified')  # always available regardless of whether any year happens to have it
    return vals

all_genders = sorted(safe_categories(
    df22_raw['Please select your gender'],
    df25_raw['Please select your gender'],
    df26_raw['Please select your gender'],
))
sel_gender = st.sidebar.multiselect("Gender:", all_genders, default=all_genders)

AGE_SORT_ORDER = ['18-24 years', '25-39 years', '40-49 years', '50-59 years', '50+ years', '60+ years', 'Not Specified']
def age_sort_key(label):
    return AGE_SORT_ORDER.index(label) if label in AGE_SORT_ORDER else len(AGE_SORT_ORDER)

all_ages = sorted(safe_categories(
    df22_raw['Kindly select your age bracket'],
    df25_raw['Kindly select your age bracket'],
    df26_raw['Kindly select your age bracket'],
), key=age_sort_key)
sel_age = st.sidebar.multiselect("Age Bracket:", all_ages, default=all_ages)

dept_col = 'Please select your department from the list below'
all_depts = sorted(safe_categories(
    df22_raw[dept_col],
    df25_raw[dept_col],
    df26_raw[dept_col],
))
sel_dept = st.sidebar.multiselect("Department:", all_depts, default=all_depts)

def apply_filters(df):
    # Normalise the same way as safe_categories so filter membership checks
    # can't be defeated by stray whitespace differences between years.
    gender_norm = df['Please select your gender'].astype(str).str.strip()
    age_norm = df['Kindly select your age bracket'].astype(str).str.strip()
    dept_norm = df[dept_col].astype(str).str.strip()
    mask = (
        gender_norm.isin(sel_gender) &
        age_norm.isin(sel_age) &
        dept_norm.isin(sel_dept)
    )
    return df[mask]

df22 = apply_filters(df22_raw)
df25 = apply_filters(df25_raw)
df26 = apply_filters(df26_raw)

show_22 = view_mode in ["All Years (2022 / 2025 / 2026)", "Comparison (2022 vs 2025)", "2022 Only"]
show_25 = view_mode in ["All Years (2022 / 2025 / 2026)", "Comparison (2022 vs 2025)", "2025 Only"]
show_26 = view_mode in ["All Years (2022 / 2025 / 2026)", "2026 Only"]
is_comparison = view_mode == "Comparison (2022 vs 2025)"
is_all = view_mode == "All Years (2022 / 2025 / 2026)"

# =========================
# TITLE
# =========================
try:
    with open("icon.png", "rb") as f:
        _icon_b64 = base64.b64encode(f.read()).decode()
    _icon_html = f'<img src="data:image/png;base64,{_icon_b64}" style="width:64px;height:64px;border-radius:8px;flex-shrink:0;">'
except Exception:
    _icon_html = ""

st.markdown(
    f'''<div style="display:flex;align-items:center;gap:18px;margin-bottom:4px;">
        {_icon_html}
        <h1 style="margin:0;line-height:1.1;">Kenya Airways Wellness Dashboard</h1>
    </div>''',
    unsafe_allow_html=True
)
badges = []
if show_22: badges.append('<span class="year-badge-2022">2022</span>')
if show_25: badges.append('<span class="year-badge-2025">2025</span>')
if show_26: badges.append('<span class="year-badge-2026">2026</span>')
st.markdown(" &nbsp;·&nbsp; ".join(badges), unsafe_allow_html=True)

# =========================
# CHART HELPERS
# =========================
def multi_year_bar(title, series_dict, category_orders=None):
    """series_dict: {'2022': series, '2025': series, '2026': series} — only include active years."""
    frames = []
    for year, s in series_dict.items():
        c = clean(s).value_counts().reset_index()
        c.columns = ['Category', 'Count']
        c['Year'] = year
        frames.append(c)
    combined = pd.concat(frames)
    fig = px.bar(combined, x='Category', y='Count', color='Year', barmode='group',
                 title=title, text_auto=True,
                 color_discrete_map=YEAR_COLORS,
                 category_orders=({'Category': category_orders} if category_orders else {}))
    fig.update_layout(legend_title_text='Year', height=380)
    return fig

def single_bar(title, series, year_label, color, category_orders=None):
    counts = clean(series).value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    fig = px.bar(counts, x='Category', y='Count', title=f"{title} ({year_label})",
                 text_auto=True, color_discrete_sequence=[color],
                 category_orders=({'x': category_orders} if category_orders else {}))
    fig.update_layout(height=380)
    return fig

def active_series():
    """Return dict of {year: df} for currently active view."""
    d = {}
    if show_22: d['2022'] = df22
    if show_25: d['2025'] = df25
    if show_26: d['2026'] = df26
    return d

# =========================
# CONSTANTS
# =========================
POS_STATES = ['Excellent', 'Very Good', 'Good']
MW_COL = 'How would you rate the state of your mental well-being?'
EAP_COL = 'Are you aware of the Employee Assistance Program services offered by Kenya Airways to all its staff and dependents through Minet?'
AWARENESS_COL = 'Have you ever attended a Mental Health Awareness Session?'
SLEEP_HR_COL = 'How many hours do you sleep per day?'
SLEEP_Q_COL = 'How is your quality of sleep?'
CURR_STATE_COL = 'How would you rate the current state of your mental well-being?'
COUNSEL_COL = 'Would you like us to link you or your dependents to our professional counselors for support?'
COPE_COL = 'How do you usually cope with the general stresses of life and the mental health challenges that come your way?'
SMOKE_COL = 'How often do you smoke?'
DRINK_COL = 'How often do you drink?'
NICOTINE_COL = 'Do you use any nicotine-containing products (this includes cigarettes, vapes/ e-cigarettes, shisha, nicotine pouches, chewing tobacco, or snuff)?'
ALCOHOL_COL = 'Do you consume any alcoholic beverages (e.g. beer, wines, spirits, liquor, traditional brews, cocktails)?'
NICOTINE_QUIT_COL = 'If you use nicotine products, would you be interested in support to reduce or quit?'
ALCOHOL_QUIT_COL = 'If you use alcoholic products, would you be interested in support to reduce or quit?'
MANAGER_SUPPORT_COL = 'At work, I feel supported by my team and/or manager.'
SOCIAL_SUPPORT_COL = 'To what extent do you feel you have people you can rely on for emotional or practical support when you need it? (This may include family, friends, colleagues, or community members, e.g., church, mosque, club, etc)'

issue_labels = {
    'MW_Issue_Work_Job_Stressors': 'Work / Job Stressors',
    'MW_Issue_Emotional_Intelligence': 'Emotional Intelligence',
    'MW_Issue_Financial_Constraints': 'Financial Constraints',
    'MW_Issue_Trauma': 'Trauma',
    'MW_Issue_Substance_Drug_Dependence': 'Substance & Drug Dependence',
    'MW_Issue_Relationship_Difficulties': 'Relationship Difficulties',
    'MW_Issue_Unresolved_Childhood_Difficulties': 'Unresolved Childhood Difficulties',
    'MW_Issue_Anxiety_Depression': 'Anxiety & Depression',
    'MW_Issue_Neurodivergence': 'Neurodivergence',
}
NO_RESPONSE_VALS = {'no response', 'na', 'n/a', 'none', 'nil', '', 'nan', 'not applicable'}

def issue_counts(df, year):
    rows = []
    for col, label in issue_labels.items():
        if col in df.columns:
            valid = df[col][~df[col].astype(str).str.lower().str.strip().isin(NO_RESPONSE_VALS)].dropna()
            rows.append({'Topic': label, 'Count': len(valid), 'Year': year})
    return pd.DataFrame(rows)

# =========================
# KPI ROW
# =========================
st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

active = active_series()

def kpi_multi(label, icon, vals_dict, unit="%", fmt=".1f"):
    """vals_dict: {year: value}. Stacks years vertically to avoid horizontal
    crowding/overlap when 3 years are active at once."""
    n_years = len(vals_dict)
    value_font = "1.5rem" if n_years <= 1 else ("1.25rem" if n_years == 2 else "1.05rem")
    rows = ""
    for year, val in vals_dict.items():
        color = YEAR_COLORS[year]
        rows += (
            f'<div style="display:flex;align-items:baseline;justify-content:center;gap:6px;line-height:1.2;">'
            f'<span class="kpi-value" style="font-size:{value_font};color:{color};margin:0;">{val:{fmt}}{unit}</span>'
            f'<span class="kpi-year" style="margin-top:0;">{year}</span>'
            f'</div>'
        )
    return f'''<div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div style="display:flex;flex-direction:column;gap:4px;margin:8px 0 2px 0;">{rows}</div>
    </div>'''

n_vals  = {y: len(d) for y, d in active.items()}
mw_vals = {y: pct(d[MW_COL], POS_STATES) for y, d in active.items() if MW_COL in d.columns}
eap_vals = {y: pct(d[EAP_COL], ['Yes']) for y, d in active.items() if EAP_COL in d.columns}
aw_vals  = {y: pct(d[AWARENESS_COL], ['Yes']) for y, d in active.items() if AWARENESS_COL in d.columns}
slp_vals = {y: d[SLEEP_HR_COL].apply(parse_sleep).mean() for y, d in active.items() if SLEEP_HR_COL in d.columns}

kpi_cols = st.columns(5)
for col_el, (label, icon, vals, unit, fmt) in zip(kpi_cols, [
    ("Respondents",       "👤", n_vals,   "",  "g"),
    ("Positive Wellbeing", "😊", mw_vals,  "%", ".1f"),
    ("EAP Awareness",     "🔔", eap_vals, "%", ".1f"),
    ("Attended Sessions", "🎓", aw_vals,  "%", ".1f"),
    ("Avg Sleep (hrs)",   "🌙", slp_vals, "h", ".1f"),
]):
    col_el.markdown(kpi_multi(label, icon, vals, unit, fmt), unsafe_allow_html=True)

# =========================
# DEMOGRAPHICS
# =========================
st.markdown('<div class="section-header">👥 Workforce Demographics</div>', unsafe_allow_html=True)
dem_c1, dem_c2, dem_c3 = st.columns(3)

with dem_c1:
    if is_all or is_comparison or (show_22 and show_25):
        frames = []
        for year, df in active.items():
            c = clean(df['Please select your gender']).value_counts().reset_index()
            c.columns = ['Cat', 'Count']; c['Year'] = year
            frames.append(c)
        fig = px.pie(pd.concat(frames), names='Cat', values='Count', facet_col='Year',
                     title="Gender Distribution", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.T10)
        fig.update_layout(height=370)
        st.plotly_chart(fig, use_container_width=True)
    else:
        year, df = list(active.items())[0]
        c = clean(df['Please select your gender']).value_counts().reset_index(); c.columns = ['Cat','Count']
        fig = px.pie(c, names='Cat', values='Count', hole=0.45, title=f"Gender Distribution ({year})",
                     color_discrete_sequence=px.colors.qualitative.T10)
        fig.update_layout(height=370)
        st.plotly_chart(fig, use_container_width=True)

with dem_c2:
    age_order = all_ages
    if len(active) > 1:
        st.plotly_chart(multi_year_bar("Age Bracket", {y: d['Kindly select your age bracket'] for y,d in active.items()}, age_order), use_container_width=True)
    else:
        year, df = list(active.items())[0]
        st.plotly_chart(single_bar("Age Bracket", df['Kindly select your age bracket'], year, YEAR_COLORS[year], age_order), use_container_width=True)

with dem_c3:
    if len(active) > 1:
        st.plotly_chart(multi_year_bar("Department Distribution", {y: d[dept_col] for y,d in active.items()}), use_container_width=True)
    else:
        year, df = list(active.items())[0]
        st.plotly_chart(single_bar("Department Distribution", df[dept_col], year, YEAR_COLORS[year]), use_container_width=True)

# =========================
# MENTAL WELL-BEING
# =========================
st.markdown('<div class="section-header">🧠 Mental Well-Being</div>', unsafe_allow_html=True)
mw_c1, mw_c2 = st.columns(2)

with mw_c1:
    if len(active) > 1:
        st.plotly_chart(
            multi_year_bar("Overall Mental Well-Being Rating",
                           {y: d[MW_COL] for y,d in active.items() if MW_COL in d.columns},
                           ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor', 'Very poor']),
            use_container_width=True)
    else:
        year, df = list(active.items())[0]
        st.plotly_chart(single_bar("Overall Mental Well-Being Rating", df[MW_COL], year, YEAR_COLORS[year]), use_container_width=True)

with mw_c2:
    curr_order = ['I am OK and coping well', "I'm just there", 'I am not OK', 'I need support', 'Not Specified']
    active_curr = {y: d for y,d in active.items() if CURR_STATE_COL in d.columns and d[CURR_STATE_COL].notna().any()}
    if len(active_curr) > 1:
        st.plotly_chart(
            multi_year_bar("Current State of Well-Being",
                           {y: d[CURR_STATE_COL] for y,d in active_curr.items()},
                           curr_order),
            use_container_width=True)
    elif active_curr:
        year, df = list(active_curr.items())[0]
        st.plotly_chart(single_bar("Current State of Well-Being", df[CURR_STATE_COL], year, YEAR_COLORS[year], curr_order), use_container_width=True)
    else:
        st.info("Current state data not available for selected year(s).")

# 2026-only: "Do you feel the need to be supported?" (NOT comparable to the
# 2022/2025 categorical "current state" question above — different question.)
NEED_SUPPORT_2026_COL = 'Need Support in Current State (2026)'
if show_26 and NEED_SUPPORT_2026_COL in df26.columns:
    st.markdown(
        '<span class="new-badge">2026 NEW</span> &nbsp;"Do you feel the need to be supported in your current mental well-being state?" '
        '— this replaced the multi-option "current state" question used in 2022/2025, so it is shown separately rather than merged into the chart above.',
        unsafe_allow_html=True
    )
    c = df26[NEED_SUPPORT_2026_COL].value_counts(dropna=False).reset_index()
    c.columns = ['Cat', 'Count']
    fig_need = px.bar(c, x='Cat', y='Count', text_auto=True,
                      title="Feels the Need to Be Supported Right Now (2026)",
                      color='Cat', color_discrete_map={'Yes': COL_2026, 'No': '#a8d5ba', 'Not Specified': '#ccc'})
    fig_need.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig_need, use_container_width=True)

# MW Issues
st.markdown("**Requested Wellness Focus Areas**")
all_issues = pd.concat([issue_counts(d, y) for y, d in active.items()])
if not all_issues.empty:
    if len(active) > 1:
        fig_iss = px.bar(all_issues, x='Count', y='Topic', color='Year', barmode='group',
                         orientation='h', text_auto=True,
                         color_discrete_map=YEAR_COLORS,
                         title="Requested Wellness Focus Areas")
    else:
        year = list(active.keys())[0]
        fig_iss = px.bar(all_issues, x='Count', y='Topic', orientation='h', text_auto=True,
                         color='Count', color_continuous_scale='Blues',
                         title=f"Requested Wellness Focus Areas ({year})")
    fig_iss.update_layout(height=420)
    st.plotly_chart(fig_iss, use_container_width=True)

# =========================
# EMOTIONAL WELLBEING — IMPACT & POSITIVITY (previously missing from this
# dashboard; present in all years but never charted)
# =========================
st.markdown('<div class="section-header">💭 Emotional Wellbeing — Impact & Positivity</div>', unsafe_allow_html=True)

EMOTIONAL_PROBLEM_COL = 'During the past 4 weeks, have you had any problems with your work or daily life due to any emotional problems, such as feeling depressed, sad, or anxious?'
WORK_IMPACT_COL = 'During the past 4 weeks, how often has your mental health affected your ability to get work done?'
RELATIONSHIP_IMPACT_COL = 'During the past 2 weeks, how often has your mental health affected your relationships?'
CONTENT_RELATIONSHIPS_COL = 'Do you feel content with your relationships and family?'

# NOTE: 2025 and 2026 use genuinely different frequency scales for these two
# questions (2025: Not at all / Not so often / Somewhat often / Very often —
# a 4-point scale; 2026: Not at all / Rarely / Sometimes / Often / Very often —
# a 5-point scale). They aren't a 1:1 wording mismatch like the timing
# questions above, so rather than force them onto an identical axis (which
# would misrepresent which 2025 option corresponds to which 2026 option), the
# combined order below simply places every option that appears in either year
# in a sensible low-to-high sequence; Plotly groups bars by year regardless.
freq_order = ['Not at all', 'Not so often', 'Rarely', 'Sometimes', 'Somewhat often', 'Often', 'Very often']

em_c1, em_c2 = st.columns(2)
with em_c1:
    active_ep = {y: d for y, d in active.items() if EMOTIONAL_PROBLEM_COL in d.columns and d[EMOTIONAL_PROBLEM_COL].notna().any()}
    if len(active_ep) > 1:
        st.plotly_chart(multi_year_bar("Had Emotional Problems Affecting Work/Life (Past 4 Weeks)",
                                       {y: d[EMOTIONAL_PROBLEM_COL] for y, d in active_ep.items()}),
                        use_container_width=True)
    elif active_ep:
        year, df = list(active_ep.items())[0]
        st.plotly_chart(single_bar("Had Emotional Problems Affecting Work/Life (Past 4 Weeks)", df[EMOTIONAL_PROBLEM_COL], year, YEAR_COLORS[year]), use_container_width=True)

with em_c2:
    active_wi = {y: d for y, d in active.items() if WORK_IMPACT_COL in d.columns and d[WORK_IMPACT_COL].notna().any()}
    if len(active_wi) > 1:
        st.plotly_chart(multi_year_bar("Mental Health Impact on Work Output (Past 4 Weeks)",
                                       {y: d[WORK_IMPACT_COL] for y, d in active_wi.items()},
                                       freq_order),
                        use_container_width=True)
    elif active_wi:
        year, df = list(active_wi.items())[0]
        st.plotly_chart(single_bar("Mental Health Impact on Work Output (Past 4 Weeks)", df[WORK_IMPACT_COL], year, YEAR_COLORS[year], freq_order), use_container_width=True)

em_c3, em_c4 = st.columns(2)
with em_c3:
    active_ri = {y: d for y, d in active.items() if RELATIONSHIP_IMPACT_COL in d.columns and d[RELATIONSHIP_IMPACT_COL].notna().any()}
    if len(active_ri) > 1:
        st.plotly_chart(multi_year_bar("Mental Health Impact on Relationships (Past 2 Weeks)",
                                       {y: d[RELATIONSHIP_IMPACT_COL] for y, d in active_ri.items()},
                                       freq_order),
                        use_container_width=True)
    elif active_ri:
        year, df = list(active_ri.items())[0]
        st.plotly_chart(single_bar("Mental Health Impact on Relationships (Past 2 Weeks)", df[RELATIONSHIP_IMPACT_COL], year, YEAR_COLORS[year], freq_order), use_container_width=True)

with em_c4:
    active_cr = {y: d for y, d in active.items() if CONTENT_RELATIONSHIPS_COL in d.columns and d[CONTENT_RELATIONSHIPS_COL].notna().any()}
    if len(active_cr) > 1:
        st.plotly_chart(multi_year_bar("Content with Relationships & Family",
                                       {y: d[CONTENT_RELATIONSHIPS_COL] for y, d in active_cr.items()}),
                        use_container_width=True)
    elif active_cr:
        year, df = list(active_cr.items())[0]
        st.plotly_chart(single_bar("Content with Relationships & Family", df[CONTENT_RELATIONSHIPS_COL], year, YEAR_COLORS[year]), use_container_width=True)

# Positivity / happiness timing questions
st.markdown("**Positivity & Happiness Markers**")
HAPPY_COL = 'When was the last time you were really happy?'
GOOD_SELF_COL = 'When was the last time you felt good about yourself?'
POSITIVE_OFTEN_COL = 'How often do you feel positive about your life?'
POSITIVE_OUTLOOK_COL = 'When was the last time you had a positive outlook on life?'

timing_order = ['Today', 'A few days ago', 'A few weeks ago', 'A few months ago', 'A few years ago', "I don't remember"]
often_order = ['Always', 'Most of the time', 'About half of the time', 'Once in a while', 'Never']

pos_c1, pos_c2 = st.columns(2)
with pos_c1:
    active_h = {y: d for y, d in active.items() if HAPPY_COL in d.columns and d[HAPPY_COL].notna().any()}
    if len(active_h) > 1:
        st.plotly_chart(multi_year_bar("Last Time Felt Really Happy",
                                       {y: d[HAPPY_COL] for y, d in active_h.items()}, timing_order),
                        use_container_width=True)
    elif active_h:
        year, df = list(active_h.items())[0]
        st.plotly_chart(single_bar("Last Time Felt Really Happy", df[HAPPY_COL], year, YEAR_COLORS[year], timing_order), use_container_width=True)

with pos_c2:
    active_g = {y: d for y, d in active.items() if GOOD_SELF_COL in d.columns and d[GOOD_SELF_COL].notna().any()}
    if len(active_g) > 1:
        st.plotly_chart(multi_year_bar("Last Time Felt Good About Self",
                                       {y: d[GOOD_SELF_COL] for y, d in active_g.items()}, timing_order),
                        use_container_width=True)
    elif active_g:
        year, df = list(active_g.items())[0]
        st.plotly_chart(single_bar("Last Time Felt Good About Self", df[GOOD_SELF_COL], year, YEAR_COLORS[year], timing_order), use_container_width=True)

pos_c3, pos_c4 = st.columns(2)
with pos_c3:
    active_po = {y: d for y, d in active.items() if POSITIVE_OFTEN_COL in d.columns and d[POSITIVE_OFTEN_COL].notna().any()}
    if len(active_po) > 1:
        st.plotly_chart(multi_year_bar("How Often Feel Positive About Life",
                                       {y: d[POSITIVE_OFTEN_COL] for y, d in active_po.items()}, often_order),
                        use_container_width=True)
    elif active_po:
        year, df = list(active_po.items())[0]
        st.plotly_chart(single_bar("How Often Feel Positive About Life", df[POSITIVE_OFTEN_COL], year, YEAR_COLORS[year], often_order), use_container_width=True)

with pos_c4:
    active_out = {y: d for y, d in active.items() if POSITIVE_OUTLOOK_COL in d.columns and d[POSITIVE_OUTLOOK_COL].notna().any()}
    if len(active_out) > 1:
        st.plotly_chart(multi_year_bar("Last Time Had a Positive Outlook on Life",
                                       {y: d[POSITIVE_OUTLOOK_COL] for y, d in active_out.items()}, timing_order),
                        use_container_width=True)
    elif active_out:
        year, df = list(active_out.items())[0]
        st.plotly_chart(single_bar("Last Time Had a Positive Outlook on Life", df[POSITIVE_OUTLOOK_COL], year, YEAR_COLORS[year], timing_order), use_container_width=True)

# =========================
# CLINICAL & EMOTIONAL HISTORY
# =========================
st.markdown('<div class="section-header">🩺 Clinical & Emotional History</div>', unsafe_allow_html=True)

CLINICAL_COLS = [
    ('Have you ever been diagnosed with a mental disorder before?', 'Personal Diagnosis'),
    ('Is there a history of mental disorder in your family?', 'Family History'),
    ('Have you seen a therapist in the recent past?', 'Seen a Therapist'),
    ('Are you going through a tough emotional situation right now?', 'Current Tough Situation'),
]

def clinical_data(df, year):
    rows = []
    for col, label in CLINICAL_COLS:
        if col in df.columns:
            rows.append({'Metric': label, 'Count': int((df[col] == 'Yes').sum()), 'Year': year})
    return pd.DataFrame(rows)

clin_all = pd.concat([clinical_data(d, y) for y, d in active.items()])
if not clin_all.empty:
    if len(active) > 1:
        fig_clin = px.bar(clin_all, x='Metric', y='Count', color='Year', barmode='group',
                          text_auto=True, color_discrete_map=YEAR_COLORS,
                          title="Clinical & Emotional Indicators")
    else:
        year = list(active.keys())[0]
        fig_clin = px.bar(clin_all, x='Metric', y='Count', text_auto=True,
                          title=f"Clinical & Emotional Indicators ({year})",
                          color='Metric', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_clin.update_layout(height=380)
    st.plotly_chart(fig_clin, use_container_width=True)

# =========================
# SLEEP
# =========================
st.markdown('<div class="section-header">🌙 Sleep</div>', unsafe_allow_html=True)
sl_c1, sl_c2 = st.columns(2)

sleep_hr_order = ['Less than 4', '4 - 6', '7 - 9', '9+']
sleep_q_order  = ['Very Good', 'Good', 'OK', 'Bad', 'Very Bad', 'Not Sure']

with sl_c1:
    if len(active) > 1:
        st.plotly_chart(multi_year_bar("Sleep Hours per Day",
                                       {y: d[SLEEP_HR_COL] for y,d in active.items() if SLEEP_HR_COL in d.columns},
                                       sleep_hr_order), use_container_width=True)
    else:
        year, df = list(active.items())[0]
        st.plotly_chart(single_bar("Sleep Hours per Day", df[SLEEP_HR_COL], year, YEAR_COLORS[year], sleep_hr_order), use_container_width=True)

with sl_c2:
    if len(active) > 1:
        st.plotly_chart(multi_year_bar("Sleep Quality",
                                       {y: d[SLEEP_Q_COL] for y,d in active.items() if SLEEP_Q_COL in d.columns},
                                       sleep_q_order), use_container_width=True)
    else:
        year, df = list(active.items())[0]
        st.plotly_chart(single_bar("Sleep Quality", df[SLEEP_Q_COL], year, YEAR_COLORS[year], sleep_q_order), use_container_width=True)

# =========================
# LIFESTYLE — SMOKING / DRINKING (2022 & 2025) + NICOTINE/ALCOHOL (2026)
# =========================
st.markdown('<div class="section-header">🚬🍺 Lifestyle Habits</div>', unsafe_allow_html=True)

# --- 2022 / 2025: frequency charts ---
legacy_years = {y: d for y, d in active.items() if y in ('2022', '2025')}
if legacy_years:
    lf_c1, lf_c2 = st.columns(2)
    with lf_c1:
        if len(legacy_years) > 1:
            st.plotly_chart(multi_year_bar("Smoking Frequency",
                                           {y: d[SMOKE_COL] for y,d in legacy_years.items() if SMOKE_COL in d.columns}),
                            use_container_width=True)
        else:
            year, df = list(legacy_years.items())[0]
            if SMOKE_COL in df.columns:
                st.plotly_chart(single_bar("Smoking Frequency", df[SMOKE_COL], year, YEAR_COLORS[year]), use_container_width=True)

    with lf_c2:
        if len(legacy_years) > 1:
            st.plotly_chart(multi_year_bar("Drinking Frequency",
                                           {y: d[DRINK_COL] for y,d in legacy_years.items() if DRINK_COL in d.columns}),
                            use_container_width=True)
        else:
            year, df = list(legacy_years.items())[0]
            if DRINK_COL in df.columns:
                st.plotly_chart(single_bar("Drinking Frequency", df[DRINK_COL], year, YEAR_COLORS[year]), use_container_width=True)

# --- 2026: nicotine & alcohol yes/no + quit support ---
if show_26:
    st.markdown(
        '<span class="new-badge">2026 NEW</span> Nicotine & Alcohol — restructured question format',
        unsafe_allow_html=True
    )
    nc1, nc2, nc3, nc4 = st.columns(4)
    for col_el, (col, title) in zip([nc1, nc2, nc3, nc4], [
        (NICOTINE_COL,     "Nicotine Use (2026)"),
        (NICOTINE_QUIT_COL,"Interest in Nicotine Quit Support"),
        (ALCOHOL_COL,      "Alcohol Use (2026)"),
        (ALCOHOL_QUIT_COL, "Interest in Alcohol Quit Support"),
    ]):
        if col in df26.columns:
            c = clean(df26[col]).value_counts().reset_index(); c.columns = ['Cat','Count']
            fig = px.pie(c, names='Cat', values='Count', hole=0.45, title=title,
                         color_discrete_sequence=[COL_2026, '#a8d5ba', '#cce8d8'])
            fig.update_layout(height=330)
            col_el.plotly_chart(fig, use_container_width=True)

# =========================
# SUPPORT NETWORKS — Manager support (2026-only, genuinely new) + Social
# support (now comparable across 2025/2026 since wording is aligned)
# =========================
has_manager = show_26 and MANAGER_SUPPORT_COL in df26.columns
active_social = {y: d for y, d in active.items() if SOCIAL_SUPPORT_COL in d.columns and d[SOCIAL_SUPPORT_COL].notna().any()}

if has_manager or active_social:
    st.markdown('<div class="section-header">🤝 Support Networks</div>', unsafe_allow_html=True)
    sp_c1, sp_c2 = st.columns(2)

    with sp_c1:
        if has_manager:
            st.markdown('<span class="new-badge">2026 NEW</span>', unsafe_allow_html=True)
            c = clean(df26[MANAGER_SUPPORT_COL]).value_counts().reset_index(); c.columns=['Cat','Count']
            fig = px.bar(c, x='Cat', y='Count', text_auto=True,
                         title="Manager / Team Support (2026)",
                         color_discrete_sequence=[COL_2026])
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

    with sp_c2:
        if len(active_social) > 1:
            st.plotly_chart(multi_year_bar("Social Support Network Extent",
                                           {y: d[SOCIAL_SUPPORT_COL] for y, d in active_social.items()}),
                            use_container_width=True)
        elif active_social:
            year, df = list(active_social.items())[0]
            st.plotly_chart(single_bar("Social Support Network Extent", df[SOCIAL_SUPPORT_COL], year, YEAR_COLORS[year]), use_container_width=True)

# Preferred channel for mental health info
INFO_PREF_COL = 'How would you prefer to receive information and advice about mental health?'
active_info = {y: d for y, d in active.items() if INFO_PREF_COL in d.columns and d[INFO_PREF_COL].notna().any()}
if active_info:
    st.markdown("**Preferred Mental Health Information Channel**")
    if len(active_info) > 1:
        st.plotly_chart(multi_year_bar("Preferred Information Channel", {y: d[INFO_PREF_COL] for y, d in active_info.items()}), use_container_width=True)
    else:
        year, df = list(active_info.items())[0]
        st.plotly_chart(single_bar("Preferred Information Channel", df[INFO_PREF_COL], year, YEAR_COLORS[year]), use_container_width=True)

# =========================
# AWARENESS & EAP
# =========================
st.markdown('<div class="section-header">📚 Awareness & EAP Engagement</div>', unsafe_allow_html=True)
aw_c1, aw_c2 = st.columns(2)

with aw_c1:
    if len(active) > 1:
        st.plotly_chart(multi_year_bar("Mental Health Awareness Sessions",
                                       {y: d[AWARENESS_COL] for y,d in active.items() if AWARENESS_COL in d.columns}),
                        use_container_width=True)
    else:
        year, df = list(active.items())[0]
        st.plotly_chart(single_bar("Mental Health Awareness Sessions", df[AWARENESS_COL], year, YEAR_COLORS[year]), use_container_width=True)

with aw_c2:
    if len(active) > 1:
        st.plotly_chart(multi_year_bar("EAP Services Awareness",
                                       {y: d[EAP_COL] for y,d in active.items() if EAP_COL in d.columns}),
                        use_container_width=True)
    else:
        year, df = list(active.items())[0]
        st.plotly_chart(single_bar("EAP Services Awareness", df[EAP_COL], year, YEAR_COLORS[year]), use_container_width=True)

if len(active) > 1:
    counsel_active = {y: d for y,d in active.items() if COUNSEL_COL in d.columns}
    if counsel_active:
        st.plotly_chart(multi_year_bar("Requested Counsellor Link-Up",
                                       {y: d[COUNSEL_COL] for y,d in counsel_active.items()}),
                        use_container_width=True)
else:
    year, df = list(active.items())[0]
    if COUNSEL_COL in df.columns:
        st.plotly_chart(single_bar("Requested Counsellor Link-Up", df[COUNSEL_COL], year, YEAR_COLORS[year]), use_container_width=True)

# =========================
# YoY TREND
# =========================
if len(active) > 1:
    st.markdown('<div class="section-header">📈 Year-on-Year Trend Summary</div>', unsafe_allow_html=True)

    trend_rows = []
    for year, df in active.items():
        trend_rows.append({
            'Metric': 'Positive Wellbeing %',
            'Year': year,
            'Value': pct(df[MW_COL], POS_STATES) if MW_COL in df.columns else np.nan,
        })
        trend_rows.append({
            'Metric': 'EAP Awareness %',
            'Year': year,
            'Value': pct(df[EAP_COL], ['Yes']) if EAP_COL in df.columns else np.nan,
        })
        trend_rows.append({
            'Metric': 'Attended Sessions %',
            'Year': year,
            'Value': pct(df[AWARENESS_COL], ['Yes']) if AWARENESS_COL in df.columns else np.nan,
        })
        trend_rows.append({
            'Metric': 'Avg Sleep hrs',
            'Year': year,
            'Value': df[SLEEP_HR_COL].apply(parse_sleep).mean() if SLEEP_HR_COL in df.columns else np.nan,
        })
        trend_rows.append({
            'Metric': 'Requested Counsellor %',
            'Year': year,
            'Value': pct(df[COUNSEL_COL], ['Yes']) if COUNSEL_COL in df.columns else np.nan,
        })

    trend_df = pd.DataFrame(trend_rows).dropna(subset=['Value'])
    fig_trend = px.line(trend_df, x='Metric', y='Value', color='Year', markers=True,
                        text='Value',
                        color_discrete_map=YEAR_COLORS,
                        title="Key Metric Trends Over Time")
    fig_trend.update_traces(texttemplate='%{y:.1f}')
    # Stagger label positions per year so close values don't overlap
    label_positions = {'2022': 'top center', '2025': 'bottom center', '2026': 'top right'}
    for trace in fig_trend.data:
        trace.textposition = label_positions.get(trace.name, 'top center')
    fig_trend.update_layout(height=460, xaxis_tickangle=-20)
    st.plotly_chart(fig_trend, use_container_width=True)

# =========================
# COPING MECHANISMS
# =========================
st.markdown('<div class="section-header">🛠️ Coping Mechanisms</div>', unsafe_allow_html=True)
cope_active = {y: d for y,d in active.items() if COPE_COL in d.columns and d[COPE_COL].notna().any()}
if len(cope_active) > 1:
    st.plotly_chart(multi_year_bar("Coping Mechanisms", {y: d[COPE_COL] for y,d in cope_active.items()}), use_container_width=True)
elif cope_active:
    year, df = list(cope_active.items())[0]
    c = clean(df[COPE_COL]).value_counts().reset_index(); c.columns=['Cat','Count']
    fig = px.bar(c, x='Cat', y='Count', text_auto=True, title=f"Coping Mechanisms ({year})",
                 color='Count', color_continuous_scale='Viridis')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# OPEN FEEDBACK EXPLORER
# =========================
st.markdown('<div class="section-header">🗣️ Open-Ended Feedback Explorer</div>', unsafe_allow_html=True)

SUPPORT_COL_25 = 'If you need support in Q5 above, please state what support you would need?'
SUPPORT_COL_26 = 'If you need support in Q6 above, please state what support you would need?'
EAP_NO_COL = 'If no, please state the reason.'

JUNK_SET = {'no support requested','nil','ok','no suport','no support needed','currently no',
            'good','n/a','none is required','not applicable','n / a','am good','not sure',
            'i am good','at the moment nil','non','no','not','i am ok','-','no need','n?a','na.','i am okay','none'}

fb_c1, fb_c2 = st.columns([1, 2])
with fb_c1:
    year_opts = list(active.keys())
    year_sel = st.radio("Year:", year_opts, horizontal=True)
    df_fb_source = {'2022': df22, '2025': df25, '2026': df26}[year_sel]

    if year_sel == '2026':
        support_col = SUPPORT_COL_26
    else:
        support_col = SUPPORT_COL_25

    fb_opts = {}
    if support_col in df_fb_source.columns:
        fb_opts["Specific Support Requested"] = support_col
    if EAP_NO_COL in df_fb_source.columns:
        fb_opts["EAP Non-Usage Reasons"] = EAP_NO_COL

    sel_cat = st.radio("Feedback Category:", list(fb_opts.keys()) if fb_opts else ["No data"])
    target_col = fb_opts.get(sel_cat)

    if target_col:
        valid_fb = df_fb_source[df_fb_source[target_col].notna()]
        valid_fb = valid_fb[~valid_fb[target_col].astype(str).str.lower().str.strip().isin(JUNK_SET)]
        unique_responses = valid_fb[target_col].unique().tolist()
        sel_response = st.selectbox(f"Responses ({len(unique_responses)} available):",
                                    ["-- Select --"] + unique_responses)
    else:
        sel_response = None

with fb_c2:
    if target_col and sel_response and sel_response != "-- Select --":
        row = valid_fb[valid_fb[target_col] == sel_response].iloc[0]
        year_color = YEAR_COLORS[year_sel]
        st.markdown(f"""
        <div class="feedback-card" style="border-left-color:{year_color}">
            <h4>Respondent Profile — {year_sel}</h4>
            <p><b>ID:</b> {row['Respondent ID']} &nbsp;|&nbsp;
               <b>Gender:</b> {row['Please select your gender']} &nbsp;|&nbsp;
               <b>Age:</b> {row['Kindly select your age bracket']}</p>
            <p><b>Department:</b> {row[dept_col]}</p>
            <hr>
            <p style="font-size:1.05rem; color:#333; font-style:italic;">"{sel_response}"</p>
            <hr>
            <p><b>Counsellor Link-Up Requested?</b>
               <span style="color:{year_color}; font-weight:700;">{row.get(COUNSEL_COL, 'N/A')}</span></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Select a response to view the respondent's full profile.")

# =========================
# EXECUTIVE SUMMARY
# =========================
st.markdown('<div class="section-header">📝 Executive Summary & Recommendations</div>', unsafe_allow_html=True)

def get_top_issue(df, year):
    ic = issue_counts(df, year)
    if ic.empty: return "N/A"
    return ic.sort_values('Count', ascending=False).iloc[0]['Topic']

snap_cols = st.columns(len(active))
palette = {'2022': st.info, '2025': st.success, '2026': st.warning}
for col_el, (year, df) in zip(snap_cols, active.items()):
    n = len(df)
    mw_v = pct(df[MW_COL], POS_STATES) if MW_COL in df.columns else 0
    eap_v = pct(df[EAP_COL], ['Yes']) if EAP_COL in df.columns else 0
    aw_v = pct(df[AWARENESS_COL], ['Yes']) if AWARENESS_COL in df.columns else 0
    bad_sleep = pct(df[SLEEP_Q_COL], ['Bad', 'Very Bad']) if SLEEP_Q_COL in df.columns else 0
    top = get_top_issue(df, year)
    with col_el:
        palette[year](f"""
**📊 {year} Snapshot**
- **{n:,}** respondents
- **{mw_v:.1f}%** positive wellbeing
- **{eap_v:.1f}%** EAP awareness
- **{aw_v:.1f}%** attended sessions
- **{bad_sleep:.1f}%** poor sleep quality
- Top focus area: **{top}**
        """)

st.markdown("""
**🚀 Strategic Recommendations (2026):**
1. **Sustained Momentum** — build on improving session attendance; expand to departments with lowest participation.
2. **EAP Trust Campaign** — address confidentiality barriers uncovered in feedback explorer.
3. **Sleep & Fatigue Management** — introduce sleep hygiene training for operational staff (Flight Ops, Technical).
4. **Nicotine & Alcohol Programme** — 2026 data reveals use rates; act on expressed quit-support interest.
5. **Manager Wellbeing Integration** — leverage manager support data to identify teams needing culture intervention.
6. **Direct Counsellor Outreach** — prioritise respondents who explicitly requested counsellor link-up.
""")





st.caption("Kenya Airways Wellness Dashboard   |   Mwenda Kimathi - Associate General Manager CIA MINET    |   Created by Lewis © 2026")