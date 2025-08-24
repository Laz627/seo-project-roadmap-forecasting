import streamlit as st
import pandas as pd
import altair as alt
import json
import math
import hashlib
from io import StringIO

# ===============================
# App Configuration
# ===============================
st.set_page_config(page_title="SEO Roadmap Planner — Lead Gen", page_icon="🗺️", layout="wide")

# ===============================
# Constants & Defaults
# ===============================
BUCKET_OPTIONS = ["Defense", "BAU", "Core", "Bet"]
MECHANISMS = {
    "Rank→CTR (Keyword CSV)": "rank_to_ctr",
    "Defense (Protect at-risk traffic)": "defense",
    "Direct CTR (BAU micro-tweaks)": "direct_ctr"
}

CONF_DEFAULTS = {
    "Defense": 0.80,
    "BAU":     0.60,
    "Core":    0.65,
    "Bet":     0.45
}
ATTR_DEFAULTS = {
    "Defense": 0.70,
    "BAU":     0.70,
    "Core":    0.70,
    "Bet":     0.60
}

FORECAST_PRESETS = {
    "Conservative": {
        "help": "Small improvements where we’re already close to page 1–2.",
        "tiers": [
            {"range": (1, 3), "gain": 0.5},
            {"range": (4, 10), "gain": 1.0},
            {"range": (11, 20), "gain": 2.0},
            {"range": (21, 100), "gain": 0.0}
        ]
    },
    "Moderate": {
        "help": "Balanced, realistic step-up across most rankings.",
        "tiers": [
            {"range": (1, 3), "gain": 1.0},
            {"range": (4, 10), "gain": 2.5},
            {"range": (11, 20), "gain": 5.0},
            {"range": (21, 100), "gain": 8.0}
        ]
    },
    "Aggressive": {
        "help": "Best case if everything lands and we execute exceptionally.",
        "tiers": [
            {"range": (1, 3), "gain": 1.5},
            {"range": (4, 10), "gain": 4.0},
            {"range": (11, 20), "gain": 8.0},
            {"range": (21, 100), "gain": 15.0}
        ]
    }
}

# CTR curve for ranks 1..20 only (page 3+ is zero)
DEFAULT_CTR_CURVE = {
    1: 0.30, 2: 0.18, 3: 0.12, 4: 0.08, 5: 0.06, 6: 0.045, 7: 0.035, 8: 0.028, 9: 0.022, 10: 0.018,
    11: 0.015, 12: 0.012, 13: 0.010, 14: 0.008, 15: 0.007, 16: 0.006, 17: 0.005, 18: 0.004, 19: 0.003, 20: 0.002
}

# ===============================
# Helpers
# ===============================
def normalize_ctr_curve(df: pd.DataFrame) -> dict:
    """Accept CSV with columns (Position, CTR) where CTR can be % or proportion; clamp to 1..20."""
    cols = [c.strip().lower() for c in df.columns]
    df = df.copy(); df.columns = cols
    pos_col = 'position' if 'position' in cols else cols[0]
    ctr_col = 'ctr' if 'ctr' in cols else cols[1]
    curve = {}
    for _, r in df.iterrows():
        try:
            p = int(round(float(r[pos_col])))
            if p < 1 or p > 20:
                continue
            c = float(r[ctr_col])
            if c > 1:  # percent → proportion
                c = c / 100.0
            curve[p] = max(0.0, min(1.0, c))
        except Exception:
            continue
    final = DEFAULT_CTR_CURVE.copy()
    final.update(curve)
    return final

@st.cache_data
def get_ctr_interp(pos: float, curve: dict) -> float:
    """Linear interpolation between integer ranks 1..20; positions >20 return 0 CTR (page 3+)."""
    if pos > 20:
        return 0.0
    if pos < 1:
        pos = 1.0
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    lo = max(1, lo); hi = min(20, hi)
    if lo == hi:
        return curve.get(lo, 0.0)
    lo_ctr = curve.get(lo, 0.0); hi_ctr = curve.get(hi, 0.0)
    w = pos - lo
    return lo_ctr * (1 - w) + hi_ctr * w

def apply_forecast_logic(position: float, preset: str) -> float:
    """Apply tiered rank gain; never better than pos=1. No lift for pos=1 handled later."""
    if position <= 1.0:
        return 1.0
    for tier in FORECAST_PRESETS[preset]["tiers"]:
        lo, hi = tier["range"]
        if lo <= position <= hi:
            return max(1.0, position - tier["gain"])
    return position

def bands_from_p50(p50: float, conf: float, width_cap: float = 0.60):
    """Uncertainty bands: lower confidence ⇒ wider band (up to ±60%)."""
    width = width_cap * (1.0 - float(conf))
    p10 = max(0.0, p50 * (1 - width))
    p90 = p50 * (1 + width)
    return p10, p50, p90

def _hash_project(proj, curve):
    key = json.dumps(proj, sort_keys=True) + json.dumps(curve, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()

# ===============================
# Core calculations (lead gen)
# ===============================
@st.cache_data
def process_project_calculations(project, curve):
    """
    Mechanisms:
      - rank_to_ctr: needs kw_data_json (Keyword, Position, Search Volume/Impressions)
      - defense: sessions_at_risk (monthly), prob_of_issue
      - direct_ctr: monthly_impressions, ctr_delta_pp, (optional) fraction_impacted

    Lead-gen funnel:
      Annual Impact (P50) = Clicks/Visits × Lead CVR × Close Rate × Booking Value × Attribution × 12
      Priority (P50) = (Impact P50 × Confidence) / Effort
      P10/P90 bands derived from Confidence.
    """
    mech = project["mechanism"]
    conf = project["confidence"]
    effort = max(project["effort"], 0.01)

    # Compute Additional Clicks / Visits per month depending on mechanism
    if mech == "rank_to_ctr":
        kw_df = pd.read_json(StringIO(project['kw_data_json']), orient='split')
        kw_df['Position'] = pd.to_numeric(kw_df['Position'], errors='coerce')
        vol_col = 'Search Volume'
        if vol_col not in kw_df.columns:
            raise ValueError("Keyword table must include 'Search Volume' column.")
        kw_df[vol_col] = pd.to_numeric(kw_df[vol_col], errors='coerce')
        kw_df = kw_df.dropna(subset=['Position', vol_col])

        kw_df['Current CTR']      = kw_df['Position'].apply(lambda p: get_ctr_interp(p, curve))
        kw_df['Target Position']  = kw_df['Position'].apply(lambda p: apply_forecast_logic(p, project['forecast_preset']))
        kw_df['Target CTR']       = kw_df['Target Position'].apply(lambda p: get_ctr_interp(p, curve))

        def ctr_delta(row):
            if row['Position'] <= 1.0:
                return 0.0
            return max(0.0, row['Target CTR'] - row['Current CTR'])
        kw_df['CTR Delta'] = kw_df.apply(ctr_delta, axis=1)
        kw_df['Additional Clicks'] = kw_df['Search Volume'] * kw_df['CTR Delta']

        fraction_impacted = project.get('fraction_impacted', 1.0)
        monthly_clicks = kw_df['Additional Clicks'].sum() * fraction_impacted
        details_df = kw_df

    elif mech == "defense":
        # Sessions_at_risk are visits already; this is a "loss avoided" model.
        monthly_clicks = project['sessions_at_risk'] * project['prob_of_issue']
        details_df = pd.DataFrame([{
            "Mechanism": "Defense",
            "Sessions at Risk (monthly)": project['sessions_at_risk'],
            "Probability of Issue": project['prob_of_issue'],
            "Expected Lost Sessions (monthly)": monthly_clicks
        }])

    elif mech == "direct_ctr":
        # Micro CTR gains applied to impressions
        fraction_impacted = project.get('fraction_impacted', 1.0)
        monthly_clicks = project['monthly_impressions'] * (project['ctr_delta_pp'] / 100.0) * fraction_impacted
        details_df = pd.DataFrame([{
            "Mechanism": "Direct CTR (BAU)",
            "Monthly Impressions": project['monthly_impressions'],
            "CTR Δ (pp)": project['ctr_delta_pp'],
            "Fraction Impacted": fraction_impacted,
            "Additional Clicks (monthly)": monthly_clicks
        }])

    else:
        raise ValueError("Unknown mechanism.")

    # Lead-gen funnel to bookings
    lead_cvr      = project['lead_cvr']
    close_rate    = project['close_rate']
    booking_value = project['booking_value']
    attribution   = project['attribution']

    impact_p50_mo = monthly_clicks * lead_cvr * close_rate * booking_value * attribution
    impact_p50    = impact_p50_mo * 12  # annualize
    priority_p50  = (impact_p50 * conf) / effort
    imp_p10, imp_p50, imp_p90 = bands_from_p50(impact_p50, conf)
    prio_10, prio_50, prio_90 = bands_from_p50(priority_p50, conf)

    return {
        "clicks_p50_monthly": monthly_clicks,
        "impact_p10": imp_p10, "impact_p50": imp_p50, "impact_p90": imp_p90,
        "priority_p10": prio_10, "priority_p50": prio_50, "priority_p90": prio_90,
        "details_df": details_df
    }

# ===============================
# Session state
# ===============================
if 'projects' not in st.session_state:
    st.session_state.projects = []

# ===============================
# Sidebar — CTR curve, files, capacity
# ===============================
st.sidebar.header("CTR Curve (Optional)")
st.sidebar.caption("Used only for **Rank→CTR** projects. We model page 1–2 (positions 1–20); page 3+ is 0 CTR.")
ctr_file = st.sidebar.file_uploader("Upload CTR Curve CSV (Position, CTR; 1..20 only)", type=["csv"])
if ctr_file is not None:
    try:
        CTR_CURVE = normalize_ctr_curve(pd.read_csv(ctr_file))
        st.sidebar.success("CTR curve loaded.")
    except Exception as e:
        CTR_CURVE = DEFAULT_CTR_CURVE
        st.sidebar.error(f"Failed to load CTR curve; using defaults. ({e})")
else:
    CTR_CURVE = DEFAULT_CTR_CURVE

st.sidebar.markdown("---")
st.sidebar.header("Roadmap Files")
if st.session_state.projects:
    st.sidebar.download_button(
        label="⬇️ Download Roadmap (JSON)",
        data=json.dumps(st.session_state.projects, indent=2),
        file_name="seo_roadmap_leadgen.json",
        mime="application/json"
    )
uploaded_roadmap = st.sidebar.file_uploader("⬆️ Upload Roadmap (JSON)", type=['json'])
if uploaded_roadmap is not None:
    try:
        new_projects = json.load(uploaded_roadmap)
        if isinstance(new_projects, list) and all('name' in p for p in new_projects):
            st.session_state.projects = new_projects
            st.sidebar.success("Roadmap loaded.")
            st.rerun()
        else:
            st.sidebar.error("Invalid roadmap format.")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

st.sidebar.markdown("---")
default_hide = st.sidebar.checkbox(
    "Default: Hide Page 3+ only rows in keyword tables",
    value=True,
    help="For Rank→CTR details, hide rows where both current and target positions are >20."
)

st.sidebar.header("Capacity by Quarter")
st.sidebar.caption("Use effort points to check if quarters are over- or under-committed.")
cap_q1 = st.sidebar.number_input("Q1 Capacity (Effort pts)", min_value=0, value=50)
cap_q2 = st.sidebar.number_input("Q2 Capacity (Effort pts)", min_value=0, value=50)
cap_q3 = st.sidebar.number_input("Q3 Capacity (Effort pts)", min_value=0, value=50)
cap_q4 = st.sidebar.number_input("Q4 Capacity (Effort pts)", min_value=0, value=50)
capacity_map = {"Unassigned": 0, "Q1": cap_q1, "Q2": cap_q2, "Q3": cap_q3, "Q4": cap_q4}

st.sidebar.markdown("---")
st.sidebar.header("Add New Project")

with st.sidebar.form("new_project_form", clear_on_submit=True):
    task_name = st.text_input("Project Name", placeholder="e.g., Improve 'Solutions' hub IA",
                              help="Action-oriented name that reads well on a slide.")
    strategic_bucket = st.selectbox("Strategic Bucket", BUCKET_OPTIONS, index=2,
                                    help="Defense=protect revenue • BAU=routine gains • Core=main growth • Bet=high risk/high reward")
    mechanism_label = st.selectbox("Mechanism", list(MECHANISMS.keys()),
                                   help="Pick the way value is created so we show the right inputs & math.")
    mechanism = MECHANISMS[mechanism_label]

    # Defaults from bucket
    default_conf = CONF_DEFAULTS.get(strategic_bucket, 0.65)
    default_attr = ATTR_DEFAULTS.get(strategic_bucket, 0.70)

    # Mechanism-specific inputs
    forecast_preset = None
    fraction_impacted = 1.0
    sessions_at_risk = None
    prob_of_issue = None
    monthly_impressions = None
    ctr_delta_pp = None

    if mechanism == "rank_to_ctr":
        uploaded_kw = st.file_uploader(
            "Upload Keyword List (CSV)",
            type=["csv"],
            help="Columns required: Keyword, Position, and Search Volume (or Impressions)."
        )
        forecast_preset = st.radio("Forecast Scenario", options=list(FORECAST_PRESETS.keys()), horizontal=True,
                                   help="How much rank improvement to assume by tier (e.g., +1.0 rank for 4–10).")
        st.caption("Preset details: "
                   "• Conservative (≈+0.5 at ranks 1–3; +1 (4–10); +2 (11–20))  "
                   "• Moderate (≈+1; +2.5; +5)  "
                   "• Aggressive (≈+1.5; +4; +8)")
        fraction_impacted = st.slider("Fraction of Keywords Impacted", 0.0, 1.0, 1.0, 0.05,
                                      help="Share of uploaded keywords expected to benefit.")
    elif mechanism == "defense":
        sessions_at_risk = st.number_input("Monthly Sessions at Risk", min_value=0, step=1000,
                                           help="Monthly organic sessions that could be lost without this work.")
        prob_of_issue = st.slider("Probability of Issue", 0.0, 1.0, 0.4, 0.05,
                                  help="Likelihood of negative impact (e.g., CWV, migration risk).")
    elif mechanism == "direct_ctr":
        monthly_impressions = st.number_input("Monthly Impressions", min_value=0, step=1000,
                                              help="Impressions across the affected pages/cluster.")
        ctr_delta_pp = st.number_input("Estimated CTR Increase (percentage points)", min_value=0.0, step=0.05, format="%.2f",
                                       help="Example: 0.30 pp for an H3→H2 cleanup across titles/descriptions.")
        fraction_impacted = st.slider("Fraction of Impressions Impacted", 0.0, 1.0, 1.0, 0.05,
                                      help="Share of impressions where the change will take effect.")

    # Lead-gen funnel inputs
    lead_cvr = st.number_input("Lead Form Conversion Rate (%)", min_value=0.0, step=0.1, format="%.2f",
                               help="Percent of incremental visits that submit a lead form.") / 100.0
    close_rate = st.number_input("Sales Close Rate (%)", min_value=0.0, step=0.1, format="%.2f",
                                 help="Percent of leads that become closed/won deals.") / 100.0
    booking_value = st.number_input("Avg Booking Value ($)", min_value=0, step=100,
                                    help="Average contract value (ACV) per closed/won deal.")

    effort = st.number_input("Effort (Points)", min_value=1, step=1,
                             help="Relative size. Use a consistent scale across projects.")
    confidence = st.slider("Confidence (%)", 0, 100, int(default_conf * 100), 5,
                           help="Higher confidence narrows the P10–P90 band and raises Priority.") / 100.0
    attribution = st.slider("SEO Attribution (%)", 0, 100, int(default_attr * 100), 5,
                            help="Share of bookings credited to SEO (vs. other channels).") / 100.0

    target_qtr = st.selectbox("Target Quarter", ["Unassigned", "Q1", "Q2", "Q3", "Q4"],
                              help="When we plan to do the work.")
    owner = st.text_input("Owner / Team", value="", help="Primary owner or team.")
    status = st.selectbox("Status", ["Backlog", "In Progress", "Completed", "On Hold"], help="Lightweight tracking.")

    add_button = st.form_submit_button("Add Project to Roadmap")
    if add_button:
        if not task_name:
            st.sidebar.error("Project Name is required.")
        else:
            new_project = {
                "name": task_name,
                "bucket": strategic_bucket,
                "mechanism": mechanism,
                "forecast_preset": forecast_preset,   # only meaningful for rank_to_ctr
                "fraction_impacted": fraction_impacted,
                "sessions_at_risk": sessions_at_risk,
                "prob_of_issue": prob_of_issue,
                "monthly_impressions": monthly_impressions,
                "ctr_delta_pp": ctr_delta_pp,
                # lead-gen funnel
                "lead_cvr": lead_cvr,
                "close_rate": close_rate,
                "booking_value": booking_value,
                # business & planning
                "effort": effort,
                "confidence": confidence,
                "attribution": attribution,
                "target_quarter": target_qtr,
                "owner": owner,
                "status": status,
                # post-launch tracking
                "actual_clicks_monthly": None,
                "actual_revenue_annual": None
            }

            if mechanism == "rank_to_ctr":
                uploaded_kw = st.session_state.get('uploaded_kw_side', None)  # not used; we'll read directly
                uploaded_kw = st.sidebar.session_state.get('new_project_form-uploaded_kw') if 'new_project_form-uploaded_kw' in st.sidebar.session_state else None

            if mechanism == "rank_to_ctr":
                # Read the uploaded file from the widget directly
                uploaded_kw = st.sidebar.session_state.get('new_project_form-Upload Keyword List (CSV)')
                # Fallback: Streamlit doesn't always expose by key label; instead re-read via the variable we created
                uploaded_kw = uploaded_kw or st.sidebar.session_state.get('Upload Keyword List (CSV)')
                uploaded_kw = uploaded_kw or st.sidebar.session_state.get('uploaded_kw')

            # Easiest & robust: re-prompt on missing upload
            # (Streamlit widget session keys vary by version; so we just use the local reference if available.)
            if mechanism == "rank_to_ctr":
                # Use the local variable from above scope if present
                # (Streamlit provides it directly as 'uploaded_kw' variable)
                pass

            if mechanism == "rank_to_ctr":
                if 'uploaded_kw' not in locals() or uploaded_kw is None:
                    st.sidebar.error("Please upload a keyword CSV for Rank→CTR.")
                    st.stop()
                kw_df = pd.read_csv(uploaded_kw)
                kw_df.columns = [c.strip() for c in kw_df.columns]
                if 'Search Volume' not in kw_df.columns and 'Impressions' in kw_df.columns:
                    kw_df = kw_df.rename(columns={'Impressions': 'Search Volume'})
                required = {'Keyword', 'Position', 'Search Volume'}
                if not required.issubset(set(kw_df.columns)):
                    st.sidebar.error("CSV must include: Keyword, Position, and Search Volume (or Impressions).")
                    st.stop()
                kw_df['Position'] = pd.to_numeric(kw_df['Position'], errors='coerce')
                kw_df['Search Volume'] = pd.to_numeric(kw_df['Search Volume'], errors='coerce')
                kw_df = kw_df.dropna(subset=['Position', 'Search Volume'])
                new_project["kw_data_json"] = kw_df.to_json(orient='split')

            st.session_state.projects.append(new_project)
            st.rerun()

# ===============================
# Main UI
# ===============================
st.title("🗺️ SEO Roadmap Planner — Lead Gen")
st.caption("Rules: **No CTR for positions >20 (page 3+)** and **no added clicks for position 1**. CTR is interpolated within ranks 1–20 (for Rank→CTR projects).")

tab_dashboard, tab_roadmap, tab_manage = st.tabs(["📈 Dashboard", "📋 Roadmap", "🛠️ Manage & Edit"])

if not st.session_state.projects:
    with tab_dashboard: st.info("Your roadmap is empty. Add a project from the sidebar to begin.")
    with tab_roadmap:   st.info("Your roadmap is empty. Add a project from the sidebar to begin.")
    with tab_manage:    st.info("Your roadmap is empty. Add a project from the sidebar to begin.")
else:
    # Compute summaries
    summary_rows, details_cache = [], {}
    for i, proj in enumerate(st.session_state.projects):
        res = process_project_calculations(proj, CTR_CURVE)
        details_cache[i] = res["details_df"]
        summary_rows.append({
            "Project": proj["name"],
            "Bucket": proj.get("bucket","Core"),
            "Mechanism": proj.get("mechanism","rank_to_ctr"),
            "Forecast": proj.get("forecast_preset","—") if proj.get("mechanism")=="rank_to_ctr" else "—",
            "Fraction Impacted": proj.get("fraction_impacted", 1.0),
            "Impact P10": res["impact_p10"],
            "Impact P50": res["impact_p50"],
            "Impact P90": res["impact_p90"],
            "Monthly Addl Clicks (P50)": res["clicks_p50_monthly"],
            "Effort": proj["effort"],
            "Confidence": proj["confidence"],
            "Attribution": proj["attribution"],
            "Priority P10": res["priority_p10"],
            "Priority P50": res["priority_p50"],
            "Priority P90": res["priority_p90"],
            "Quarter": proj.get("target_quarter","Unassigned"),
            "Owner": proj.get("owner",""),
            "Status": proj.get("status","Backlog"),
            "project_index": i
        })
    roadmap_df = pd.DataFrame(summary_rows).sort_values(by="Priority P50", ascending=False).reset_index(drop=True)

    # ---------- Dashboard ----------
    with tab_dashboard:
        st.subheader("2×2 Priority Matrix (Impact vs Effort)")
        st.caption("Bubble = project. **Y**: annual booking impact (P50). **X**: effort. Error bars show uncertainty (P10–P90).")
        dash_num = roadmap_df.copy()
        dash_num["Impact"] = dash_num["Impact P50"]
        dash_num["Impact_low"] = dash_num["Impact P10"]
        dash_num["Impact_high"] = dash_num["Impact P90"]
        if len(dash_num):
            impact_thresh = float(dash_num["Impact"].median())
            effort_thresh = float(dash_num["Effort"].median())
            points = alt.Chart(dash_num).mark_circle(size=220, opacity=0.85).encode(
                x=alt.X('Effort:Q', title='Effort (relative units)'),
                y=alt.Y('Impact:Q', title='Annual Booking Impact (P50)', axis=alt.Axis(format='$,.0f')),
                color=alt.Color('Bucket:N', legend=alt.Legend(title='Bucket')),
                shape=alt.Shape('Status:N', legend=alt.Legend(title='Status')),
                tooltip=[
                    alt.Tooltip('Project:N', title='Project'),
                    alt.Tooltip('Mechanism:N', title='Mechanism'),
                    alt.Tooltip('Bucket:N', title='Bucket'),
                    alt.Tooltip('Status:N', title='Status'),
                    alt.Tooltip('Owner:N', title='Owner'),
                    alt.Tooltip('Quarter:N', title='Quarter'),
                    alt.Tooltip('Effort:Q', title='Effort'),
                    alt.Tooltip('Impact:Q', title='Impact (P50)', format='$,.0f'),
                    alt.Tooltip('Impact_low:Q', title='Impact P10', format='$,.0f'),
                    alt.Tooltip('Impact_high:Q', title='Impact P90', format='$,.0f'),
                    alt.Tooltip('Priority P50:Q', title='Priority (P50)', format=',.0f')
                ]
            )
            rules = alt.Chart(dash_num).mark_rule(color='gray').encode(
                x='Effort:Q', y=alt.Y('Impact_low:Q', title=''), y2='Impact_high:Q'
            )
            vline = alt.Chart(pd.DataFrame({'Effort':[effort_thresh]})).mark_rule(color='gray').encode(x='Effort:Q')
            hline = alt.Chart(pd.DataFrame({'Impact':[impact_thresh]})).mark_rule(color='gray').encode(y='Impact:Q')
            st.altair_chart((rules + points + vline + hline).interactive(), use_container_width=True)
        else:
            st.info("Add projects to see the matrix.")

        st.markdown("---")
        st.subheader("Effort Allocation by Strategic Bucket")
        st.caption("How planned effort is distributed across Defense / BAU / Core / Bet.")
        if len(roadmap_df):
            bucket_df = roadmap_df.groupby('Bucket', as_index=False)['Effort'].sum()
            bar = alt.Chart(bucket_df).mark_bar().encode(
                x=alt.X('Bucket:N', title='Strategic Bucket'),
                y=alt.Y('Effort:Q', title='Total Effort'),
                tooltip=[alt.Tooltip('Bucket:N', title='Bucket'),
                         alt.Tooltip('Effort:Q', title='Effort', format=',.0f')]
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("Add projects to see effort allocation.")

        # === Strategic Bucket Cheat Sheet (added) ===
        st.subheader("Reference: Strategic Bucket Cheat Sheet")
        with st.expander("Click here to see how common SEO tasks map to strategic buckets"):
            st.markdown("""
            | Activity | Primary Bucket(s) | Strategic Rationale & Intent |
            | :--- | :--- | :--- |
            | **Cleaning Up Tech Debt** | 🛡️ **Defense** | Fix underlying problems, prevent future issues, and protect existing performance (e.g., CWV, crawlability). |
            | **Enhancing Existing Pages** | 🌱 **Core** or ⚙️ **BAU** | **Core** for a major overhaul of a key commercial page. **BAU** for a minor content refresh. |
            | **Creating New Pages** | 🌱 **Core** or 🎯 **Bet** | **Core** for creating pages using a proven, scalable template. A **Bet** for a major new content hub in an unproven topic area. |
            | **Performing SEO Testing** | 🎯 **Bet** or 🌱 **Core** | A **Bet** for a large-scale, high-variance test (e.g., new navigation). **Core** for a controlled test as part of continuous optimization (e.g., title tags). |
            | **Mild Changes (e.g., internal link swaps)** | ⚙️ **BAU** | Low-effort, routine hygiene. Part of keeping the site healthy with small, incremental impact. |
            """)

    # ---------- Roadmap ----------
    with tab_roadmap:
        st.header("Prioritized SEO Roadmap — Lead Gen")
        st.caption("Priority = (Annual Booking Impact × Confidence) / Effort. We show **P50** and a **P10–P90** range to make uncertainty explicit.")

        disp = roadmap_df.copy()
        disp["Impact (P50)"] = disp["Impact P50"].apply(lambda x: f"${x:,.0f}")
        disp["Impact Range (P10–P90)"] = disp.apply(lambda r: f"${r['Impact P10']:,.0f}–${r['Impact P90']:,.0f}", axis=1)
        disp["Monthly Addl Clicks (P50)"] = disp["Monthly Addl Clicks (P50)"].apply(lambda x: f"{x:,.0f}")
        disp["Priority (P50)"] = disp["Priority P50"].apply(lambda x: f"{x:,.0f}")
        disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{x:.0%}")
        disp["Attribution"] = disp["Attribution"].apply(lambda x: f"{x:.0%}")
        disp["Fraction Impacted"] = disp["Fraction Impacted"].apply(lambda x: f"{x:.0%}")

        st.dataframe(
            disp[["Project","Bucket","Mechanism","Forecast","Quarter","Owner","Status","Fraction Impacted",
                  "Impact (P50)","Impact Range (P10–P90)","Monthly Addl Clicks (P50)",
                  "Effort","Confidence","Attribution","Priority (P50)"]],
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Quarterly Plan & Capacity")
        st.caption("Compares planned effort to available capacity by quarter.")
        if len(roadmap_df):
            plan = roadmap_df.groupby('Quarter', as_index=False)['Effort'].sum()
            plan['Capacity'] = plan['Quarter'].map(capacity_map).fillna(0)
            plan['Over/Under'] = plan['Capacity'] - plan['Effort']
            plan['Over Capacity?'] = plan['Over/Under'] < 0
            st.dataframe(
                plan[['Quarter','Effort','Capacity','Over/Under','Over Capacity?']].style.format({
                    'Effort':'{:,.0f}','Capacity':'{:,.0f}','Over/Under':'{:,.0f}'
                }),
                use_container_width=True
            )
        else:
            st.info("Assign Target Quarters to projects to see planning & capacity.")

        st.markdown("---")
        st.subheader("Exports")
        out_summary = roadmap_df.drop(columns=['project_index']).copy()
        st.download_button(
            "⬇️ Download Prioritized Summary (CSV)",
            data=out_summary.to_csv(index=False).encode('utf-8'),
            file_name="seo_roadmap_summary_leadgen.csv",
            mime="text/csv"
        )

    # ---------- Manage & Edit ----------
    with tab_manage:
        st.header("Manage & Edit Projects")
        for rank_idx, row in roadmap_df.iterrows():
            idx = row['project_index']
            proj = st.session_state.projects[idx]
            mech = proj.get('mechanism')

            label = [k for k,v in MECHANISMS.items() if v == mech][0] if mech in MECHANISMS.values() else mech
            with st.expander(f"**{proj['name']}** (Rank #{rank_idx + 1} | Priority P50: {row['Priority P50']:,.0f} | {label})"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader("Edit Project Details")
                    with st.form(key=f"edit_form_{idx}"):
                        edited_name = st.text_input("Project Name", value=proj['name'], key=f"name_{idx}")
                        edited_bucket = st.selectbox("Strategic Bucket", BUCKET_OPTIONS,
                                                     index=BUCKET_OPTIONS.index(proj.get('bucket','Core')),
                                                     key=f"bucket_{idx}")
                        # Mechanism is not mutable post-creation to keep fields consistent
                        st.text_input("Mechanism", value=label, disabled=True, key=f"mech_{idx}",
                                      help="Mechanism is fixed for this project. Create a new project to change it.")

                        # Mechanism-specific editors
                        if mech == "rank_to_ctr":
                            edited_forecast = st.radio("Forecast Scenario", options=list(FORECAST_PRESETS.keys()),
                                                       index=list(FORECAST_PRESETS.keys()).index(proj['forecast_preset']),
                                                       horizontal=True, key=f"forecast_{idx}",
                                                       help="Controls expected rank improvement by tier.")
                            edited_frac = st.slider("Fraction of Keywords Impacted", 0.0, 1.0,
                                                    value=proj.get('fraction_impacted',1.0), key=f"frac_{idx}")
                        elif mech == "defense":
                            edited_forecast = None
                            edited_frac = proj.get('fraction_impacted', 1.0)  # unused but preserved
                            cda, cdb = st.columns(2)
                            sessions_at_risk = cda.number_input("Monthly Sessions at Risk",
                                                                value=int(proj.get('sessions_at_risk') or 0),
                                                                min_value=0, step=1000, key=f"sar_{idx}")
                            prob_of_issue = cdb.slider("Probability of Issue", 0.0, 1.0,
                                                       value=float(proj.get('prob_of_issue') or 0.4),
                                                       key=f"poi_{idx}")
                        elif mech == "direct_ctr":
                            edited_forecast = None
                            cda, cdb, cdc = st.columns(3)
                            monthly_impressions = cda.number_input("Monthly Impressions",
                                                                   value=int(proj.get('monthly_impressions') or 0),
                                                                   min_value=0, step=1000, key=f"imps_{idx}")
                            ctr_delta_pp = cdb.number_input("Estimated CTR Increase (pp)",
                                                            value=float(proj.get('ctr_delta_pp') or 0.0),
                                                            min_value=0.0, step=0.05, format="%.2f", key=f"pp_{idx}")
                            edited_frac = cdc.slider("Fraction of Impressions Impacted", 0.0, 1.0,
                                                     value=proj.get('fraction_impacted',1.0), key=f"frac_{idx}")

                        # Lead-gen funnel + planning
                        c1, c2, c3 = st.columns(3)
                        edited_lead_cvr = c1.number_input("Lead Form Conversion Rate (%)",
                                                          value=proj['lead_cvr']*100, key=f"leadcvr_{idx}") / 100.0
                        edited_close = c2.number_input("Sales Close Rate (%)",
                                                       value=proj['close_rate']*100, key=f"close_{idx}") / 100.0
                        edited_booking = c3.number_input("Avg Booking Value ($)",
                                                         value=proj['booking_value'], key=f"book_{idx}")

                        c4, c5, c6 = st.columns(3)
                        edited_effort = c4.number_input("Effort", value=proj['effort'], key=f"effort_{idx}")
                        edited_conf = c5.slider("Confidence (%)", 0, 100, value=int(proj['confidence']*100),
                                                key=f"conf_{idx}") / 100.0
                        edited_attr = c6.slider("SEO Attribution (%)", 0, 100, value=int(proj['attribution']*100),
                                                key=f"attr_{idx}") / 100.0

                        c7, c8 = st.columns(2)
                        edited_qtr = c7.selectbox("Target Quarter", ["Unassigned","Q1","Q2","Q3","Q4"],
                                                  index=["Unassigned","Q1","Q2","Q3","Q4"].index(proj.get('target_quarter',"Unassigned")),
                                                  key=f"qtr_{idx}")
                        edited_status = c8.selectbox("Status", ["Backlog","In Progress","Completed","On Hold"],
                                                     index=["Backlog","In Progress","Completed","On Hold"].index(proj.get('status',"Backlog")),
                                                     key=f"status_{idx}")
                        edited_owner = st.text_input("Owner / Team", value=proj.get('owner',""), key=f"owner_{idx}")

                        # Actuals if completed (optional tracking)
                        if edited_status == "Completed":
                            a1, a2 = st.columns(2)
                            edited_actual_clicks = a1.number_input("Actual Monthly Clicks/Visits Gained (or Saved)",
                                                                   value=float(proj.get('actual_clicks_monthly') or 0),
                                                                   min_value=0.0, step=1.0, key=f"act_clicks_{idx}")
                            edited_actual_rev = a2.number_input("Actual Annual Bookings ($)",
                                                                value=float(proj.get('actual_revenue_annual') or 0),
                                                                min_value=0.0, step=100.0, key=f"act_rev_{idx}")
                        else:
                            edited_actual_clicks = proj.get('actual_clicks_monthly')
                            edited_actual_rev = proj.get('actual_revenue_annual')

                        update_button = st.form_submit_button("Update Project")
                        if update_button:
                            proj['name'] = edited_name
                            proj['bucket'] = edited_bucket
                            proj['confidence'] = edited_conf
                            proj['attribution'] = edited_attr
                            proj['target_quarter'] = edited_qtr
                            proj['status'] = edited_status
                            proj['owner'] = edited_owner
                            proj['lead_cvr'] = edited_lead_cvr
                            proj['close_rate'] = edited_close
                            proj['booking_value'] = edited_booking
                            proj['effort'] = edited_effort
                            proj['fraction_impacted'] = edited_frac
                            proj['actual_clicks_monthly'] = edited_actual_clicks
                            proj['actual_revenue_annual'] = edited_actual_rev

                            if mech == "rank_to_ctr":
                                proj['forecast_preset'] = edited_forecast
                            elif mech == "defense":
                                proj['sessions_at_risk'] = sessions_at_risk
                                proj['prob_of_issue'] = prob_of_issue
                            elif mech == "direct_ctr":
                                proj['monthly_impressions'] = monthly_impressions
                                proj['ctr_delta_pp'] = ctr_delta_pp
                            st.rerun()

                with col2:
                    st.subheader("Actions")
                    if st.button("Delete Project", key=f"delete_{idx}", type="primary", use_container_width=True):
                        st.session_state.projects.pop(idx)
                        st.rerun()

                st.subheader("Mechanism Details")
                if mech == "rank_to_ctr":
                    st.caption("Tip: Hide keywords that are page 3+ both before and after; they can’t generate clicks.")
                    hide_page3_only = st.checkbox(
                        "Hide rows where current and target positions are both > 20 (Page 3+ only)",
                        value=default_hide, key=f"hide_p3_{idx}"
                    )
                    details_df = details_cache[idx]
                    if hide_page3_only and {"Position","Target Position"}.issubset(details_df.columns):
                        mask = ~((details_df["Position"] > 20) & (details_df["Target Position"] > 20))
                        details_df = details_df.loc[mask].copy()
                    st.dataframe(
                        details_df.style.format({
                            "Search Volume": "{:,.0f}",
                            "Position": "{:.1f}",
                            "Current CTR": "{:.2%}",
                            "Target Position": "{:.1f}",
                            "Target CTR": "{:.2%}",
                            "CTR Delta": "{:.2%}",
                            "Additional Clicks": "{:,.1f}"
                        }),
                        use_container_width=True
                    )
                    st.download_button(
                        "Download Keyword Details (CSV)",
                        data=details_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"{proj['name'].replace(' ','_')}_details.csv",
                        mime="text/csv"
                    )
                else:
                    st.dataframe(details_cache[idx], use_container_width=True)
