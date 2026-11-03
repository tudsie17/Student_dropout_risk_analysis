import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ===============================================================
# SETTINGS (edit these lines if needed)
# ===============================================================
st.set_page_config(page_title="Student Dropout Risk", page_icon="🎓", layout="wide")

DATA_SOURCE = ("[Student Academic Performance & Dropout Risk on Kaggle]"
               "(https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk)")

# Colours: one main colour + one highlight, used on every chart
BASE = "#7C93B0"        # normal (blue-grey)
HIGHLIGHT = "#D1495B"   # highest risk (red)
NEUTRAL = "#BCC3CC"     # "no difference" (light grey)
RISK_SCALE = ["#FBEAEC", "#E58C97", "#D1495B", "#7A1023"]
FONT = "Inter, Segoe UI, sans-serif"


# ===============================================================
# PAGE STYLE (CSS): fonts, banner, cards, tabs
# ===============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, input, textarea { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }

.hero { background: linear-gradient(120deg, #1F2A44 0%, #33456B 55%, #8E3B52 100%);
        color: #fff; padding: 30px 34px; border-radius: 20px; margin-bottom: 18px; }
.hero .title { font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin-bottom: 8px; }
.hero .sub { font-size: 1.02rem; opacity: .88; }
.hero .chip { display: inline-block; margin: 14px 8px 0 0; padding: 4px 12px; border-radius: 999px;
              background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28); font-size: .8rem; }

.kpi { border: 1px solid rgba(128,128,128,.22); border-radius: 16px; padding: 16px 18px;
       background: rgba(128,128,128,.05); height: 100%; }
.kpi .label { font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65; font-weight: 600; }
.kpi .value { font-size: 2rem; font-weight: 800; margin-top: 2px; }
.kpi .note  { font-size: .84rem; opacity: .72; }
.kpi.red    { border-left: 6px solid #D1495B; }
.kpi.red .value { color: #D1495B; }

.unused-insight { border-radius: 16px; padding: 16px 18px; height: 100%;
           background: rgba(209,73,91,.07); border: 1px solid rgba(209,73,91,.25); }
.insight .big { font-size: 1.8rem; font-weight: 800; color: #D1495B; }
.insight .text { font-size: .92rem; }

.rec { border: 1px solid rgba(128,128,128,.22); border-radius: 16px; padding: 16px 18px;
       margin-bottom: 14px; background: rgba(128,128,128,.04); }
.rec .num { display: inline-block; width: 28px; height: 28px; line-height: 28px; text-align: center;
            border-radius: 50%; background: #1F2A44; color: #fff; font-weight: 700; margin-right: 8px; }
.rec .head { font-weight: 700; font-size: 1.02rem; }
.rec .action { margin: 8px 0 6px 0; font-size: .92rem; }
.rec .evidence { font-size: .9rem; }
.rec .evidence b { color: #D1495B; }
.rec .see { font-size: .78rem; opacity: .6; margin-top: 4px; }

.chart-title { font-weight: 700; font-size: 1.05rem; margin: 6px 0 0 0; }
.chart-sub { font-size: .85rem; opacity: .68; margin-bottom: 2px; }

button[data-baseweb="tab"] p { font-size: 0.98rem; font-weight: 600; }

/* calmer filter tags in the sidebar (instead of bright coral) */
[data-testid="stSidebar"] span[data-baseweb="tag"],
[data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] span[data-tag] {
    background-color: rgba(124,147,176,.20) !important; color: inherit !important; }
[data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] span[data-tag] * { color: inherit !important; }
.side-count { font-size: .9rem; opacity: .8; margin-top: 4px; }

.kpi.small { padding: 10px 14px; }
.kpi.small .value { font-size: 1.4rem; }
.finding { border-radius: 16px; padding: 16px 20px; border: 1px solid rgba(128,128,128,.22);
           border-left: 6px solid #D1495B; background: rgba(128,128,128,.04); height: 100%; }
.finding .label { font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65; font-weight: 600; }
.finding .main { font-size: 1.15rem; font-weight: 700; margin-top: 6px; }
.finding .main b { color: #D1495B; }
.notice { border-radius: 12px; padding: 12px 16px; background: rgba(224,159,62,.12);
          border: 1px solid rgba(224,159,62,.45); font-size: .92rem; margin-bottom: 12px; }
@media (max-width: 900px) { .hero .title { font-size: 1.6rem; } .kpi .value { font-size: 1.6rem; } }
</style>
""", unsafe_allow_html=True)


# ===============================================================
# LOAD THE DATA
# ===============================================================
@st.cache_data
def load_data(source):
    return pd.read_csv(source)


FILE_PATH = Path(__file__).parent / "student_dropout_risk_2026.csv"
if FILE_PATH.exists():
    df = load_data(FILE_PATH)
else:
    uploaded = st.file_uploader("Upload student_dropout_risk_2026.csv", type="csv")
    if uploaded is None:
        st.info("Dataset not found next to this file. Upload the CSV to continue.")
        st.stop()
    df = load_data(uploaded)


# ===============================================================
# HELPER FUNCTIONS
# ===============================================================
def dropout_rate(data):
    """Dropout rate in % for a group of students."""
    return data["dropout"].mean() * 100


def rate_table(data, column, order=None, names=None):
    """Dropout rate and number of students for each group in a column."""
    table = data.groupby(column, observed=True)["dropout"].agg(rate="mean", students="size").reset_index()
    table["rate"] *= 100
    table = table.rename(columns={column: "group"})
    if names:
        table["group"] = table["group"].map(names)
    table["group"] = table["group"].astype(str)
    if order:
        table = table.set_index("group").reindex([g for g in order if g in set(table["group"])]).reset_index()
    return table


def rate_of(table, group):
    """One group's rate from a rate_table (None if that group is filtered out)."""
    row = table.loc[table["group"] == group, "rate"]
    return row.iloc[0] if len(row) else None


def pct(value, decimals=0):
    return "n/a" if value is None else f"{value:.{decimals}f}%"


def chart_heading(title, subtitle=None):
    """Short title that states the finding, plus an optional one-line subtitle."""
    st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="chart-sub">{subtitle}</div>', unsafe_allow_html=True)


def tidy(fig, height=340, y_title="Dropout rate (%)"):
    """Same clean look for every chart."""
    fig.update_layout(height=height, margin=dict(t=16, b=40, l=10, r=10), font=dict(family=FONT),
                      showlegend=False, yaxis_title=y_title,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


CHART_COUNT = [0]


def show(fig):
    """Draw a chart. Each one gets its own ID so the same chart can appear twice."""
    CHART_COUNT[0] += 1
    st.plotly_chart(fig, width="stretch", key=f"chart_{CHART_COUNT[0]}",
                    config={"displayModeBar": False})


def bar_chart(table, x_title, highlight=True, height=320):
    """Bar chart of dropout rates: highest group red, the rest one colour, dotted line = average."""
    colours = ([HIGHLIGHT if r == table["rate"].max() else BASE for r in table["rate"]]
               if highlight else [NEUTRAL] * len(table))
    fig = go.Figure(go.Bar(
        x=table["group"], y=table["rate"], marker_color=colours, marker_cornerradius=6,
        text=table["rate"].map("{:.0f}%".format), textposition="outside", cliponaxis=False,
        customdata=table["students"],
        hovertemplate="%{x}<br>Dropout rate: %{y:.1f}%<br>Students: %{customdata:,}<extra></extra>"))
    fig.add_hline(y=avg_rate, line_dash="dot", line_color="#888")
    fig.update_xaxes(type="category", title=x_title)
    fig.update_yaxes(range=[0, min(100, max(50, table["rate"].max() * 1.2))])
    return tidy(fig, height)


def line_chart(table, x_title, height=320):
    """Line chart for factors that rise step by step (e.g. failed units)."""
    colours = [HIGHLIGHT if r == table["rate"].max() else BASE for r in table["rate"]]
    fig = go.Figure(go.Scatter(
        x=table["group"], y=table["rate"], mode="lines+markers+text",
        line=dict(color=BASE, width=3, shape="spline"), fill="tozeroy", fillcolor="rgba(124,147,176,.15)",
        marker=dict(size=11, color=colours, line=dict(color="white", width=2)),
        text=table["rate"].map("{:.0f}%".format), textposition="top center", cliponaxis=False,
        customdata=table["students"],
        hovertemplate="%{x}<br>Dropout rate: %{y:.1f}%<br>Students: %{customdata:,}<extra></extra>"))
    fig.add_hline(y=avg_rate, line_dash="dot", line_color="#888")
    fig.update_xaxes(type="category", title=x_title, range=[-0.4, len(table) - 0.6])
    fig.update_yaxes(range=[0, min(100, table["rate"].max() * 1.25)])
    return tidy(fig, height)


# ===============================================================
# SIDEBAR FILTERS
# ===============================================================
st.sidebar.markdown("## 🔍 Filters")
st.sidebar.caption("Choose which students to include.")


FILTERS = {  # label: (column, fixed order or None)
    "Gender": ("gender", None),
    "Course type": ("course_type", None),
    "Family income": ("family_income_level", ["Low", "Middle", "High"]),
    "Study mode": ("study_mode", None),
}


def filter_options(column, order):
    return order if order else sorted(df[column].unique())


def reset_filters():
    """Tick every option again in every filter."""
    for column, order in FILTERS.values():
        st.session_state[f"filter_{column}"] = filter_options(column, order)


def sidebar_filter(label):
    column, order = FILTERS[label]
    options = filter_options(column, order)
    if f"filter_{column}" not in st.session_state:
        st.session_state[f"filter_{column}"] = options
    return st.sidebar.multiselect(label, options=options, key=f"filter_{column}")


sel_gender = sidebar_filter("Gender")
sel_course = sidebar_filter("Course type")
sel_income = sidebar_filter("Family income")
sel_mode = sidebar_filter("Study mode")
st.sidebar.button("↺ Reset filters", on_click=reset_filters, width="stretch")

fdf = df[
    df["gender"].isin(sel_gender)
    & df["course_type"].isin(sel_course)
    & df["family_income_level"].isin(sel_income)
    & df["study_mode"].isin(sel_mode)
]
st.sidebar.markdown(f'<div class="side-count">👥 <b>{len(fdf):,}</b> of {len(df):,} students selected</div>',
                    unsafe_allow_html=True)


# ===============================================================
# BANNER: who it's for + the question
# ===============================================================
st.markdown(f"""
<div class="hero">
  <div class="title">🎓 Student Dropout Risk</div>
  <div class="sub"><b>For the Head of Student Support:</b> which students are most likely to drop out,
  and where should support go first?</div>
  <span class="chip">👥 {len(fdf):,} of {len(df):,} students selected</span>
  <span class="chip">🧪 Data appears synthetic, see About</span>
</div>
""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("No students match these filters. Tick at least one option in every filter, "
               "or press **Reset filters** in the side menu.")
    st.stop()

avg_rate = dropout_rate(fdf)
all_rate = dropout_rate(df)


# ===============================================================
# CALCULATIONS USED IN SEVERAL PLACES
# ===============================================================
gpa_bins = [-0.01, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
gpa_names = ["0–1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0", "3.0–3.5", "3.5–4.0"]
att_bins = [-0.01, 50, 60, 70, 80, 90, 100]
att_names = ["0–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
grid = fdf.assign(gpa_band=pd.cut(fdf["gpa_semester_1"], gpa_bins, labels=gpa_names),
                  att_band=pd.cut(fdf["attendance_rate"], att_bins, labels=att_names))
gpa_table = rate_table(grid, "gpa_band")
low_gpa, high_gpa = rate_of(gpa_table, "0–1.5"), rate_of(gpa_table, "3.5–4.0")

fails = fdf.assign(g=fdf["units_failed"].clip(upper=4).astype(int).astype(str).replace("4", "4+"))
fail_table = rate_table(fails, "g", order=["0", "1", "2", "3", "4+"])
deadlines = fdf.assign(g=fdf["missed_deadlines"].clip(upper=5).astype(int).astype(str).replace("5", "5+"))
dl_table = rate_table(deadlines, "g", order=["0", "1", "2", "3", "4", "5+"])

yes_no = {0: "No", 1: "Yes"}
inc_table = rate_table(fdf, "family_income_level", order=["Low", "Middle", "High"])
tui_table = rate_table(fdf, "tuition_up_to_date", names={0: "Behind", 1: "Up to date"})
job_table = rate_table(fdf, "part_time_job", names=yes_no)
fg_table = rate_table(fdf, "first_generation", names=yes_no)

LABELS = {
    "age": "Age", "first_generation": "First-generation", "disability_status": "Disability",
    "scholarship_holder": "Scholarship", "tuition_up_to_date": "Tuition up to date",
    "part_time_job": "Part-time job", "weekly_work_hours": "Weekly work hours", "semester": "Semester",
    "international_student": "International", "admission_grade": "Admission grade",
    "previous_qualification_grade": "Previous grade", "units_enrolled": "Units enrolled",
    "units_passed": "Units passed", "units_failed": "Units failed", "gpa_semester_1": "GPA semester 1",
    "gpa_semester_2": "GPA semester 2", "attendance_rate": "Attendance",
    "assignments_submitted_rate": "Assignments submitted", "library_visits_per_month": "Library visits",
    "lms_logins_per_week": "Online logins", "extracurricular_activities": "Extracurriculars",
    "missed_deadlines": "Missed deadlines", "stress_level": "Stress level",
    "mental_health_support": "Mental health support", "satisfaction_score": "Satisfaction",
    "commute_time_mins": "Commute time", "sleep_hours": "Sleep hours",
}
corr = fdf[list(LABELS)].corrwith(fdf["dropout"]).dropna().rename(index=LABELS).sort_values()


# ===============================================================
# TOP: KPI CARDS + 3 KEY INSIGHTS
# ===============================================================
def kpi(label, value, note="", red=False):
    return (f'<div class="kpi {"red" if red else ""}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>')


diff = avg_rate - all_rate
low_inc, high_inc = rate_of(inc_table, "Low"), rate_of(inc_table, "High")
k1, k2, k3 = st.columns([1, 1, 1.6])
k1.markdown(kpi("Students in selection", f"{len(fdf):,}",
                "all students" if len(fdf) == len(df) else f"of {len(df):,} students"), unsafe_allow_html=True)
k2.markdown(kpi("Dropout rate", f"{avg_rate:.1f}%",
                f"{int(fdf['dropout'].sum()):,} students dropped out"
                + ("" if abs(diff) < 0.05 else f" · {diff:+.1f} pts vs all students"), red=True),
            unsafe_allow_html=True)
k3.markdown(f'<div class="finding"><div class="label">Key finding: academic performance</div>'
            f'<div class="main">GPA ≤ 1.5: <b>{pct(low_gpa)}</b> dropout rate · GPA &gt; 3.5: {pct(high_gpa)}</div>'
            f'<div class="note" style="font-size:.84rem;opacity:.72;margin-top:4px">'
            f'Details in the Academic tab</div></div>', unsafe_allow_html=True)
st.write("")


# ===============================================================
# TABS (in story order)
# ===============================================================
tabs = st.tabs(["📚 Academic", "💷 Money & work", "⚖️ Little difference",
                "✅ Recommendations", "🧪 Risk estimate", "🔎 Explore", "ℹ️ About"])
tab_academic, tab_money, tab_nodiff, tab_action, tab_risk, tab_explore, tab_about = tabs


# ---------------------------------------------------------------
# TAB: ACADEMIC WARNING SIGNS
# ---------------------------------------------------------------
with tab_academic:
    cell_rate = grid.pivot_table(index="gpa_band", columns="att_band", values="dropout",
                                 aggfunc="mean", observed=False) * 100
    cell_size = grid.pivot_table(index="gpa_band", columns="att_band", values="dropout",
                                 aggfunc="size", observed=False)
    cell_rate = cell_rate.where(cell_size >= 30).iloc[::-1]   # hide tiny cells, highest GPA on top

    g1, g2, g3 = st.columns(3)
    stayed = fdf[fdf["dropout"] == 0]
    left_ = fdf[fdf["dropout"] == 1]
    g1.markdown(kpi("Average GPA (semester 1)", f"{fdf['gpa_semester_1'].mean():.2f}", "all selected students",
                    ) .replace('class="kpi ', 'class="kpi small '), unsafe_allow_html=True)
    g2.markdown(kpi("Average GPA: dropped out", f"{left_['gpa_semester_1'].mean():.2f}" if len(left_) else "n/a",
                    "students who left").replace('class="kpi ', 'class="kpi small '), unsafe_allow_html=True)
    g3.markdown(kpi("Average GPA: stayed", f"{stayed['gpa_semester_1'].mean():.2f}" if len(stayed) else "n/a",
                    "retained or graduated").replace('class="kpi ', 'class="kpi small '), unsafe_allow_html=True)
    st.write("")

    # how much attendance matters in the lowest vs highest GPA band (first vs last column)
    def band_gap(row):
        vals = cell_rate.loc[row].dropna() if row in cell_rate.index else pd.Series(dtype=float)
        return (vals.iloc[0], vals.iloc[-1]) if len(vals) > 1 else (None, None)

    lo_a, lo_b = band_gap("0–1.5")
    hi_a, hi_b = band_gap("3.5–4.0")
    att_note = ""
    if lo_a is not None and hi_a is not None:
        att_note = (f" Attendance makes a bigger difference at low GPA ({lo_a:.0f}% → {lo_b:.0f}% from lowest to "
                    f"highest attendance) than at high GPA ({hi_a:.0f}% → {hi_b:.0f}%).")

    chart_heading("Dropout rate by GPA and attendance",
                  f"GPA ≤ 1.5: {pct(low_gpa)} · GPA > 3.5: {pct(high_gpa)}. Darker cells indicate higher observed "
                  f"dropout rates. Hover to see how many students are in each cell; cells with fewer than 30 "
                  f"students are left blank.{att_note}")
    heat_sizes = cell_size.iloc[::-1]
    fig_heat = go.Figure(go.Heatmap(
        z=cell_rate.values, x=list(cell_rate.columns.astype(str)), y=list(cell_rate.index.astype(str)),
        customdata=heat_sizes.values, zmin=0, zmax=100, colorscale=RISK_SCALE, xgap=3, ygap=3,
        text=[[("" if pd.isna(v) else f"{v:.0f}") for v in row] for row in cell_rate.values],
        texttemplate="%{text}",
        colorbar=dict(title=dict(text="Dropout rate (%)", side="right"), ticksuffix="%", thickness=12),
        hovertemplate="GPA %{y} · attendance %{x}<br>Dropout rate: %{z:.1f}%<br>"
                      "Students: %{customdata:,}<extra></extra>"))
    tidy(fig_heat, 440, y_title="GPA in semester 1")
    fig_heat.update_xaxes(title="Attendance rate")
    fig_heat.update_yaxes(autorange="reversed")   # highest GPA at the top
    show(fig_heat)

    c1, c2 = st.columns(2)
    with c1:
        f0, f4 = rate_of(fail_table, "0"), rate_of(fail_table, "4+")
        chart_heading("Dropout rate by number of failed units",
                      f"{pct(f0, 1)} with none vs {pct(f4, 1)} with 4 or more"
                      + (f" (+{f4 - f0:.1f} percentage points)" if f0 is not None and f4 is not None else ""))
        show(line_chart(fail_table, "Units failed"))
    with c2:
        d0, d3 = rate_of(dl_table, "0"), rate_of(dl_table, "3")
        chart_heading("Dropout rate by number of missed deadlines",
                      f"{pct(d0, 1)} with none vs {pct(d3, 1)} with three"
                      + (f" (+{d3 - d0:.1f} percentage points)" if d0 is not None and d3 is not None else ""))
        show(line_chart(dl_table, "Missed deadlines"))

    with st.expander("📋 Every factor ranked by correlation with dropout"):
        corr_df = corr.reset_index()
        corr_df.columns = ["Factor", "r"]
        colours = [HIGHLIGHT if v > 0 else BASE for v in corr_df["r"]]
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Bar(x=corr_df["r"], y=corr_df["Factor"], orientation="h", width=0.12,
                                  marker_color=colours, hoverinfo="skip"))
        fig_corr.add_trace(go.Scatter(x=corr_df["r"], y=corr_df["Factor"], mode="markers+text",
                                      marker=dict(size=11, color=colours),
                                      text=corr_df["r"].map("{:+.2f}".format),
                                      textposition=["middle left" if v < 0 else "middle right" for v in corr_df["r"]],
                                      hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        tidy(fig_corr, 720, y_title="")
        fig_corr.update_xaxes(title="Correlation with dropout (−1 to +1)", range=[-0.75, 0.65],
                              showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=True)
        chart_heading("Correlation of each factor with dropout",
                      f"Strongest: {corr.idxmax()} ({corr.max():+.2f}) and {corr.idxmin()} ({corr.min():+.2f}). "
                      "Red = higher values go with more dropout; blue = less dropout; near 0 = little "
                      "association. Correlation is not cause.")
        show(fig_corr)


# ---------------------------------------------------------------
# TAB: MONEY & WORK
# ---------------------------------------------------------------
with tab_money:
    m1, m2 = st.columns([1, 1.25])
    with m1:
        chart_heading("Dropout rate by family income",
                      f"Low {pct(low_inc)} · High {pct(high_inc)}"
                      + (f" · difference {low_inc - high_inc:.0f} percentage points" if low_inc and high_inc else ""))
        show(bar_chart(inc_table, "Family income", height=360))

    with m2:
        chart_heading("Dropout rate with and without each financial or work pressure",
                      "Observed associations in this dataset, not proof that these factors cause dropout.")
        pairs = [
            ("Behind on tuition", rate_of(tui_table, "Behind"), rate_of(tui_table, "Up to date")),
            ("Has a part-time job", rate_of(job_table, "Yes"), rate_of(job_table, "No")),
            ("First-generation student", rate_of(fg_table, "Yes"), rate_of(fg_table, "No")),
        ]
        pairs = [p for p in pairs if p[1] is not None and p[2] is not None]
        fig_db = go.Figure()
        for name, with_it, without in pairs:
            fig_db.add_trace(go.Scatter(x=[without, with_it], y=[name, name], mode="lines",
                                        line=dict(color="rgba(128,128,128,.35)", width=6), hoverinfo="skip",
                                        showlegend=False))
        fig_db.add_trace(go.Scatter(x=[p[2] for p in pairs], y=[p[0] for p in pairs], mode="markers+text",
                                    name="Without", marker=dict(size=18, color=BASE),
                                    text=[f"{p[2]:.0f}%" for p in pairs], textposition="bottom center",
                                    hovertemplate="Without: %{x:.1f}%<extra></extra>"))
        fig_db.add_trace(go.Scatter(x=[p[1] for p in pairs], y=[p[0] for p in pairs], mode="markers+text",
                                    name="With", marker=dict(size=18, color=HIGHLIGHT),
                                    text=[f"{p[1]:.0f}%" for p in pairs], textposition="bottom center",
                                    hovertemplate="With: %{x:.1f}%<extra></extra>"))
        fig_db.add_vline(x=avg_rate, line_dash="dot", line_color="#888")
        tidy(fig_db, 360, y_title="")
        fig_db.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
        fig_db.update_xaxes(title="Dropout rate (%)", range=[15, 52], showgrid=True,
                            gridcolor="rgba(128,128,128,.15)")
        fig_db.update_yaxes(autorange="reversed")
        show(fig_db)


# ---------------------------------------------------------------
# TAB: WHAT DOESN'T MATTER
# ---------------------------------------------------------------
with tab_nodiff:
    commute = fdf.assign(g=pd.cut(fdf["commute_time_mins"], [-1, 10, 20, 30, 60, 120]))
    ages = fdf.assign(g=pd.cut(fdf["age"], [14, 18, 21, 24, 27, 30]))
    checks = {
        "International vs domestic": rate_table(fdf, "international_student"),
        "Commute time": rate_table(commute, "g"),
        "Age group": rate_table(ages, "g"),
        "Course type": rate_table(fdf, "course_type"),
        "Nationality": rate_table(fdf, "nationality"),
        "Gender": rate_table(fdf, "gender"),
    }
    compare = {"Family income (for comparison)": inc_table, "Units failed (for comparison)": fail_table}

    rows = [(name, t["rate"].min(), t["rate"].max(), False) for name, t in checks.items() if len(t) > 1]
    rows += [(name, t["rate"].min(), t["rate"].max(), True) for name, t in compare.items() if len(t) > 1]
    biggest_flat = max((r[2] - r[1] for r in rows if not r[3]), default=0)

    chart_heading("Dropout rates across selected background groups",
                  f"Each bar spans the lowest to highest group dropout rate. Grey groups differ by at most "
                  f"{biggest_flat:.1f} percentage points; family income and failed units are shown in red for comparison.")
    fig_range = go.Figure()
    for name, lo, hi, is_ref in rows:
        colour = HIGHLIGHT if is_ref else NEUTRAL
        fig_range.add_trace(go.Scatter(
            x=[lo, hi], y=[name, name], mode="lines+markers",
            line=dict(color=colour, width=10), marker=dict(size=14, color=colour),
            hovertemplate=f"{name}<br>Lowest group: {lo:.1f}%<br>Highest group: {hi:.1f}%<extra></extra>"))
        fig_range.add_annotation(x=hi, y=name, text=f"  {hi - lo:.1f} pts", showarrow=False,
                                 xanchor="left", font=dict(size=12))
    fig_range.add_vline(x=avg_rate, line_dash="dot", line_color="#888",
                        annotation_text=f"Average {avg_rate:.0f}%", annotation_position="top")
    tidy(fig_range, 420, y_title="")
    fig_range.update_xaxes(title="Dropout rate (%)", range=[0, 75], showgrid=True,
                           gridcolor="rgba(128,128,128,.15)")
    fig_range.update_yaxes(autorange="reversed")
    show(fig_range)
    st.info("**No substantial difference observed in this dataset** for international status, commute time, "
            "age, course type, nationality or gender. This does not prove these factors are irrelevant at every "
            "university, but in this data they do not separate students at higher risk.")


# ---------------------------------------------------------------
# TAB: WHAT TO DO
# ---------------------------------------------------------------
with tab_action:
    fail_any = dropout_rate(fdf[fdf["units_failed"] >= 1]) if (fdf["units_failed"] >= 1).any() else None
    recs = [
        ("Academic support alert",
         f"GPA ≤ 1.5 after semester 1: <b>{pct(low_gpa, 1)}</b> dropout rate vs <b>{pct(high_gpa, 1)}</b> above 3.5",
         "Offer an academic advising appointment early in semester 2.",
         "Track semester-2 GPA and dropout for advised students against a comparable group.",
         "Academic tab: GPA and attendance heatmap"),
        ("Advising after a failed unit",
         f"Any failed unit: <b>{pct(fail_any, 1)}</b> vs <b>{pct(rate_of(fail_table, '0'), 1)}</b> with none",
         "Invite students to a study-skills or resit-planning session after their first failed unit.",
         "Compare pass rates and dropout with students who were not invited, or invited later.",
         "Academic tab: failed units chart"),
        ("Check-ins after missed deadlines",
         f"None: <b>{pct(rate_of(dl_table, '0'), 1)}</b> · three: <b>{pct(rate_of(dl_table, '3'), 1)}</b> dropout rate",
         "Send a tutor check-in after a missed deadline, before a pattern builds up.",
         "Measure later missed deadlines and dropout before and after introducing check-ins.",
         "Academic tab: missed deadlines chart"),
        ("Tuition payment support",
         f"Behind on tuition: <b>{pct(rate_of(tui_table, 'Behind'), 1)}</b> vs "
         f"<b>{pct(rate_of(tui_table, 'Up to date'), 1)}</b> up to date",
         "Offer payment plans or hardship-fund advice when a payment becomes overdue.",
         "Compare dropout for students who take up a plan with similar students who do not.",
         "Money & work tab"),
        ("Support for working and first-generation students",
         f"Part-time job <b>{pct(rate_of(job_table, 'Yes'), 1)}</b> vs <b>{pct(rate_of(job_table, 'No'), 1)}</b> · "
         f"first-generation <b>{pct(rate_of(fg_table, 'Yes'), 1)}</b> vs <b>{pct(rate_of(fg_table, 'No'), 1)}</b>",
         "Pilot flexible timetabling for working students and peer mentoring for first-generation students.",
         "Run as a pilot in one department and compare dropout with a similar department.",
         "Money & work tab"),
    ]
    r1, r2 = st.columns(2)
    for n, (head, signal, action, evaluation, see) in enumerate(recs, start=1):
        (r1 if n % 2 else r2).markdown(
            f'<div class="rec"><span class="num">{n}</span><span class="head">{head}</span>'
            f'<div class="action"><b>Signal:</b> {signal}</div>'
            f'<div class="action"><b>Possible action:</b> {action}</div>'
            f'<div class="action"><b>Evaluation:</b> {evaluation}</div>'
            f'<div class="see">Evidence: {see}</div></div>', unsafe_allow_html=True)
    st.caption("These are proposals to test, not proven solutions: the dataset shows which groups have higher "
               "dropout rates, but it cannot show which interventions will work.")


# ---------------------------------------------------------------
# TAB: RISK CHECKER (a simple prediction model)
# ---------------------------------------------------------------
MODEL_FEATURES = {
    "gpa_semester_1": "GPA semester 1", "attendance_rate": "Attendance", "units_failed": "Units failed",
    "missed_deadlines": "Missed deadlines", "assignments_submitted_rate": "Assignments submitted",
    "weekly_work_hours": "Weekly work hours", "stress_level": "Stress level",
    "tuition_up_to_date": "Tuition up to date", "part_time_job": "Part-time job",
    "first_generation": "First-generation", "income_low": "Low income", "income_middle": "Middle income",
}


def model_inputs(data):
    """Turn student records into the numbers the model uses."""
    X = data[[c for c in MODEL_FEATURES if not c.startswith("income_")]].copy()
    X["income_low"] = (data["family_income_level"] == "Low").astype(int)
    X["income_middle"] = (data["family_income_level"] == "Middle").astype(int)
    return X


@st.cache_resource
def train_model(data):
    """Logistic regression: learns how each factor changes the chance of dropping out."""
    X, y = model_inputs(data), data["dropout"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    auc = roc_auc_score(y_test, model.predict_proba(scaler.transform(X_test))[:, 1])
    return scaler, model, auc


with tab_risk:
    scaler, model, auc = train_model(df)
    st.markdown('<div class="notice">🧪 <b>Illustrative risk estimate (demonstration only).</b> '
                'The model is trained on a dataset that appears synthetic. The score is not a decision about an '
                'individual student, and a lower score is not a reason to withhold support.</div>',
                unsafe_allow_html=True)
    st.markdown("Move the sliders to see how the estimate changes for a hypothetical student.")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        a, b = st.columns(2)
        gpa = a.slider("GPA (semester 1)", 0.0, 4.0, 2.5, 0.1)
        attendance = b.slider("Attendance (%)", 0, 100, 75)
        failed = a.slider("Units failed", 0, 10, 0)
        missed = b.slider("Missed deadlines", 0, 10, 1)
        assignments = a.slider("Assignments submitted (%)", 0, 100, 80)
        work = b.slider("Weekly work hours", 0, 40, 0)
        stress = a.slider("Stress level (1–10)", 1, 10, 5)
        income = b.selectbox("Family income", ["Low", "Middle", "High"], index=1)
        t1, t2, t3 = st.columns(3)
        tuition = t1.toggle("Tuition paid", value=True)
        job = t2.toggle("Part-time job")
        first_gen = t3.toggle("First-gen")

    student = pd.DataFrame([{
        "gpa_semester_1": gpa, "attendance_rate": attendance, "units_failed": failed,
        "missed_deadlines": missed, "assignments_submitted_rate": assignments, "weekly_work_hours": work,
        "stress_level": stress, "tuition_up_to_date": int(tuition), "part_time_job": int(job),
        "first_generation": int(first_gen), "family_income_level": income,
    }])
    x = scaler.transform(model_inputs(student))
    risk = model.predict_proba(x)[0, 1] * 100

    if risk >= 60:
        level, colour = "Higher estimated risk", HIGHLIGHT
    elif risk >= 30:
        level, colour = "Moderate estimated risk", "#E09F3E"
    else:
        level, colour = "Lower estimated risk", BASE

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge", value=risk, domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=f"<b style='color:{colour}'>{level}</b><br>"
                            f"<span style='font-size:13px'>Dataset average: {all_rate:.0f}% (dark marker)</span>",
                       font=dict(family=FONT, size=20)),
            gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=colour, thickness=0.3),
                       steps=[dict(range=[0, 30], color="rgba(124,147,176,.15)"),
                              dict(range=[30, 60], color="rgba(224,159,62,.18)"),
                              dict(range=[60, 100], color="rgba(209,73,91,.18)")],
                       threshold=dict(line=dict(color="#555", width=3), value=all_rate))))
        gauge.add_annotation(x=0.5, y=0.05, xref="paper", yref="paper", showarrow=False, xanchor="center",
                             text=f"<b>{risk:.0f}%</b>", font=dict(size=50, family=FONT, color=colour))
        gauge.update_layout(height=280, margin=dict(t=70, b=10, l=40, r=45),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT))
        show(gauge)

        # What pushes this student's risk up or down, compared with an average student
        contrib = pd.Series(model.coef_[0] * x[0], index=list(MODEL_FEATURES.values()))
        contrib["Family income"] = contrib.pop("Low income") + contrib.pop("Middle income")
        contrib = contrib[contrib.abs() > 0.05].sort_values(key=abs).tail(6).sort_values()
        if len(contrib):
            chart_heading("What's driving this estimate", "Compared with an average student")
            fig_c = go.Figure(go.Bar(
                x=contrib.values, y=contrib.index, orientation="h", marker_cornerradius=4,
                marker_color=[HIGHLIGHT if v > 0 else BASE for v in contrib.values],
                text=["raises risk" if v > 0 else "lowers risk" for v in contrib.values],
                textposition="auto", hoverinfo="skip"))
            tidy(fig_c, 240, y_title="")
            fig_c.update_xaxes(showticklabels=False, title="", zeroline=True, zerolinecolor="#888")
            show(fig_c)

    with st.expander("⚙️ Technical details"):
        st.markdown(
            f"- **Model:** logistic regression on {len(MODEL_FEATURES)} inputs, trained on 80% of the "
            f"{len(df):,} students and tested on the other 20%.\n"
            f"- **Test result:** AUC {auc:.2f}. A student who dropped out gets a higher score than one who stayed "
            f"{auc * 100:.0f}% of the time.\n"
            "- **Limits:** performance on synthetic data does not establish real-world predictive value. A real "
            "model would need validation on real student records, fairness checks and governance before use.\n"
            "- **Driver chart:** shows each input's contribution relative to an average student in the dataset.")


# ---------------------------------------------------------------
# TAB: EXPLORE ANY GROUP
# ---------------------------------------------------------------
with tab_explore:
    groups = {
        "Family income": ("family_income_level", ["Low", "Middle", "High"], None),
        "Tuition up to date": ("tuition_up_to_date", ["No", "Yes"], yes_no),
        "Part-time job": ("part_time_job", ["Yes", "No"], yes_no),
        "First-generation student": ("first_generation", ["Yes", "No"], yes_no),
        "Scholarship holder": ("scholarship_holder", ["Yes", "No"], yes_no),
        "Disability": ("disability_status", ["Yes", "No"], yes_no),
        "Uses mental health support": ("mental_health_support", ["Yes", "No"], yes_no),
        "Extracurricular activities": ("extracurricular_activities", ["Yes", "No"], yes_no),
        "International student": ("international_student", ["Yes", "No"], yes_no),
        "Study mode": ("study_mode", None, None),
        "Course type": ("course_type", None, None),
        "Gender": ("gender", None, None),
        "Semester": ("semester", None, None),
    }
    e1, e2 = st.columns([1, 2])
    choice = e1.selectbox("Compare dropout rate by:", list(groups))
    column, order, names = groups[choice]
    t = rate_table(fdf, column, order=order, names=names)
    gap = t["rate"].max() - t["rate"].min()
    e2.markdown(kpi("Gap between groups", f"{gap:.1f} percentage points",
                    "no substantial difference" if gap < 3 else "a substantial difference", red=gap >= 3),
                unsafe_allow_html=True)
    chart_heading(f"Dropout rate by {choice.lower()}",
                  f"Range {t['rate'].min():.1f}% to {t['rate'].max():.1f}% "
                  f"({'no substantial difference observed' if gap < 3 else f'{gap:.1f} percentage points apart'})")
    show(bar_chart(t, choice, highlight=gap >= 3))

    q75 = fdf.loc[fdf["dropout"] == 1, "gpa_semester_1"].quantile(0.75)
    q25 = fdf.loc[fdf["dropout"] == 0, "gpa_semester_1"].quantile(0.25)
    chart_heading("GPA distribution: students who dropped out vs stayed",
                  f"Three-quarters of dropouts had a GPA below {q75:.1f}; three-quarters of those who stayed were "
                  f"above {q25:.1f}. The box covers the middle half of students; the line is the median.")
    status = fdf.assign(Status=fdf["dropout"].map({0: "Retained / graduated", 1: "Dropped out"}))
    fig_box = px.box(status, x="gpa_semester_1", y="Status", color="Status", orientation="h",
                     color_discrete_map={"Dropped out": HIGHLIGHT, "Retained / graduated": BASE},
                     category_orders={"Status": ["Dropped out", "Retained / graduated"]})
    tidy(fig_box, 260, y_title="")
    fig_box.update_xaxes(title="GPA in semester 1", showgrid=True, gridcolor="rgba(128,128,128,.15)")
    show(fig_box)


# ---------------------------------------------------------------
# TAB: ABOUT & DATA
# ---------------------------------------------------------------
with tab_about:
    st.markdown(
        f"#### About this data\n"
        f"- **Source:** {DATA_SOURCE}, downloaded as published. Our team did not generate or alter the records.\n"
        f"- **Size:** {len(df):,} students · {df.shape[1]} columns · {df['nationality'].nunique()} nationalities.\n"
        "- **Dropout variable:** the `dropout` column, where 1 = the student dropped out and 0 = retained or "
        "graduated.\n"
        "- **Dropout rate:** the number of students in a group with dropout = 1, divided by all students in that "
        "group, × 100.\n"
        "- **Synthetic data:** the Kaggle page does not say whether the data is real. Its patterns (exactly "
        "100,000 rows; almost identical rates across nationality, age and course) suggest it is synthetic.\n"
        "- **Correlation is not causation:** charts show associations, not causes.\n"
        "- **Generalisation:** the findings demonstrate the method and should not be applied to a real university "
        "without validation on its own data."
    )
    st.markdown(f"#### Selected student records ({len(fdf):,})")
    st.dataframe(fdf.head(100), width="stretch", hide_index=True)
    st.download_button("⬇️ Download selected students (CSV)", data=fdf.to_csv(index=False).encode("utf-8"),
                       file_name="selected_students.csv", mime="text/csv")
