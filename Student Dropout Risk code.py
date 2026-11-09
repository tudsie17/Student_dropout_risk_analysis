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

DATA_SOURCE = "https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk"

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

.insight { border-radius: 16px; padding: 16px 18px; height: 100%;
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


def sidebar_filter(label, column, order=None):
    options = order if order else sorted(df[column].unique())
    return st.sidebar.multiselect(label, options=options, default=options)


sel_gender = sidebar_filter("Gender", "gender")
sel_course = sidebar_filter("Course type", "course_type")
sel_income = sidebar_filter("Family income", "family_income_level", order=["Low", "Middle", "High"])
sel_mode = sidebar_filter("Study mode", "study_mode")

fdf = df[
    df["gender"].isin(sel_gender)
    & df["course_type"].isin(sel_course)
    & df["family_income_level"].isin(sel_income)
    & df["study_mode"].isin(sel_mode)
]
st.sidebar.divider()
st.sidebar.metric("Students selected", f"{len(fdf):,}", help=f"Out of {len(df):,} students")


# ===============================================================
# BANNER: who it's for + the question
# ===============================================================
st.markdown(f"""
<div class="hero">
  <div class="title">🎓 Student Dropout Risk</div>
  <div class="sub"><b>For the Head of Student Support:</b> which students are most likely to drop out,
  and where should support go first?</div>
  <span class="chip">👥 {len(fdf):,} of {len(df):,} students selected</span>
  <span class="chip">🧪 Synthetic dataset, see the About tab</span>
  <span class="chip">🎯 Try the risk checker tab</span>
</div>
""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("No students match these filters. Tick at least one option in every filter.")
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

fail_names = ["0", "1", "2", "3", "4–5", "6–8", "9+"]   # top groups split so the steep end stays visible
fails = fdf.assign(g=pd.cut(fdf["units_failed"], [-1, 0, 1, 2, 3, 5, 8, 99], labels=fail_names))
fail_table = rate_table(fails, "g", order=fail_names)
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
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Students", f"{len(fdf):,}", "in the current selection"), unsafe_allow_html=True)
k2.markdown(kpi("Dropped out", f"{int(fdf['dropout'].sum()):,}", f"about 1 in {round(100 / avg_rate)}"),
            unsafe_allow_html=True)
k3.markdown(kpi("Dropout rate", f"{avg_rate:.1f}%",
                "same as all students" if abs(diff) < 0.05 else f"{diff:+.1f} pts vs all students", red=True),
            unsafe_allow_html=True)
k4.markdown(kpi("Average GPA", f"{fdf['gpa_semester_1'].mean():.2f}",
                f"attendance {fdf['attendance_rate'].mean():.0f}%"), unsafe_allow_html=True)

st.write("")
low_inc, high_inc = rate_of(inc_table, "Low"), rate_of(inc_table, "High")
i1, i2, i3 = st.columns(3)
i1.markdown(f'<div class="insight"><div class="big">{pct(low_gpa)}</div><div class="text">'
            f'of students with a GPA of 1.5 or below drop out, against {pct(high_gpa)} above 3.5.</div></div>',
            unsafe_allow_html=True)
i2.markdown(f'<div class="insight"><div class="big">{pct(rate_of(fail_table, "9+"))}</div><div class="text">'
            f'drop out after failing 9 or more units, against {pct(rate_of(fail_table, "0"))} with none.</div></div>',
            unsafe_allow_html=True)
if low_inc and high_inc:
    i3.markdown(f'<div class="insight"><div class="big">{low_inc / high_inc:.1f}×</div><div class="text">'
                f'Low-income students drop out this much more often than high-income students '
                f'({pct(low_inc)} vs {pct(high_inc)}).</div></div>', unsafe_allow_html=True)
else:
    i3.markdown(f'<div class="insight"><div class="big">{pct(rate_of(tui_table, "Behind"))}</div>'
                f'<div class="text">of students behind on tuition drop out, against '
                f'{pct(rate_of(tui_table, "Up to date"))} who are up to date.</div></div>', unsafe_allow_html=True)
st.write("")


# ===============================================================
# TABS (in story order)
# ===============================================================
tabs = st.tabs(["📚 Academic warning signs", "💷 Money & work", "⚖️ What doesn't matter",
                "✅ What to do", "🎯 Risk checker", "🔎 Explore", "ℹ️ About & data"])
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

    chart_heading(f"Low grades are the clearest warning sign: {pct(low_gpa)} drop out below a GPA of 1.5",
                  "Dropout rate (%) for each mix of GPA and attendance. Darker = higher risk.")
    fig_heat = px.imshow(cell_rate, text_auto=".0f", aspect="auto", zmin=0, zmax=100,
                         color_continuous_scale=RISK_SCALE,
                         labels=dict(x="Attendance rate", y="GPA in semester 1", color="Dropout %"))
    fig_heat.update_traces(hovertemplate="GPA %{y} · attendance %{x}<br>Dropout rate: %{z:.1f}%<extra></extra>",
                           xgap=3, ygap=3)
    tidy(fig_heat, 440, y_title="GPA in semester 1")
    fig_heat.update_layout(coloraxis_colorbar=dict(title="", ticksuffix="%", thickness=12))
    show(fig_heat)

    c1, c2 = st.columns(2)
    with c1:
        chart_heading(f"Risk climbs with every failed unit: {pct(rate_of(fail_table, '0'))} → "
                      f"{pct(rate_of(fail_table, '9+'))} with 9 or more")
        show(line_chart(fail_table, "Units failed"))
    with c2:
        chart_heading(f"Dropout rate rises with missed deadlines: {pct(rate_of(dl_table, '0'), 1)} with none → "
                      f"{pct(rate_of(dl_table, '3'), 1)} with three")
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
        fig_corr.update_xaxes(title="Correlation with dropout (−1 to +1)", range=[min(-0.1, corr.min() - 0.2), max(0.1, corr.max() + 0.2)],
                              showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=True)
        chart_heading(f"{corr.idxmax()} ({corr.max():+.2f}) and {corr.idxmin()} ({corr.min():+.2f}) "
                      "are most strongly linked to dropout",
                      "Red = more of this goes with more dropout. Blue = less dropout. Near 0 = no link. "
                      "Correlation is not cause.")
        show(fig_corr)


# ---------------------------------------------------------------
# TAB: MONEY & WORK
# ---------------------------------------------------------------
with tab_money:
    m1, m2 = st.columns([1, 1.25])
    with m1:
        chart_heading(f"Low-income students drop out most ({pct(low_inc)} vs {pct(high_inc)} high-income)")
        show(bar_chart(inc_table, "Family income", height=360))

    with m2:
        chart_heading("Late tuition, a part-time job and being first-generation all raise risk",
                      "Dropout rate with vs without each pressure")
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
        db_values = [v for p in pairs for v in p[1:]]
        db_range = [max(0, min(db_values) - 8), min(100, max(db_values) + 8)] if db_values else [0, 100]
        fig_db.update_xaxes(title="Dropout rate (%)", range=db_range, showgrid=True,
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

    chart_heading(f"Background factors barely move dropout: every group is within {biggest_flat:.0f} points",
                  "Each bar spans the lowest to highest group dropout rate. Short grey bar = little observed difference.")
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
    fig_range.update_xaxes(title="Dropout rate (%)",
                           range=[0, min(100, max([r[2] for r in rows] + [avg_rate]) + 12)], showgrid=True,
                           gridcolor="rgba(128,128,128,.15)")
    fig_range.update_yaxes(autorange="reversed")
    show(fig_range)
    st.info("**No substantial difference in observed dropout rates was found in this dataset** for international "
            "status, commute time, age, course type, nationality or gender. This does not prove these factors are "
            "irrelevant elsewhere, but here they do not help identify which students need support.")
    mode_table = rate_table(fdf, "study_mode")
    if len(mode_table) > 1:
        lo_m = mode_table.loc[mode_table["rate"].idxmin()]
        hi_m = mode_table.loc[mode_table["rate"].idxmax()]
        mode_gap = hi_m["rate"] - lo_m["rate"]
        inc_gap = inc_table["rate"].max() - inc_table["rate"].min() if len(inc_table) > 1 else None
        if mode_gap >= 3:
            st.markdown(
                f"**Study mode is the exception:** {lo_m['group']} students drop out at {lo_m['rate']:.1f}% and "
                f"{hi_m['group']} students at {hi_m['rate']:.1f}%, a {mode_gap:.1f}-point gap. That is above our "
                f"illustrative 3-point threshold but much smaller than the gaps for "
                + (f"family income ({inc_gap:.1f} points) and " if inc_gap is not None else "")
                + "failed units, so we report it without building a recommendation on it.")
        else:
            st.markdown(f"**Study mode** also shows little observed difference in this selection "
                        f"({lo_m['rate']:.1f}% to {hi_m['rate']:.1f}%).")


# ---------------------------------------------------------------
# TAB: WHAT TO DO
# ---------------------------------------------------------------
with tab_action:
    fail_any = dropout_rate(fdf[fdf["units_failed"] >= 1]) if (fdf["units_failed"] >= 1).any() else None
    intl = checks["International vs domestic"]
    recs = [
        ("Early academic alert after semester 1",
         "Contact students with a GPA of 1.5 or below, or any failed unit.",
         f"GPA ≤1.5: <b>{pct(low_gpa, 1)}</b> vs <b>{pct(high_gpa, 1)}</b> above 3.5 · "
         f"any failed unit: <b>{pct(fail_any, 1)}</b> vs <b>{pct(rate_of(fail_table, '0'), 1)}</b>",
         "Academic warning signs tab"),
        ("Check in after the first missed deadline",
         "Don't wait for a pattern: one missed deadline should trigger a tutor check-in.",
         f"None: <b>{pct(rate_of(dl_table, '0'), 1)}</b> · one: <b>{pct(rate_of(dl_table, '1'), 1)}</b> · "
         f"three: <b>{pct(rate_of(dl_table, '3'), 1)}</b>",
         "Academic warning signs tab"),
        ("Payment plans for students behind on tuition",
         "Treat late tuition as an early signal for financial support.",
         f"Behind: <b>{pct(rate_of(tui_table, 'Behind'), 1)}</b> vs up to date: "
         f"<b>{pct(rate_of(tui_table, 'Up to date'), 1)}</b>",
         "Money & work tab"),
        ("Target low-income, working and first-generation students",
         "Bursaries, flexible timetables and peer mentoring.",
         f"Low income <b>{pct(low_inc, 1)}</b> vs high <b>{pct(high_inc, 1)}</b> · job "
         f"<b>{pct(rate_of(job_table, 'Yes'), 1)}</b> vs <b>{pct(rate_of(job_table, 'No'), 1)}</b> · "
         f"first-gen <b>{pct(rate_of(fg_table, 'Yes'), 1)}</b> vs <b>{pct(rate_of(fg_table, 'No'), 1)}</b>",
         "Money & work tab"),
        ("Don't target by nationality, international status or commute",
         "In this dataset these groups have very similar dropout rates, so targeting them is unlikely to reach more at-risk students.",
         f"International <b>{pct(rate_of(intl, '1'), 1)}</b> vs domestic <b>{pct(rate_of(intl, '0'), 1)}</b>",
         "What doesn't matter tab"),
    ]
    r1, r2 = st.columns(2)
    for n, (head, action, evidence, see) in enumerate(recs, start=1):
        (r1 if n % 2 else r2).markdown(
            f'<div class="rec"><span class="num">{n}</span><span class="head">{head}</span>'
            f'<div class="action">{action}</div><div class="evidence">📊 {evidence}</div>'
            f'<div class="see">See: {see}</div></div>', unsafe_allow_html=True)
    st.caption("These groups are at higher risk, but the data cannot prove which actions will work. "
               "Pilot each action and measure dropout before and after.")


# ---------------------------------------------------------------
# TAB: RISK CHECKER (a simple prediction model, two versions)
# ---------------------------------------------------------------
# Plain names for every input either model can use
INPUT_NAMES = {
    "gpa_semester_1": "GPA semester 1", "admission_grade": "Admission grade",
    "previous_qualification_grade": "Previous qualification grade", "attendance_rate": "Attendance",
    "units_failed": "Units failed", "missed_deadlines": "Missed deadlines",
    "assignments_submitted_rate": "Assignments submitted", "weekly_work_hours": "Weekly work hours",
    "stress_level": "Stress level", "tuition_up_to_date": "Tuition up to date", "part_time_job": "Part-time job",
    "first_generation": "First-generation", "scholarship_holder": "Scholarship",
    "income_low": "Low income", "income_middle": "Middle income",
}
# Early alert: only information available by the end of semester 1 (or before entry)
EARLY_FEATURES = ["gpa_semester_1", "admission_grade", "previous_qualification_grade", "weekly_work_hours",
                  "part_time_job", "first_generation", "scholarship_holder", "income_low", "income_middle"]
# Full record: everything in the original version, including counts whose timing is not documented
FULL_FEATURES = ["gpa_semester_1", "attendance_rate", "units_failed", "missed_deadlines",
                 "assignments_submitted_rate", "weekly_work_hours", "stress_level", "tuition_up_to_date",
                 "part_time_job", "first_generation", "income_low", "income_middle"]
MODEL_VERSIONS = {
    "Early alert: semester-1 information only": EARLY_FEATURES,
    "Full record: all available information": FULL_FEATURES,
}

# When each input is measured (from the dataset itself; the Kaggle page does not document timing)
TIMING = pd.DataFrame([
    ("GPA semester 1", "End of semester 1", "✓", "✓"),
    ("Admission grade, previous qualification grade", "Before entry", "✓", ""),
    ("Family income, first-generation, scholarship", "At entry (background)", "✓", "Income, first-gen"),
    ("Part-time job, weekly work hours", "Not documented; treated as known early", "✓", "✓"),
    ("Units failed", "Not documented; may cover the whole period", "", "✓"),
    ("Missed deadlines", "Not documented; may cover the whole period", "", "✓"),
    ("Attendance, assignments submitted", "Not documented; may cover the whole period", "", "✓"),
    ("Stress level, tuition up to date", "Not documented", "", "✓"),
], columns=["Input", "When it is measured", "Early alert", "Full record"])


def model_inputs(data, features):
    """Turn student records into the numbers a model uses."""
    X = pd.DataFrame(index=data.index)
    for f in features:
        if f == "income_low":
            X[f] = (data["family_income_level"] == "Low").astype(int)
        elif f == "income_middle":
            X[f] = (data["family_income_level"] == "Middle").astype(int)
        else:
            X[f] = data[f]
    return X


@st.cache_resource
def train_model(data, version):
    """Logistic regression: learns how each input changes the chance of dropping out."""
    features = MODEL_VERSIONS[version]
    X, y = model_inputs(data, features), data["dropout"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    auc = roc_auc_score(y_test, model.predict_proba(scaler.transform(X_test))[:, 1])
    return scaler, model, auc


with tab_risk:
    st.markdown("Enter a student's details to estimate their dropout risk.")
    version = st.radio("Model version", list(MODEL_VERSIONS), horizontal=True,
                       help="The early-alert version matches the 'early academic alert after semester 1' "
                            "recommendation: it only uses information available by then.")
    features = MODEL_VERSIONS[version]
    scaler, model, auc = train_model(df, version)

    with st.expander("🕒 When is each input measured?"):
        st.dataframe(TIMING, hide_index=True, width="stretch")
        st.caption("The Kaggle page does not say when units failed, missed deadlines, attendance or assignments "
                   "are recorded, and their averages are the same for students in semester 1 and semester 8. "
                   "If they cover the whole period up to dropping out, they partly describe the outcome itself, "
                   "which can make the full-record model look better than it would be in practice. "
                   "The early-alert version avoids them.")

    early = features is EARLY_FEATURES
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        a, b = st.columns(2)
        values = {"gpa_semester_1": a.slider("GPA (semester 1)", 0.0, 4.0, 2.5, 0.1)}
        if early:
            values["admission_grade"] = b.slider("Admission grade", 0, 200,
                                                 int(df["admission_grade"].median()))
            values["previous_qualification_grade"] = a.slider("Previous qualification grade", 0, 200,
                                                              int(df["previous_qualification_grade"].median()))
            values["weekly_work_hours"] = b.slider("Weekly work hours", 0, 40, 0)
        else:
            values["attendance_rate"] = b.slider("Attendance (%)", 0, 100, 75)
            values["units_failed"] = a.slider("Units failed", 0, int(df["units_failed"].max()), 0)
            values["missed_deadlines"] = b.slider("Missed deadlines", 0, int(df["missed_deadlines"].max()), 1)
            values["assignments_submitted_rate"] = a.slider("Assignments submitted (%)", 0, 100, 80)
            values["weekly_work_hours"] = b.slider("Weekly work hours", 0, 40, 0)
            values["stress_level"] = a.slider("Stress level (1–10)", 1, 10, 5)
        values["family_income_level"] = b.selectbox("Family income", ["Low", "Middle", "High"],
                                                                     index=1)
        t1, t2, t3 = st.columns(3)
        if early:
            values["scholarship_holder"] = int(t1.toggle("Scholarship"))
        else:
            values["tuition_up_to_date"] = int(t1.toggle("Tuition paid", value=True))
        values["part_time_job"] = int(t2.toggle("Part-time job"))
        values["first_generation"] = int(t3.toggle("First-gen"))

    student = pd.DataFrame([values])
    x = scaler.transform(model_inputs(student, features))
    risk = model.predict_proba(x)[0, 1] * 100

    if risk >= 60:
        level, colour = "High risk", HIGHLIGHT
    elif risk >= 30:
        level, colour = "Medium risk", "#E09F3E"
    else:
        level, colour = "Low risk", "#3C8D6E"

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge", value=risk, domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=f"<b style='color:{colour}'>{level}</b><br>"
                            f"<span style='font-size:13px'>Average student: {all_rate:.0f}%</span>",
                       font=dict(family=FONT, size=20)),
            gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=colour, thickness=0.3),
                       steps=[dict(range=[0, 30], color="rgba(60,141,110,.15)"),
                              dict(range=[30, 60], color="rgba(224,159,62,.18)"),
                              dict(range=[60, 100], color="rgba(209,73,91,.18)")],
                       threshold=dict(line=dict(color="#555", width=3), value=all_rate))))
        gauge.add_annotation(x=0.5, y=0.05, xref="paper", yref="paper", showarrow=False, xanchor="center",
                             text=f"<b>{risk:.0f}%</b>", font=dict(size=50, family=FONT, color=colour))
        gauge.update_layout(height=280, margin=dict(t=70, b=10, l=30, r=30),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT))
        show(gauge)

        # What pushes this student's risk up or down, compared with an average student
        contrib = pd.Series(model.coef_[0] * x[0], index=[INPUT_NAMES[f] for f in features])
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

    other = [v for v in MODEL_VERSIONS if v != version][0]
    other_auc = train_model(df, other)[2]
    st.caption(f"Simple logistic regression trained on all {len(df):,} students. Tested on students it had not "
               f"seen, this version ranks a dropout above a non-dropout {auc * 100:.0f}% of the time "
               f"(AUC {auc:.2f}; the other version: {other_auc:.2f}). For a demonstration only: this is "
               "synthetic data, and a real model would need testing before being used on students.")


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
    e2.markdown(kpi("Gap between groups", f"{gap:.1f} pts",
                    "below the illustrative 3-point threshold" if gap < 3
                    else "at or above the illustrative 3-point threshold", red=gap >= 3),
                unsafe_allow_html=True)
    chart_heading(f"{choice}: {'little observed difference' if gap < 3 else f'gap of {gap:.1f} points'} "
                  f"({t['rate'].min():.1f}% to {t['rate'].max():.1f}%)",
                  "The 3-percentage-point threshold is an illustrative rule of thumb for this dashboard, "
                  "not a statistical test.")
    show(bar_chart(t, choice, highlight=gap >= 3))

    q75 = fdf.loc[fdf["dropout"] == 1, "gpa_semester_1"].quantile(0.75)
    q25 = fdf.loc[fdf["dropout"] == 0, "gpa_semester_1"].quantile(0.25)
    chart_heading(f"Most dropouts had a GPA below {q75:.1f}; most who stayed were above {q25:.1f}",
                  "The box covers the middle half of students; the line is the median.")
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
        f"- **Source:** {DATA_SOURCE}\n"
        f"- **Size:** {len(df):,} students · {df.shape[1]} columns · {df['nationality'].nunique()} nationalities\n"
        "- **Dropout:** the `dropout` column (1 = dropped out, 0 = retained or graduated). "
        "*Dropout rate* = share of a group with dropout = 1.\n"
        "- **Synthetic data:** the dataset appears synthetic (exactly 100,000 rows; near-identical rates "
        "across nationality, age and course). Findings demonstrate the method, not a real university.\n"
        "- **Correlation is not cause:** charts show what goes together with dropout, not what causes it."
    )
    st.markdown(f"#### Selected student records ({len(fdf):,})")
    st.dataframe(fdf.head(100), width="stretch", hide_index=True)
    st.download_button("⬇️ Download selected students (CSV)", data=fdf.to_csv(index=False).encode("utf-8"),
                       file_name="selected_students.csv", mime="text/csv")
