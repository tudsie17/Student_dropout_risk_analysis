import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ===============================================================
# SETTINGS (edit these lines if needed)
# ===============================================================
st.set_page_config(page_title="Student Dropout Risk", page_icon="🎓", layout="wide")

DATA_SOURCE = "student_dropout_risk_2026.csv. Add where the dataset came from here (e.g. a Kaggle link)."

# One main colour + one highlight colour, used on every chart
BASE = "#8FA3B8"        # normal bars (blue-grey)
HIGHLIGHT = "#D1495B"   # the highest-risk group (red)
NEUTRAL = "#C4C9CF"     # "no difference" charts (light grey)
RISK_SCALE = ["#FBEAEC", "#E58C97", "#D1495B", "#7A1023"]  # light → dark red for the heatmap

# Plain-English names for every column we show
LABELS = {
    "age": "Age", "first_generation": "First-generation student", "disability_status": "Disability",
    "scholarship_holder": "Scholarship holder", "tuition_up_to_date": "Tuition up to date",
    "part_time_job": "Part-time job", "weekly_work_hours": "Weekly work hours", "semester": "Semester",
    "international_student": "International student", "admission_grade": "Admission grade",
    "previous_qualification_grade": "Previous qualification grade", "units_enrolled": "Units enrolled",
    "units_passed": "Units passed", "units_failed": "Units failed", "gpa_semester_1": "GPA semester 1",
    "gpa_semester_2": "GPA semester 2", "attendance_rate": "Attendance rate",
    "assignments_submitted_rate": "Assignments submitted", "library_visits_per_month": "Library visits per month",
    "lms_logins_per_week": "Online learning logins per week", "extracurricular_activities": "Extracurricular activities",
    "missed_deadlines": "Missed deadlines", "stress_level": "Stress level",
    "mental_health_support": "Uses mental health support", "satisfaction_score": "Satisfaction score",
    "commute_time_mins": "Commute time", "sleep_hours": "Sleep hours",
}


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
        st.info("Dataset not found next to dashboard.py. Upload the CSV to continue.")
        st.stop()
    df = load_data(uploaded)


# ===============================================================
# HELPER FUNCTIONS (small reusable tools for the charts)
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
    """Look up one group's rate in a rate_table (None if the group is filtered out)."""
    row = table.loc[table["group"] == group, "rate"]
    return row.iloc[0] if len(row) else None


def fmt(value):
    return "n/a" if value is None else f"{value:.1f}%"


def show_rate_chart(table, title, x_label, highlight=True, height=330):
    """Bar chart of dropout rates: one colour, highest group in red, dotted line = average.
    The title is written above the chart so long titles wrap instead of being cut off."""
    if highlight:
        colours = [HIGHLIGHT if r == table["rate"].max() else BASE for r in table["rate"]]
    else:
        colours = [NEUTRAL] * len(table)
    fig = px.bar(table, x="group", y="rate", text=table["rate"].map("{:.1f}%".format),
                 custom_data=["students"])
    fig.update_traces(marker_color=colours, textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}<br>Dropout rate: %{y:.1f}%<br>Students: %{customdata[0]:,}<extra></extra>")
    fig.add_hline(y=avg_rate, line_dash="dot", line_color="#555")
    fig.update_layout(xaxis_title=x_label, yaxis_title="Dropout rate (%)", xaxis_type="category",
                      yaxis_range=[0, min(100, max(50, table["rate"].max() * 1.2))],
                      height=height, margin=dict(t=20, b=40), showlegend=False)
    st.markdown(f"**{title}**")
    st.plotly_chart(fig, width="stretch")


def gap_title(name, table):
    """Title for the 'does it matter?' charts, based on the real gap."""
    gap = table["rate"].max() - table["rate"].min()
    lo, hi = table["rate"].min(), table["rate"].max()
    if gap < 3:
        return f"{name}: no real difference ({lo:.1f}% to {hi:.1f}%)"
    return f"{name}: gap of {gap:.1f} points ({lo:.1f}% to {hi:.1f}%)"


# ===============================================================
# SIDEBAR FILTERS
# ===============================================================
st.sidebar.title("🔍 Filters")
st.sidebar.caption("Choose which students to include. Every chart and number updates.")


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
st.sidebar.metric("Students selected", f"{len(fdf):,}", help=f"Out of {len(df):,} students in total")


# ===============================================================
# FIRST SCREEN: WHO IT'S FOR + THE QUESTION
# ===============================================================
st.title("🎓 Student Dropout Risk Dashboard")
st.markdown(
    "**For:** the Head of Student Support  \n"
    "**Question:** *Which students are most likely to drop out, and where should support go first?*"
)
st.caption(f"Showing **{len(fdf):,}** of {len(df):,} students based on the filters in the side menu.")

with st.expander("ℹ️ About this data"):
    st.markdown(
        f"- **Source:** {DATA_SOURCE}\n"
        f"- **Size:** {len(df):,} students, {df.shape[1]} columns, {df['nationality'].nunique()} nationalities.\n"
        "- **Dropout definition:** the `dropout` column. 1 = the student dropped out, "
        "0 = retained or graduated. *Dropout rate* = share of students in a group with dropout = 1.\n"
        "- **Synthetic data note:** the dataset looks synthetic (exactly 100,000 rows and almost identical "
        "rates across every nationality, age and course). Treat the findings as a demonstration of the "
        "method, not as facts about a real university.\n"
        "- **Correlation is not cause:** the charts show which factors go together with dropping out. "
        "They do not prove that changing a factor will prevent dropout."
    )

if fdf.empty:
    st.warning("No students match these filters. Tick at least one option in every filter.")
    st.stop()

avg_rate = dropout_rate(fdf)       # average for the current selection (used as the dashed line)
all_rate = dropout_rate(df)        # average for everyone


# ===============================================================
# 1. THE SIZE OF THE PROBLEM
# ===============================================================
st.header("1. The size of the problem")
k1, k2, k3 = st.columns(3)
k1.metric("Students selected", f"{len(fdf):,}")
k2.metric("Students who dropped out", f"{int(fdf['dropout'].sum()):,}")
k3.metric("Dropout rate", f"{avg_rate:.1f}%",
          delta=f"{avg_rate - all_rate:+.1f} points vs all students", delta_color="inverse")
st.markdown(f"About **1 in {round(100 / avg_rate)}** of the selected students drops out.")


# ===============================================================
# 2. ACADEMIC WARNING SIGNS
# ===============================================================
st.header("2. Academic warning signs")
st.caption("On every bar chart: the red bar is the highest-risk group, and the dotted line is the average dropout rate of the selected students.")

# --- Headline chart: dropout rate for each GPA × attendance combination ---
gpa_bins = [-0.01, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
gpa_names = ["0–1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0", "3.0–3.5", "3.5–4.0"]
att_bins = [-0.01, 50, 60, 70, 80, 90, 100]
att_names = ["0–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]

grid = fdf.assign(
    gpa_band=pd.cut(fdf["gpa_semester_1"], gpa_bins, labels=gpa_names),
    att_band=pd.cut(fdf["attendance_rate"], att_bins, labels=att_names),
)
cell_rate = grid.pivot_table(index="gpa_band", columns="att_band", values="dropout",
                             aggfunc="mean", observed=False) * 100
cell_size = grid.pivot_table(index="gpa_band", columns="att_band", values="dropout",
                             aggfunc="size", observed=False)
cell_rate = cell_rate.where(cell_size >= 30)       # hide cells with fewer than 30 students
cell_rate = cell_rate.iloc[::-1]                   # highest GPA at the top

gpa_table = rate_table(grid, "gpa_band")
low_gpa, high_gpa = rate_of(gpa_table, "0–1.5"), rate_of(gpa_table, "3.5–4.0")

fig_heat = px.imshow(
    cell_rate, text_auto=".0f", aspect="auto", zmin=0, zmax=100,
    color_continuous_scale=RISK_SCALE,
    labels=dict(x="Attendance rate", y="GPA in semester 1", color="Dropout rate (%)"),
)
fig_heat.update_traces(hovertemplate="GPA %{y}, attendance %{x}<br>Dropout rate: %{z:.1f}%<extra></extra>")
heat_title = (f"Low GPA is the clearest warning sign: {fmt(low_gpa)} of students with a GPA of 1.5 or below "
          f"drop out, against {fmt(high_gpa)} above 3.5")
fig_heat.update_layout(height=480, margin=dict(t=20))
st.subheader(heat_title)
st.plotly_chart(fig_heat, width="stretch")
st.caption("Each square shows the % of students in that GPA and attendance range who dropped out. "
           "Darker = higher risk. Squares with fewer than 30 students are left blank.")

# --- Units failed and missed deadlines ---
c1, c2 = st.columns(2)

with c1:
    fails = fdf.assign(units_failed_group=fdf["units_failed"].clip(upper=4).astype(int).astype(str).replace("4", "4+"))
    fail_table = rate_table(fails, "units_failed_group", order=["0", "1", "2", "3", "4+"])
    show_rate_chart(
        fail_table,
        f"Every failed unit adds risk: {fmt(rate_of(fail_table, '0'))} with none, "
        f"{fmt(rate_of(fail_table, '4+'))} with 4 or more",
        "Units failed")

with c2:
    deadlines = fdf.assign(deadline_group=fdf["missed_deadlines"].clip(upper=5).astype(int).astype(str).replace("5", "5+"))
    dl_table = rate_table(deadlines, "deadline_group", order=["0", "1", "2", "3", "4", "5+"])
    show_rate_chart(
        dl_table,
        f"Missed deadlines: {fmt(rate_of(dl_table, '0'))} with none, "
        f"{fmt(rate_of(dl_table, '1'))} after one, {fmt(rate_of(dl_table, '3'))} with three",
        "Missed deadlines")

# --- All candidate factors ranked by correlation ---
numeric_factors = [c for c in LABELS if c in fdf.columns]
corr = fdf[numeric_factors].corrwith(fdf["dropout"]).dropna().rename(index=LABELS).sort_values()
corr_df = corr.reset_index()
corr_df.columns = ["Factor", "Correlation with dropout"]
corr_df["Direction"] = corr_df["Correlation with dropout"].apply(
    lambda x: "More of this, more dropout" if x > 0 else "More of this, less dropout")

top_pos = corr.idxmax()
top_neg = corr.idxmin()
fig_corr = px.bar(
    corr_df, x="Correlation with dropout", y="Factor", orientation="h", color="Direction",
    text=corr_df["Correlation with dropout"].map("{:+.2f}".format),
    color_discrete_map={"More of this, more dropout": HIGHLIGHT, "More of this, less dropout": BASE},
)
corr_title = (f"All factors ranked: {top_pos} ({corr.max():+.2f}) and {top_neg} ({corr.min():+.2f}) "
          f"are most strongly linked to dropout")
fig_corr.update_layout(
    xaxis_title="Correlation with dropout (−1 to +1)", yaxis_title="", legend_title="",
    height=750, legend=dict(orientation="h", y=-0.08),
)
with st.expander("📋 See every factor ranked by correlation with dropout", expanded=False):
    st.markdown(f"**{corr_title}**")
    st.plotly_chart(fig_corr, width="stretch")
    st.caption("Correlation shows how strongly a factor moves together with dropout. "
               "Close to 0 = no link. It does not prove cause.")


# ===============================================================
# 3. MONEY AND WORK PRESSURES
# ===============================================================
st.header("3. Money and work pressures")
yes_no = {0: "No", 1: "Yes"}

m1, m2 = st.columns(2)
with m1:
    inc_table = rate_table(fdf, "family_income_level", order=["Low", "Middle", "High"])
    show_rate_chart(
        inc_table,
        f"Low-income students drop out most: Low {fmt(rate_of(inc_table, 'Low'))}, "
        f"Middle {fmt(rate_of(inc_table, 'Middle'))}, High {fmt(rate_of(inc_table, 'High'))}",
        "Family income")
with m2:
    tui_table = rate_table(fdf, "tuition_up_to_date", order=["Behind on tuition", "Up to date"],
                           names={0: "Behind on tuition", 1: "Up to date"})
    show_rate_chart(
        tui_table,
        f"Students behind on tuition: {fmt(rate_of(tui_table, 'Behind on tuition'))} drop out, "
        f"against {fmt(rate_of(tui_table, 'Up to date'))} who are up to date",
        "Tuition payments")

m3, m4 = st.columns(2)
with m3:
    job_table = rate_table(fdf, "part_time_job", order=["Yes", "No"], names=yes_no)
    show_rate_chart(
        job_table,
        f"Part-time job: {fmt(rate_of(job_table, 'Yes'))} drop out with a job, "
        f"{fmt(rate_of(job_table, 'No'))} without",
        "Has a part-time job")
with m4:
    fg_table = rate_table(fdf, "first_generation", order=["Yes", "No"], names=yes_no)
    show_rate_chart(
        fg_table,
        f"First-generation students: {fmt(rate_of(fg_table, 'Yes'))} drop out, "
        f"against {fmt(rate_of(fg_table, 'No'))} of other students",
        "First in family at university")


# ===============================================================
# 4. WHAT DOES NOT MAKE A DIFFERENCE
# ===============================================================
st.header("4. What does not make a difference")
st.markdown("These factors are worth knowing about because they **do not** separate students at risk. "
            "Dropout stays close to the average for every group, so they should not drive support decisions.")

n1, n2 = st.columns(2)
with n1:
    intl_table = rate_table(fdf, "international_student", order=["International", "Domestic"],
                            names={0: "Domestic", 1: "International"})
    show_rate_chart(intl_table, gap_title("International vs domestic", intl_table),
                               "Student type", highlight=False)
with n2:
    commute = fdf.assign(commute_band=pd.cut(fdf["commute_time_mins"], [-1, 10, 20, 30, 60, 120],
                                             labels=["0–10 min", "11–20 min", "21–30 min", "31–60 min", "61–120 min"]))
    com_table = rate_table(commute, "commute_band")
    show_rate_chart(com_table, gap_title("Commute time", com_table),
                               "Commute time (one way)", highlight=False)

n3, n4 = st.columns(2)
with n3:
    ages = fdf.assign(age_band=pd.cut(fdf["age"], [14, 18, 21, 24, 27, 30],
                                      labels=["15–18", "19–21", "22–24", "25–27", "28–30"]))
    age_table = rate_table(ages, "age_band")
    show_rate_chart(age_table, gap_title("Age", age_table), "Age group", highlight=False)
with n4:
    course_table = rate_table(fdf, "course_type")
    show_rate_chart(course_table, gap_title("Course type", course_table), "Course type",
                               highlight=False)

nat_table = rate_table(fdf, "nationality").sort_values("rate", ascending=False)
show_rate_chart(nat_table, gap_title("Nationality", nat_table), "Nationality", highlight=False)


# ===============================================================
# 5. WHAT TO DO
# ===============================================================
st.header("5. What to do")
st.markdown("Each recommendation is linked to the chart and number that supports it.")

fail_none = rate_of(fail_table, "0")
fail_any = dropout_rate(fdf[fdf["units_failed"] >= 1]) if (fdf["units_failed"] >= 1).any() else None

recommendations = [
    ("Run an early academic alert after semester 1",
     "Contact every student with a GPA of 1.5 or below, or at least one failed unit.",
     f"GPA 1.5 or below: **{fmt(low_gpa)}** drop out vs **{fmt(high_gpa)}** above 3.5. "
     f"Any failed unit: **{fmt(fail_any)}** vs **{fmt(fail_none)}** with none.",
     "Section 2: GPA × attendance heatmap and units-failed chart"),
    ("Follow up on the first missed deadline",
     "Make a missed deadline trigger a check-in from a tutor, instead of waiting for several.",
     f"**{fmt(rate_of(dl_table, '0'))}** with no missed deadlines, **{fmt(rate_of(dl_table, '1'))}** "
     f"after one, **{fmt(rate_of(dl_table, '3'))}** after three.",
     "Section 2: missed-deadlines chart"),
    ("Offer payment plans to students behind on tuition",
     "Use late tuition as an early signal for financial-support outreach.",
     f"Behind on tuition: **{fmt(rate_of(tui_table, 'Behind on tuition'))}** vs "
     f"**{fmt(rate_of(tui_table, 'Up to date'))}** up to date.",
     "Section 3: tuition chart"),
    ("Target support at low-income, working and first-generation students",
     "Prioritise bursaries for low-income students, flexible timetables for working students, "
     "and mentoring for first-generation students.",
     f"Low income **{fmt(rate_of(inc_table, 'Low'))}** vs high **{fmt(rate_of(inc_table, 'High'))}**; "
     f"part-time job **{fmt(rate_of(job_table, 'Yes'))}** vs **{fmt(rate_of(job_table, 'No'))}**; "
     f"first-generation **{fmt(rate_of(fg_table, 'Yes'))}** vs **{fmt(rate_of(fg_table, 'No'))}**.",
     "Section 3: income, part-time job and first-generation charts"),
    ("Don't build programmes around nationality, international status or commute time",
     "These groups drop out at about the same rate, so targeting them would not reach more at-risk students.",
     f"International **{fmt(rate_of(intl_table, 'International'))}** vs domestic "
     f"**{fmt(rate_of(intl_table, 'Domestic'))}**; commute time "
     f"{com_table['rate'].min():.1f}% to {com_table['rate'].max():.1f}%.",
     "Section 4: international and commute-time charts"),
]

for number, (title, action, evidence, chart) in enumerate(recommendations, start=1):
    with st.container(border=True):
        st.markdown(f"**{number}. {title}**  \n{action}")
        st.markdown(f"📊 **Evidence:** {evidence}  \n🔗 *See: {chart}*")

st.caption("These groups are at higher risk, but the data cannot show which actions will work. "
           "Pilot each action and measure the dropout rate before and after.")


# ===============================================================
# RAW DATA
# ===============================================================
with st.expander("📄 View and download the selected student records"):
    st.dataframe(fdf.head(100), width="stretch", hide_index=True)
    st.download_button("⬇️ Download selected students (CSV)", data=fdf.to_csv(index=False).encode("utf-8"),
                       file_name="selected_students.csv", mime="text/csv")
