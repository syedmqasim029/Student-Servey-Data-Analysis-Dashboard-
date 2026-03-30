import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _normalize_str_series(s: pd.Series) -> pd.Series:
    """Trim whitespace and normalize common survey formatting issues."""
    return s.astype(str).str.strip().replace({"nan": np.nan})


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the survey CSV and convert it into a clean dataframe with normalized
    column names that match the rest of the analysis/plotting code.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names (Google Forms sometimes adds trailing spaces).
    df.columns = [_normalize_str_series(pd.Series([c])).iloc[0] for c in df.columns]

    # Drop fully blank columns (e.g., extra trailing column in exports).
    blank_cols = [c for c in df.columns if str(c).strip() == ""]
    if blank_cols:
        df = df.drop(columns=blank_cols)

    col_map = {
        "Current Major": "Major",
        "Current University Year:": "Year",
        "City Type": "City_Type",
        "How sure are you about your future job?": "Career_Certainty",
        "Expected Starting Salary:": "Expected_Salary",
        "How many hours do you spend daily on self learning?": "Self_Learn_Hours",
        "Current CGPA:": "CGPA",
        "How confident are you in your technical/professional skills right now?": "Skill_Confidence",
        "How hopeful are you that the current job market has space for you?": "Job_Market_Hope",
        "Mentor Availability:": "Mentor_Availability",
        "How supportive is your family regarding your own career choice?": "Family_Support",
        "Biggest Career Hurdle:": "Biggest_Hurdle",
    }
    df = df.rename(columns=col_map)

    # Ensure key columns exist (fail early with a helpful message).
    required = [
        "Gender",
        "Major",
        "Year",
        "City_Type",
        "Career_Certainty",
        "Expected_Salary",
        "Self_Learn_Hours",
        "CGPA",
        "Skill_Confidence",
        "Job_Market_Hope",
        "Mentor_Availability",
        "Family_Support",
        "Biggest_Hurdle",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}. Found columns: {list(df.columns)}")

    # Strip important categoricals.
    for c in ["Gender", "Major", "Year", "City_Type", "Mentor_Availability", "Family_Support", "Biggest_Hurdle"]:
        df[c] = _normalize_str_series(df[c])

    # Normalize a few known category strings to keep charts clean.
    df["City_Type"] = df["City_Type"].replace(
        {"Metropolitan (Lahore/Karachi/Islamabad)": "Metropolitan"}
    )

    df["Year"] = df["Year"].replace({"1st ": "1st", "2nd ": "2nd", "3rd ": "3rd", "4th ": "4th"})

    # Convert numeric columns safely.
    numeric_cols = [
        "Career_Certainty",
        "Expected_Salary",
        "Self_Learn_Hours",
        "CGPA",
        "Skill_Confidence",
        "Job_Market_Hope",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing key values.
    df = df.dropna(subset=required).copy()

    # Ordinal encodings (used for correlations / consistent ordering in plots).
    year_order = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
    df["Year_Num"] = df["Year"].map(year_order)

    support_order = {"Neutral": 0, "Highly Supportive": 1}
    df["Support_Num"] = df["Family_Support"].map(support_order)

    mentor_order = {"No": 0, "Searching": 1, "Yes": 2}
    df["Mentor_Num"] = df["Mentor_Availability"].map(mentor_order)

    # Guard against any unexpected categorical values breaking ordering.
    df = df.dropna(subset=["Year_Num", "Support_Num", "Mentor_Num"]).copy()

    # Cap salary at 99th percentile for visuals that get skewed by extreme values.
    if len(df) > 0:
        salary_cap = df["Expected_Salary"].quantile(0.99)
        df["Expected_Salary_Capped"] = df["Expected_Salary"].clip(upper=salary_cap)
    else:
        df["Expected_Salary_Capped"] = df["Expected_Salary"]

    return df


def _apply_common_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family="Arial", size=13),
        title_font_size=16,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=50),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def build_figures(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create a consistent set of improved Plotly figures from the cleaned data."""
    hurdle_colors = {
        "Lack of Mentorship/Guidance": "#534AB7",
        "Mental Health (Anxiety/Stress)": "#D85A30",
        "Financial Constraints": "#1D9E75",
        "Parental/Societal Pressure": "#D4537E",
        "Lack of Technical Skills": "#378ADD",
        "Other": "#888780",
    }

    hurdle_counts = df["Biggest_Hurdle"].value_counts().reset_index()
    hurdle_counts.columns = ["Biggest_Hurdle", "Count"]
    hurdle_counts["Percent"] = (hurdle_counts["Count"] / len(df) * 100).round(1)
    hurdle_counts = hurdle_counts.sort_values("Count", ascending=False)

    # 1) Biggest hurdles (bar chart with counts + %)
    fig_hurdles = px.bar(
        hurdle_counts,
        x="Count",
        y="Biggest_Hurdle",
        orientation="h",
        color="Biggest_Hurdle",
        color_discrete_map=hurdle_colors,
        text="Count",
        title="Biggest Career Hurdles Reported by Students",
    )
    fig_hurdles.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Students=%{x}<br>Percent=%{customdata[0]}%<extra></extra>",
        customdata=np.stack([hurdle_counts["Percent"]], axis=1),
    )
    fig_hurdles.update_layout(showlegend=False)
    fig_hurdles.update_xaxes(title_text="Number of Students")
    fig_hurdles.update_yaxes(title_text="")

    fig_hurdles = _apply_common_layout(fig_hurdles, title="Biggest Career Hurdles Reported by Students", height=420)

    # 2) Mentor availability vs family support (heatmap of %)
    ct = pd.crosstab(df["Mentor_Availability"], df["Family_Support"])
    pct = (
        ct.div(ct.sum(axis=1), axis=0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        * 100
    ).round(1)

    fig_mentor_family = px.imshow(
        pct,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlBu",
        labels=dict(x="Family Support", y="Mentor Availability", color="Percent"),
        title="Mentor Availability vs Family Support (Row %)",
    )
    fig_mentor_family.update_layout(coloraxis_colorbar=dict(tickformat=".0f"))
    fig_mentor_family = _apply_common_layout(
        fig_mentor_family, title="Mentor Availability vs Family Support (Row %)", height=420
    )

    # 3) Gender distribution (bar)
    gender_counts = df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    fig_gender = px.bar(
        gender_counts.sort_values("Count", ascending=False),
        x="Gender",
        y="Count",
        text="Count",
        color="Gender",
        color_discrete_sequence=["#378ADD", "#D4537E", "#888780", "#1D9E75"],
        title="Gender Distribution",
    )
    fig_gender.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>Students=%{y}<extra></extra>")
    fig_gender = _apply_common_layout(fig_gender, title="Gender Distribution", height=360)
    fig_gender.update_yaxes(title_text="Number of Students")
    fig_gender.update_xaxes(title_text="")

    # 4) City type distribution (bar)
    city_counts = df["City_Type"].value_counts().reset_index()
    city_counts.columns = ["City_Type", "Count"]
    fig_city = px.bar(
        city_counts.sort_values("Count", ascending=False),
        x="City_Type",
        y="Count",
        text="Count",
        color="City_Type",
        color_discrete_sequence=["#534AB7", "#1D9E75", "#D85A30", "#888780", "#378ADD"],
        title="City Type of Respondents",
    )
    fig_city.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>Students=%{y}<extra></extra>")
    fig_city = _apply_common_layout(fig_city, title="City Type of Respondents", height=360)
    fig_city.update_yaxes(title_text="Number of Students")
    fig_city.update_xaxes(title_text="")

    # 5) Career certainty vs hope (scatter with CGPA bubble size)
    df_plot = df.copy()
    fig_scatter = px.scatter(
        df_plot,
        x="Career_Certainty",
        y="Job_Market_Hope",
        color="Biggest_Hurdle",
        size="CGPA",
        color_discrete_map=hurdle_colors,
        hover_data=["Gender", "Major", "Year", "City_Type", "Family_Support", "Mentor_Availability", "CGPA"],
        opacity=0.85,
        title="Career Certainty vs Job Market Hope (Bubble size = CGPA)",
    )
    fig_scatter.update_traces(
        marker=dict(line=dict(width=0.5, color="rgba(0,0,0,0.15)")),
        selector=dict(mode="markers"),
    )
    fig_scatter.update_xaxes(range=[0.5, 10.5], title_text="Career Certainty (1-10)")
    fig_scatter.update_yaxes(range=[0.5, 10.5], title_text="Job Market Hope (1-10)")
    fig_scatter = _apply_common_layout(fig_scatter, title="Career Certainty vs Job Market Hope (Bubble size = CGPA)", height=520)

    # 6) Avg confidence & hope by university year (grouped bars + std error)
    year_order_list = ["1st", "2nd", "3rd", "4th"]
    df_plot2 = df.copy()
    df_plot2["Year"] = pd.Categorical(df_plot2["Year"], categories=year_order_list, ordered=True)

    g = df_plot2.groupby("Year", observed=False).agg(
        Skill_Confidence_mean=("Skill_Confidence", "mean"),
        Skill_Confidence_std=("Skill_Confidence", "std"),
        Skill_Confidence_n=("Skill_Confidence", "count"),
        Job_Market_Hope_mean=("Job_Market_Hope", "mean"),
        Job_Market_Hope_std=("Job_Market_Hope", "std"),
        Job_Market_Hope_n=("Job_Market_Hope", "count"),
    )

    def _se(std, n):
        return std / np.sqrt(n) if n and n > 0 else np.nan

    skill_se = [_se(s, n) for s, n in zip(g["Skill_Confidence_std"], g["Skill_Confidence_n"])]
    hope_se = [_se(s, n) for s, n in zip(g["Job_Market_Hope_std"], g["Job_Market_Hope_n"])]

    fig_year = go.Figure()
    fig_year.add_trace(
        go.Bar(
            name="Avg Skill Confidence",
            x=g.index.astype(str).tolist(),
            y=g["Skill_Confidence_mean"].round(2).tolist(),
            error_y=dict(type="data", array=np.array(skill_se, dtype=float)),
            marker_color="#534AB7",
        )
    )
    fig_year.add_trace(
        go.Bar(
            name="Avg Job Market Hope",
            x=g.index.astype(str).tolist(),
            y=g["Job_Market_Hope_mean"].round(2).tolist(),
            error_y=dict(type="data", array=np.array(hope_se, dtype=float)),
            marker_color="#1D9E75",
        )
    )
    fig_year.update_layout(
        barmode="group",
        legend=dict(title="Metric"),
    )
    fig_year.update_xaxes(title_text="University Year")
    fig_year.update_yaxes(title_text="Average Score (1-10)", range=[0, 11])
    fig_year = _apply_common_layout(fig_year, title="Avg Confidence & Hope by University Year", height=420)

    # 7) Family support vs skill confidence (violin + points)
    fig_family_violin = px.violin(
        df,
        x="Family_Support",
        y="Skill_Confidence",
        color="Family_Support",
        color_discrete_map={"Highly Supportive": "#1D9E75", "Neutral": "#888780"},
        box=True,
        points="all",
        hover_data=["Gender", "Major", "Year", "Mentor_Availability"],
        title="Family Support vs Skill Confidence",
        category_orders={"Family_Support": ["Highly Supportive", "Neutral"]},
    )
    fig_family_violin = _apply_common_layout(
        fig_family_violin, title="Family Support vs Skill Confidence", height=430
    )

    # 8) Correlation heatmap (numeric scales)
    corr_cols = ["Career_Certainty", "CGPA", "Skill_Confidence", "Job_Market_Hope", "Self_Learn_Hours"]
    corr = df[corr_cols].corr().round(2)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlBu",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix (Selected Numeric Variables)",
    )
    fig_corr = _apply_common_layout(fig_corr, title="Correlation Matrix (Selected Numeric Variables)", height=420)

    return {
        "fig_hurdles": fig_hurdles,
        "fig_mentor_family": fig_mentor_family,
        "fig_gender": fig_gender,
        "fig_city": fig_city,
        "fig_scatter": fig_scatter,
        "fig_year": fig_year,
        "fig_family_violin": fig_family_violin,
        "fig_corr": fig_corr,
    }


def build_additional_figures(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Extra charts that were missing from the first dashboard version."""
    hurdle_colors = {
        "Lack of Mentorship/Guidance": "#534AB7",
        "Mental Health (Anxiety/Stress)": "#D85A30",
        "Financial Constraints": "#1D9E75",
        "Parental/Societal Pressure": "#D4537E",
        "Lack of Technical Skills": "#378ADD",
        "Other": "#888780",
    }

    # Self-learning hours distribution (histogram)
    fig_self_learn = px.histogram(
        df,
        x="Self_Learn_Hours",
        nbins=7,
        color_discrete_sequence=["#534AB7"],
        title="Daily Self-Learning Hours Distribution",
        labels={"Self_Learn_Hours": "Hours per Day", "count": "Number of Students"},
    )
    fig_self_learn.update_layout(bargap=0.1)
    fig_self_learn.update_traces(
        hovertemplate="Hours per Day=%{x}<br>Students=%{y}<extra></extra>",
        marker_line_width=0.5,
    )
    fig_self_learn = _apply_common_layout(
        fig_self_learn, title="Daily Self-Learning Hours Distribution", height=380
    )

    # CGPA vs Career Certainty scatter
    fig_cgpa_cert = px.scatter(
        df,
        x="CGPA",
        y="Career_Certainty",
        color="Biggest_Hurdle",
        color_discrete_map=hurdle_colors,
        hover_data=["Gender", "Major", "Year", "Mentor_Availability", "Family_Support"],
        opacity=0.85,
        title="CGPA vs Career Certainty (Color = Biggest Hurdle)",
        labels={"CGPA": "CGPA", "Career_Certainty": "Career Certainty (1-10)"},
    )
    fig_cgpa_cert.update_traces(marker=dict(size=11, line=dict(width=0.5, color="rgba(0,0,0,0.15)")))
    fig_cgpa_cert.update_xaxes(range=[2.8, 4.1])
    fig_cgpa_cert.update_yaxes(range=[0.5, 10.5])
    fig_cgpa_cert = _apply_common_layout(
        fig_cgpa_cert, title="CGPA vs Career Certainty (Color = Biggest Hurdle)", height=520
    )

    # Expected salary by major (use capped salary for readability)
    salary_col = "Expected_Salary_Capped" if "Expected_Salary_Capped" in df.columns else "Expected_Salary"
    fig_salary_major = px.box(
        df,
        x="Major",
        y=salary_col,
        color="Major",
        points="outliers",
        hover_data=["Gender", "Year", "City_Type"],
        title="Expected Starting Salary by Major (PKR/month)",
        labels={salary_col: "Expected Salary (PKR/month)", "Major": "Major"},
    )
    fig_salary_major.update_layout(showlegend=False)
    fig_salary_major = _apply_common_layout(
        fig_salary_major, title="Expected Starting Salary by Major (PKR/month)", height=430
    )

    return {
        "fig_self_learn": fig_self_learn,
        "fig_cgpa_cert": fig_cgpa_cert,
        "fig_salary_major": fig_salary_major,
    }


def compute_summary(df: pd.DataFrame) -> Dict[str, str]:
    """Compute a short plain-text summary for the dashboard."""
    hurdle_counts = df["Biggest_Hurdle"].value_counts()
    top_hurdle = hurdle_counts.idxmax()
    top_hurdle_pct = (hurdle_counts.max() / len(df) * 100).round(1)

    mentor_no_pct = (df["Mentor_Availability"] == "No").mean() * 100

    def _avg_by_family(family_label: str) -> float:
        subset = df[df["Family_Support"] == family_label]
        if len(subset) == 0:
            return float("nan")
        return subset["Skill_Confidence"].mean()

    corr = df[["CGPA", "Career_Certainty"]].corr().iloc[0, 1]

    return {
        "top_hurdle": f"Top hurdle: {top_hurdle} ({top_hurdle_pct}%)",
        "mentor_no": f"Students with no mentor: {mentor_no_pct:.1f}%",
        "avg_skill_conf_high": f"Avg skill confidence (Highly Supportive family): {_avg_by_family('Highly Supportive'):.2f}",
        "avg_skill_conf_neutral": f"Avg skill confidence (Neutral family): {_avg_by_family('Neutral'):.2f}",
        "corr_cgpa_certainty": f"Correlation (CGPA vs Career Certainty): {corr:.3f}",
        "n_responses": f"Responses used: {len(df)}",
    }


def default_csv_path() -> str:
    """Best-effort default path based on the repo folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "Contact Information 2.csv")
    return candidate

