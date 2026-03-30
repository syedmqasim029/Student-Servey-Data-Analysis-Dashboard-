import os

import plotly.io as pio
from flask import Flask, render_template

from survey_analysis import (
    build_additional_figures,
    build_figures,
    compute_summary,
    default_csv_path,
    load_and_clean,
)


def create_app() -> Flask:
    app = Flask(__name__)

    csv_path = os.environ.get("CSV_PATH", default_csv_path())
    df = load_and_clean(csv_path)
    figures = {}
    figures.update(build_figures(df))
    figures.update(build_additional_figures(df))
    summary = compute_summary(df)

    # Convert Plotly figures to HTML once (faster page loads).
    fig_html = {}
    for key, fig in figures.items():
        # Include Plotly JS once for the page (via CDN in the template).
        fig_html[key] = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            summary=summary,
            **fig_html,
        )

    return app


app = create_app()


if __name__ == "__main__":
    # Local dev server
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)

