"""
Build a static GitHub Pages site from the survey dashboard.

GitHub Pages serves static files only, so we generate `site/index.html`
by rendering the Flask route once, then committing the output.
"""

import os

import app as flask_app_module


def main() -> None:
    site_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
    os.makedirs(site_dir, exist_ok=True)

    client = flask_app_module.app.test_client()
    resp = client.get("/")
    html = resp.data.decode("utf-8")

    with open(os.path.join(site_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Prevent Jekyll from rewriting the generated HTML.
    nojekyll_path = os.path.join(site_dir, ".nojekyll")
    if not os.path.exists(nojekyll_path):
        with open(nojekyll_path, "w", encoding="utf-8") as f:
            f.write("")

    print(f"Static site generated at: {site_dir}/index.html")


if __name__ == "__main__":
    main()

