"""PDF incident report for one analysis run."""
from __future__ import annotations

import datetime as dt

import pandas as pd
from fpdf import FPDF


def _clean(s) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")


def build_pdf(source: str, summary: dict, res: pd.DataFrame, settings: dict) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "RELIC-IDS incident report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, _clean(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}   Source: {source}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    lines = [f"Flows analysed: {summary['flows']:,}",
             f"Alerts: {summary['alerts']:,} ({summary['alert_rate']:.1%} of flows)",
             f"Alerting sources: {summary.get('sources', 0)}"]
    lines += [f"{k}: {v:,}" for k, v in summary["by_type"].items()]
    for ln in lines:
        pdf.cell(0, 6, _clean(ln), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    alerts = res[res["verdict"] == "Attack"]
    if not alerts.empty and "srcip" in alerts:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Alerts by source", new_x="LMARGIN", new_y="NEXT")
        g = (alerts.groupby(["srcip", "attack_type", "severity"]).size().reset_index(name="flows")
             .sort_values("flows", ascending=False).head(25))
        widths = [55, 45, 30, 25]
        pdf.set_font("Helvetica", "B", 9)
        for h, w in zip(["Source", "Type", "Severity", "Flows"], widths):
            pdf.cell(w, 7, h, border="B")
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for _, r in g.iterrows():
            for v, w in zip([r.srcip, r.attack_type, r.severity, f"{r.flows:,}"], widths):
                pdf.cell(w, 6, _clean(v))
            pdf.ln()
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Evidence", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for ev in alerts["evidence"].dropna().drop_duplicates().head(20):
            pdf.multi_cell(0, 5, _clean("- " + ev), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Settings", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for k, v in settings.items():
        pdf.multi_cell(0, 4.5, _clean(f"{k}: {v}"), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
