# =========================================================
# FILE: src/report_generator.py
# =========================================================

from __future__ import annotations

import os

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_report_directory():

    os.makedirs(
        "reports",
        exist_ok=True
    )


def generate_timestamp():

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def generate_report_path():

    create_report_directory()

    timestamp = generate_timestamp()

    report_path = (
        f"reports/"
        f"ML_Studio_Report_{timestamp}.pdf"
    )

    return report_path


def _register_times_family() -> tuple[str, str, str]:

    font_family = {
        "regular": "Times-Roman",
        "bold": "Times-Bold",
        "italic": "Times-Italic",
    }
    font_directory = Path("C:/Windows/Fonts")
    font_candidates = {
        "MLStudioTimes": font_directory / "times.ttf",
        "MLStudioTimesBold": font_directory / "timesbd.ttf",
        "MLStudioTimesItalic": font_directory / "timesi.ttf",
    }

    try:
        for font_name, font_path in font_candidates.items():
            if font_path.exists() and font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
        if all(name in pdfmetrics.getRegisteredFontNames() for name in font_candidates):
            font_family = {
                "regular": "MLStudioTimes",
                "bold": "MLStudioTimesBold",
                "italic": "MLStudioTimesItalic",
            }
    except Exception:
        pass

    return (
        font_family["regular"],
        font_family["bold"],
        font_family["italic"],
    )


def _build_styles():

    regular_font, bold_font, italic_font = _register_times_family()
    sample_styles = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "MLStudioTitle",
            parent=sample_styles["Title"],
            fontName=bold_font,
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "MLStudioSubtitle",
            parent=sample_styles["BodyText"],
            fontName=italic_font,
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=18,
        ),
        "section_heading": ParagraphStyle(
            "MLStudioSectionHeading",
            parent=sample_styles["Heading2"],
            fontName=bold_font,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceBefore=4,
            spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "MLStudioBody",
            parent=sample_styles["BodyText"],
            fontName=regular_font,
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#1F2937"),
        ),
        "small": ParagraphStyle(
            "MLStudioSmall",
            parent=sample_styles["BodyText"],
            fontName=regular_font,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#6B7280"),
        ),
        "table_header": ParagraphStyle(
            "MLStudioTableHeader",
            parent=sample_styles["BodyText"],
            fontName=bold_font,
            fontSize=10,
            leading=12,
            textColor=colors.white,
        ),
        "table_body": ParagraphStyle(
            "MLStudioTableBody",
            parent=sample_styles["BodyText"],
            fontName=regular_font,
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#111827"),
        ),
    }

    return styles, regular_font, bold_font


def _on_page(canvas, document):

    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.setLineWidth(0.5)
    canvas.line(document.leftMargin, 0.75 * inch, letter[0] - document.rightMargin, 0.75 * inch)
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawRightString(letter[0] - document.rightMargin, 0.52 * inch, f"Page {document.page}")
    canvas.restoreState()


def _add_title_block(story, styles):

    generated_at = datetime.now().strftime("%B %d, %Y %I:%M %p")
    story.append(
        Paragraph(
            "ML Studio Automated Machine Learning Report",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            f"Prepared for workflow review and presentation readiness • Generated {generated_at}",
            styles["subtitle"],
        )
    )


def _add_section_heading(story, styles, heading):

    story.append(
        Paragraph(
            f"&#9632; {heading}",
            styles["section_heading"],
        )
    )


def _build_key_value_table(rows, styles, *, column_widths=(2.2 * inch, 4.6 * inch)):

    table_rows = [
        [
            Paragraph("<b>Item</b>", styles["table_header"]),
            Paragraph("<b>Details</b>", styles["table_header"]),
        ]
    ]
    for label, value in rows:
        table_rows.append(
            [
                Paragraph(str(label), styles["table_body"]),
                Paragraph(str(value), styles["table_body"]),
            ]
        )

    table = Table(table_rows, colWidths=list(column_widths), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#D1D5DB")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9CA3AF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _add_key_value_section(story, styles, heading, rows):

    _add_section_heading(story, styles, heading)
    story.append(_build_key_value_table(rows, styles))
    story.append(Spacer(1, 0.2 * inch))


def _add_paragraph_list_section(story, styles, heading, items):

    _add_section_heading(story, styles, heading)
    for item in items:
        story.append(
            Paragraph(
                f"&bull; {item}",
                styles["body"],
            )
        )
        story.append(Spacer(1, 0.06 * inch))
    story.append(Spacer(1, 0.12 * inch))


def generate_complete_report(

    dataset,

    training_results,

    best_model_results,

    preprocessing_results=None,

    explainability_summary=None
):

    report_path = generate_report_path()

    document = SimpleDocTemplate(

        report_path,

        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.9 * inch,
    )

    styles, _, _ = _build_styles()
    story = []

    _add_title_block(story, styles)

    dataset_rows = [
        ("Dataset Rows", dataset.shape[0]),
        ("Dataset Columns", dataset.shape[1]),
        ("Duplicate Rows", int(dataset.duplicated().sum())),
        ("Missing Values", int(dataset.isna().sum().sum())),
        ("Column Sample", ", ".join(map(str, list(dataset.columns[:8]))) + (" ..." if dataset.shape[1] > 8 else "")),
    ]
    _add_key_value_section(story, styles, "Dataset Summary", dataset_rows)

    if preprocessing_results is not None:
        cleaning_summary = preprocessing_results.get("cleaning_summary", {})
        preprocessing_rows = [
            ("Training Rows", preprocessing_results["X_train"].shape[0]),
            ("Testing Rows", preprocessing_results["X_test"].shape[0]),
            ("Processed Features", preprocessing_results["X_train"].shape[1]),
            ("Remaining Rows After Cleaning", cleaning_summary.get("remaining_rows", dataset.shape[0])),
            ("Dropped Rows", cleaning_summary.get("dropped_rows", 0)),
            ("Duplicate Handling", cleaning_summary.get("duplicate_handling", "keep").replace("_", " ").title()),
        ]
        _add_key_value_section(story, styles, "Preprocessing Summary", preprocessing_rows)

    training_rows = [
        ("Problem Type", training_results["problem_type"].title()),
        ("Best Model", training_results["best_model_name"]),
        ("Models Trained", len(training_results.get("trained_models", {}))),
        ("Training Errors", len(training_results.get("errors", {}))),
    ]
    _add_key_value_section(story, styles, "Training Summary", training_rows)

    metrics_rows = [
        (metric_name, metric_value)
        for metric_name, metric_value in best_model_results["metrics"].items()
    ]
    _add_key_value_section(story, styles, "Best Model Metrics", metrics_rows)

    if explainability_summary is not None:
        explainability_rows = [
            (key, value)
            for key, value in explainability_summary.items()
        ]
        _add_key_value_section(story, styles, "Explainability Summary", explainability_rows)

    report_notes = [
        "This report reflects the latest dataset state available in the ML Studio workflow at generation time.",
        "Model metrics correspond to the currently active best model in session.",
        f"Generated by ML Studio on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}.",
    ]
    _add_paragraph_list_section(story, styles, "Report Notes", report_notes)

    story.append(
        Paragraph(
            "End of Report",
            styles["small"],
        )
    )

    document.build(
        story,
        onFirstPage=_on_page,
        onLaterPages=_on_page,
    )

    return report_path
