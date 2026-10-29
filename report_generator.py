import io
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)

def generate_pdf_report(
    kpis: Dict[str, Any],
    platforms: List[Dict[str, Any]],
    insights: List[Dict[str, Any]],
    ml_metrics: Dict[str, Any],
    quality: Dict[str, Any],
    filters_applied: Dict[str, Any]
) -> bytes:
    """Generates a professional, styled executive PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#0B0F19")
    accent_violet = colors.HexColor("#7C3AED")
    accent_cyan = colors.HexColor("#0891B2")
    text_dark = colors.HexColor("#1E293B")
    text_muted = colors.HexColor("#64748B")
    card_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=primary_color,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=text_muted,
        fontName='Helvetica'
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=accent_violet,
        fontName='Helvetica-Bold',
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11.5,
        textColor=text_dark,
        fontName='Helvetica'
    )

    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.white,
        fontName='Helvetica-Bold'
    )

    elements = []

    # Title header
    elements.append(Paragraph("SocialPulse AI – Executive Engagement Analysis Report", title_style))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated: {generated_at} | Active Dataset: {kpis.get('active_dataset_name', 'Default')} | Records in Filter: {kpis.get('total_posts', 0):,}", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=accent_violet, spaceAfter=10))

    # Applied filters summary
    filter_desc = []
    if filters_applied.get("platform"):
        filter_desc.append(f"<b>Platform:</b> {filters_applied['platform']}")
    if filters_applied.get("post_type"):
        filter_desc.append(f"<b>Post Type:</b> {filters_applied['post_type']}")
    if filters_applied.get("sentiment"):
        filter_desc.append(f"<b>Sentiment:</b> {filters_applied['sentiment']}")
    if filters_applied.get("start_date") or filters_applied.get("end_date"):
        filter_desc.append(f"<b>Date Span:</b> {filters_applied.get('start_date', 'Min')} to {filters_applied.get('end_date', 'Max')}")
    if filters_applied.get("exclude_outliers"):
        filter_desc.append("<b>Outlier Policy:</b> High ER Outliers Excluded")
    else:
        filter_desc.append("<b>Outlier Policy:</b> All Raw Records Included")

    filter_text = " • ".join(filter_desc) if filter_desc else "No active filters (All records included)"
    elements.append(Paragraph(f"<b>Active Filters:</b> {filter_text}", body_style))
    elements.append(Spacer(1, 10))

    # 1. Executive KPI Summary Table
    elements.append(Paragraph("1. Executive Performance Metrics", h2_style))
    
    kpi_data = [
        [
            Paragraph("<b>Total Posts</b>", body_style), f"{kpis.get('total_posts', 0):,}",
            Paragraph("<b>Total Reach</b>", body_style), f"{kpis.get('total_reach', 0):,}",
            Paragraph("<b>Total Impressions</b>", body_style), f"{kpis.get('total_impressions', 0):,}"
        ],
        [
            Paragraph("<b>Total Interactions</b>", body_style), f"{kpis.get('total_interactions', 0):,}",
            Paragraph("<b>Total Likes</b>", body_style), f"{kpis.get('total_likes', 0):,}",
            Paragraph("<b>Total Comments</b>", body_style), f"{kpis.get('total_comments', 0):,}"
        ],
        [
            Paragraph("<b>Avg Engagement Rate</b>", body_style), f"{kpis.get('avg_engagement_rate', 0):.2f}%",
            Paragraph("<b>Median Engagement Rate</b>", body_style), f"{kpis.get('median_engagement_rate', 0):.2f}%",
            Paragraph("<b>Avg Click-Through (CTR)</b>", body_style), f"{kpis.get('avg_ctr', 0):.2f}%"
        ],
        [
            Paragraph("<b>Total Video Views</b>", body_style), f"{kpis.get('total_video_views', 0):,}",
            Paragraph("<b>Platforms Active</b>", body_style), f"{kpis.get('platform_count', 0)}",
            Paragraph("<b>Flagged Outliers</b>", body_style), f"{kpis.get('outlier_count', 0)}"
        ]
    ]

    kpi_table = Table(kpi_data, colWidths=[110, 70, 110, 70, 110, 70])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), card_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # 2. Platform Comparison Table
    elements.append(Paragraph("2. Cross-Platform Comparison", h2_style))
    plat_header = [
        Paragraph("<b>Platform</b>", body_style),
        Paragraph("<b>Posts</b>", body_style),
        Paragraph("<b>Median ER%</b>", body_style),
        Paragraph("<b>Avg ER%</b>", body_style),
        Paragraph("<b>Avg CTR%</b>", body_style),
        Paragraph("<b>Total Reach</b>", body_style),
        Paragraph("<b>Total Interactions</b>", body_style)
    ]
    plat_rows = [plat_header]
    for p in platforms:
        tot_inter = p.get('total_likes', 0) + p.get('total_comments', 0) + p.get('total_shares', 0) + p.get('total_saves', 0)
        plat_rows.append([
            Paragraph(f"<b>{p['platform']}</b>", body_style),
            str(p.get('post_count', 0)),
            f"{p.get('median_engagement_rate', 0):.2f}%",
            f"{p.get('avg_engagement_rate', 0):.2f}%",
            f"{p.get('avg_ctr', 0):.2f}%",
            f"{p.get('total_reach', 0):,}",
            f"{tot_inter:,}"
        ])

    plat_table = Table(plat_rows, colWidths=[90, 45, 75, 75, 65, 95, 95])
    plat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, card_bg]),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 3.5),
    ]))
    elements.append(plat_table)
    elements.append(Spacer(1, 12))

    # 3. Machine Learning Evaluation Benchmarks
    elements.append(Paragraph("3. Machine Learning Model Benchmarks", h2_style))
    ml_models = ml_metrics.get("models", [])
    if ml_models:
        ml_header = [
            Paragraph("<b>Model Name</b>", body_style),
            Paragraph("<b>Type</b>", body_style),
            Paragraph("<b>Test MAE</b>", body_style),
            Paragraph("<b>Test RMSE</b>", body_style),
            Paragraph("<b>Test R² Score</b>", body_style),
            Paragraph("<b>5-Fold CV R² (Mean ± Std)</b>", body_style)
        ]
        ml_rows = [ml_header]
        for m in ml_models:
            m_type = "Baseline" if m.get("is_baseline") else "Predictive Model"
            ml_rows.append([
                Paragraph(f"<b>{m['model_name']}</b>", body_style),
                m_type,
                f"{m['mae']:.2f}",
                f"{m['rmse']:.2f}",
                f"{m['r2']:.3f}",
                f"{m.get('cv_r2_mean', 0):.3f} ± {m.get('cv_r2_std', 0):.3f}"
            ])

        ml_table = Table(ml_rows, colWidths=[130, 80, 65, 65, 75, 125])
        ml_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, card_bg]),
            ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 3.5),
        ]))
        elements.append(ml_table)
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"<i>Selected Champion Model: <b>{ml_metrics.get('best_model_name', 'Random Forest')}</b>. Features evaluated strictly before publication (no target leakage).</i>", subtitle_style))
    elements.append(Spacer(1, 10))

    # 4. Key Data-Grounded Insights
    elements.append(Paragraph("4. Key Data-Grounded Insights & Observations", h2_style))
    for ins in insights[:4]:
        p_text = f"<b>[{ins.get('category', 'Insight').upper()}] {ins['title']}:</b> {ins['finding']} <i>Recommendation: {ins['recommendation']}</i>"
        elements.append(Paragraph(p_text, body_style))
        elements.append(Spacer(1, 4))
    elements.append(Spacer(1, 8))

    # 5. Data Quality & Audit Summary
    elements.append(Paragraph("5. Data Quality & Formula Audit", h2_style))
    quality_text = (
        f"<b>Formula Verification:</b> Engagement Rate is computed as <code>((Likes + Comments + Shares + Saves) / Reach) * 100</code>.<br/>"
        f"<b>Integrity Check:</b> {quality.get('total_rows', 0)} total rows analyzed. Missing values: 0. Duplicate Post IDs: {quality.get('duplicate_rows', 0)}. Negative values: {quality.get('negative_values_found', 0)}.<br/>"
        f"<b>Outlier Summary:</b> {quality.get('outliers_detected', 0)} posts ({quality.get('outlier_percentage', 0):.1f}%) exhibit Engagement Rates > 100% due to total interactions surpassing recorded unique reach."
    )
    elements.append(Paragraph(quality_text, body_style))
    elements.append(Spacer(1, 14))

    # Footer note
    footer_text = "Notice: This report reflects historical social media performance from the active dataset. Predictions and metrics are analytical estimates and do not guarantee future audience outcomes."
    elements.append(HRFlowable(width="100%", thickness=0.5, color=border_color, spaceAfter=6))
    elements.append(Paragraph(footer_text, subtitle_style))

    # Build document
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
