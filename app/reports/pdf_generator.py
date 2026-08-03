import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class ReportPDFGenerator:
    @staticmethod
    def generate_pdf(record_data: dict) -> io.BytesIO:
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
        
        # Custom Styles
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1e1b4b')
        )
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#6366f1')
        )
        heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'BodyDark',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155')
        )
        disclaimer_style = ParagraphStyle(
            'DisclaimerText',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#64748b')
        )

        story = []

        # 1. Header & Branding
        story.append(Paragraph("WriteLens AI — Verification Report", title_style))
        story.append(Paragraph("Tagline: Know Who Wrote Your Words.", subtitle_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366f1'), spaceAfter=15))

        # 2. Executive Summary Table
        is_ai = record_data.get('prediction') == 'AI Generated'
        pred_color = colors.HexColor('#ef4444') if is_ai else colors.HexColor('#10b981')
        
        pred_p = Paragraph(f"<font color='{pred_color.hexval()}'><b>{record_data.get('prediction', 'Unknown')}</b></font>", styles['Heading2'])
        conf_p = Paragraph(f"<b>{record_data.get('confidence_score', 0)}%</b>", body_style)
        ai_prob_p = Paragraph(f"AI Probability: <b>{record_data.get('ai_prob', 0)}%</b>", body_style)
        human_prob_p = Paragraph(f"Human Probability: <b>{record_data.get('human_prob', 0)}%</b>", body_style)
        date_str = record_data.get('created_at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        summary_data = [
            [Paragraph("<b>Prediction Result:</b>", body_style), pred_p],
            [Paragraph("<b>Confidence Score:</b>", body_style), conf_p],
            [Paragraph("<b>Probability Breakdown:</b>", body_style), Paragraph(f"{ai_prob_p.text} | {human_prob_p.text}", body_style)],
            [Paragraph("<b>Report Generated On:</b>", body_style), Paragraph(str(date_str), body_style)],
            [Paragraph("<b>Document Title:</b>", body_style), Paragraph(record_data.get('title', 'Untitled Document'), body_style)]
        ]

        summary_table = Table(summary_data, colWidths=[140, 380])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))

        story.append(Paragraph("Executive Summary", heading_style))
        story.append(summary_table)
        story.append(Spacer(1, 15))

        # 3. Writing Statistics Table
        stats = record_data.get('stats', {})
        story.append(Paragraph("Document Writing Statistics", heading_style))
        
        stats_data = [
            [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style)],
            [Paragraph("Word Count", body_style), Paragraph(str(stats.get('word_count', 0)), body_style), Paragraph("Sentence Count", body_style), Paragraph(str(stats.get('sentence_count', 0)), body_style)],
            [Paragraph("Paragraph Count", body_style), Paragraph(str(stats.get('paragraph_count', 0)), body_style), Paragraph("Reading Time", body_style), Paragraph(str(stats.get('reading_time_formatted', '0 sec')), body_style)],
            [Paragraph("Avg Sentence Length", body_style), Paragraph(f"{stats.get('avg_sentence_length', 0)} words", body_style), Paragraph("Vocabulary Richness (TTR)", body_style), Paragraph(f"{round(stats.get('type_token_ratio', 0)*100, 1)}%", body_style)],
            [Paragraph("Punctuation Density", body_style), Paragraph(str(stats.get('punctuation_density', 0)), body_style), Paragraph("Burstiness Standard Dev", body_style), Paragraph(str(stats.get('sentence_length_std', 0)), body_style)]
        ]

        stats_table = Table(stats_data, colWidths=[140, 120, 140, 120])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e0e7ff')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 15))

        # 4. Explanations of Prediction
        explanations = record_data.get('explanations', [])
        if explanations:
            story.append(Paragraph("Detection Reasoning & Stylometric Factors", heading_style))
            for exp in explanations:
                exp_title = exp.get('title', '')
                exp_desc = exp.get('description', '')
                bullet_color = "#ef4444" if exp.get('type') == 'ai' else "#10b981"
                p_text = f"<font color='{bullet_color}'>■</font> <b>{exp_title}</b>: {exp_desc}"
                story.append(Paragraph(p_text, body_style))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 10))

        # 5. AI Source Estimation (if AI)
        ai_sources = record_data.get('ai_sources', {})
        sources_list = ai_sources.get('sources', [])
        if is_ai and sources_list:
            story.append(Paragraph("AI Model Source Estimation (Experimental)", heading_style))
            src_text_list = []
            for s in sources_list:
                src_text_list.append(f"{s['name']}: {s['probability']}%")
            story.append(Paragraph(" | ".join(src_text_list), body_style))
            story.append(Spacer(1, 3))
            story.append(Paragraph(f"<i>Note: {ai_sources.get('disclaimer', '')}</i>", disclaimer_style))
            story.append(Spacer(1, 15))

        # 6. Sentence Breakdown Table
        sentences = record_data.get('sentence_analysis', [])
        if sentences:
            story.append(Paragraph("Sentence-Level Inspection (Top Sentences)", heading_style))
            
            sent_table_data = [[
                Paragraph("<b>Sentence</b>", body_style),
                Paragraph("<b>Risk Level</b>", body_style),
                Paragraph("<b>Confidence</b>", body_style),
                Paragraph("<b>Reason</b>", body_style)
            ]]

            # Limit to top 15 sentences to avoid overly long PDFs
            for s in sentences[:15]:
                r_color = "#ef4444" if "High" in s.get('risk_level', '') else ("#f59e0b" if "Moderate" in s.get('risk_level', '') else "#10b981")
                sent_table_data.append([
                    Paragraph(s.get('sentence', '')[:80] + ('...' if len(s.get('sentence', '')) > 80 else ''), body_style),
                    Paragraph(f"<font color='{r_color}'><b>{s.get('risk_level', '')}</b></font>", body_style),
                    Paragraph(f"{s.get('confidence', 0)}%", body_style),
                    Paragraph(s.get('reason', ''), body_style)
                ])

            sent_table = Table(sent_table_data, colWidths=[200, 100, 60, 160])
            sent_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING', (0,0), (-1,-1), 4),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(sent_table)

        # 7. Document Footer
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceAfter=10))
        story.append(Paragraph("Generated automatically by WriteLens AI Content Detection Engine. Confidential Document.", disclaimer_style))

        doc.build(story)
        buffer.seek(0)
        return buffer
