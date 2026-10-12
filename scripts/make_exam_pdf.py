"""Genere IA_TP_Examen_IDS.pdf depuis IA_TP_Examen_IDS.md + graphiques outputs/.

Usage : .venv/bin/python scripts/make_exam_pdf.py
Dependance : fpdf2. Police DejaVu (Unicode, accents conserves).
"""
import os
import re

from fpdf import FPDF

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(BASE, "IA_TP_Examen_IDS.md")
PDF_PATH = os.path.join(BASE, "IA_TP_Examen_IDS.pdf")
IMG_COMPARE = os.path.join(BASE, "outputs", "comparaison_metriques.png")
IMG_CM_SVM = os.path.join(BASE, "outputs", "matrice_confusion_svm.png")
IMG_CM_HYB = os.path.join(BASE, "outputs", "matrice_confusion_hybride.png")
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT = "DejaVu"


def clean(s):
    """Nettoie le markdown inline (police Unicode : accents conserves)."""
    return s.replace("`", "").replace("**", "")


class ExamPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(FONT, style="I", size=9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, clean("IDS hybride SVM + K-Means (NSL-KDD) - TP Examen"), align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT, size=9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font(FONT, style="B", size=14)
        self.set_text_color(30, 60, 120)
        self.multi_cell(0, 8, clean(title))
        self.ln(1)

    def body(self, text):
        self.set_font(FONT, size=11)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 6, clean(text))
        self.ln(2)

    def metrics_table(self, header, rows):
        self.set_font(FONT, style="B", size=10)
        widths = [52, 24, 24, 24, 24, 24]
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(header):
            self.cell(widths[i], 8, clean(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font(FONT, size=10)
        self.set_text_color(20, 20, 20)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(235, 242, 255)
            for i, v in enumerate(row):
                self.cell(widths[i], 8, clean(v), border=1, fill=fill, align="C")
            self.ln()
            fill = not fill
        self.ln(3)


def main():
    text = open(MD_PATH, encoding="utf-8").read()
    parts = re.split(r"^## (\d+)\. (.*)$", text, flags=re.M)
    # parts = [intro, num1, titre1, corps1, num2, titre2, corps2, ...]
    pdf = ExamPDF()
    pdf.alias_nb_pages()
    pdf.add_font(FONT, "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
    pdf.add_font(FONT, "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
    pdf.add_font(FONT, "I", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_page()
    pdf.set_font(FONT, style="B", size=20)
    pdf.set_text_color(30, 60, 120)
    pdf.multi_cell(0, 10, clean("IA TP Examen - IDS hybride pour la détection "
                                "des intrusions réseau (SVM + K-Means)"))
    pdf.ln(2)
    pdf.set_font(FONT, size=12)
    pdf.set_text_color(20, 20, 20)
    intro = parts[0].split("#", 1)[-1].strip().lstrip("IA TP Examen").strip(" -\n")
    pdf.multi_cell(0, 6, clean(intro))
    pdf.ln(2)

    for i in range(1, len(parts), 3):
        num, title, body_text = parts[i], parts[i + 1], parts[i + 2].strip()
        pdf.section(f"{num}. {title}")
        if num == "7":
            # Texte avant le tableau, puis tableau fpdf, puis suite
            # (les lignes markdown du tableau source sont ignorees).
            before, rest = body_text.split("| Modèle", 1)
            pdf.body(before.strip())
            pdf.metrics_table(["Modèle", "Accuracy", "Precision", "Recall", "F1", "FPR"],
                              [["SVM seul", "0.7934", "0.9249", "0.6934", "0.7926", "0.0743"],
                               ["Hybride", "0.7622", "0.9205", "0.6372", "0.7531", "0.0727"]])
            after = "\n".join(l for l in rest.splitlines()
                              if not l.strip().startswith("|")).strip()
            pdf.body(after)
            for img, cap in [(IMG_COMPARE, "Comparaison des métriques (KDDTest+)"),
                             (IMG_CM_SVM, "Matrice de confusion - SVM seul"),
                             (IMG_CM_HYB, "Matrice de confusion - hybride")]:
                if os.path.exists(img):
                    if pdf.get_y() > 190:
                        pdf.add_page()
                    pdf.set_font(FONT, style="I", size=10)
                    pdf.cell(0, 7, clean(cap), align="C")
                    pdf.ln(8)
                    pdf.image(img, x=25, w=160)
                    pdf.ln(4)
        else:
            # Sous-titres ### rendus en gras.
            for block in re.split(r"^### (.*)$", body_text, flags=re.M):
                block = block.strip()
                if not block:
                    continue
                pdf.body(block)
    pdf.output(PDF_PATH)
    print(f"[OK] {PDF_PATH} ({os.path.getsize(PDF_PATH)} octets, "
          f"{pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
