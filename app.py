"""
app.py  (ENHANCED)
------------------
Original routes preserved:
  GET  /          → renders input form
  POST /predict   → prediction

NEW routes added:
  POST /download_report  → generates + returns PDF report
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load model bundle (unchanged)
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found. Please run  python train_model.py  first."
    )

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

scaler        = bundle["scaler"]
lr_model      = bundle["lr_model"]
rf_model      = bundle["rf_model"]
feature_names = bundle["feature_names"]
lr_accuracy   = bundle["lr_accuracy"]
rf_accuracy   = bundle["rf_accuracy"]

print(f"Model loaded. LR accuracy: {lr_accuracy}% | RF accuracy: {rf_accuracy}%")


# ─────────────────────────────────────────────
# FEATURE 1  Dynamic AI Health Tips
# ─────────────────────────────────────────────
def build_dynamic_tips(glucose, bmi, age, blood_pressure, high_risk):
    tips = []

    # Glucose
    if glucose >= 200:
        tips.append("Your glucose is critically high (>=200 mg/dL). Seek immediate medical review and strictly avoid all added sugars, white rice, and refined carbs.")
    elif glucose >= 140:
        tips.append("Your glucose is elevated (140-199 mg/dL). Reduce sugar intake significantly: cut out sugary drinks, sweets, and processed foods.")
    elif glucose >= 100:
        tips.append("Your glucose is in the pre-diabetic range (100-139 mg/dL). Switch to low-GI foods: oats, legumes, leafy greens, and whole grains.")
    else:
        tips.append("Your glucose level looks healthy. Keep eating a balanced, low-sugar diet to maintain it.")

    # BMI
    if bmi >= 35:
        tips.append("Your BMI indicates severe obesity. Focus on weight loss: even losing 5-10% of body weight can cut diabetes risk by up to 58%. Consider consulting a dietitian.")
    elif bmi >= 30:
        tips.append("Your BMI is in the obese range. Prioritise weight loss through a calorie-deficit diet and daily movement (aim for 10,000 steps/day).")
    elif bmi >= 25:
        tips.append("Your BMI is slightly overweight. Incorporate 30 minutes of moderate exercise 5 days a week.")
    else:
        tips.append("Your BMI is in the healthy range. Keep up your activity levels and nutritious diet.")

    # Age
    if age >= 60:
        tips.append("At 60+, regular health checkups every 6 months are essential. Ask your doctor about HbA1c testing and kidney function screening.")
    elif age >= 45:
        tips.append("Over 45? Schedule an annual blood glucose and cholesterol check-up. Early detection is the best prevention.")
    elif age >= 35:
        tips.append("Start tracking your blood sugar annually. Your 30s are a great time to build long-term healthy habits.")

    # Blood Pressure
    if blood_pressure >= 90:
        tips.append("Your blood pressure is high. Reduce salt intake, avoid processed foods, and consider stress-reduction techniques like yoga or meditation.")
    elif blood_pressure >= 80:
        tips.append("Your blood pressure is slightly elevated. Limit caffeine, alcohol, and salty foods.")

    # General
    if high_risk:
        tips.append("Aim for at least 150 minutes of aerobic exercise per week: this is the single most impactful lifestyle change.")
        tips.append("Drink 8-10 glasses of water daily. Replace juice and soda with water or herbal teas.")
        tips.append("Poor sleep raises blood sugar. Target 7-9 hours of quality sleep every night.")
    else:
        tips.append("Your risk is low: keep it that way with consistent healthy habits.")
        tips.append("Eat the rainbow: aim for 5+ servings of vegetables and fruits daily.")
        tips.append("Maintain quality sleep and manage stress: both directly affect insulin sensitivity.")

    return tips


# ─────────────────────────────────────────────
# FEATURE 4  Daily Plan Generator
# ─────────────────────────────────────────────
def build_daily_plan(glucose, bmi):
    plan = {}

    if bmi >= 30:
        plan["morning"] = "Wake at 6:30 AM, 25-min brisk walk or light jog, cold water with lemon"
    else:
        plan["morning"] = "Wake at 7:00 AM, 15-min stretching or yoga, glass of water before coffee"

    if glucose >= 140 and bmi >= 30:
        plan["diet"] = "Breakfast: Oats + nuts (no sugar). Lunch: Grilled chicken + salad. Dinner: Stir-fried vegetables + lentils. Zero sugary drinks."
    elif glucose >= 140:
        plan["diet"] = "Breakfast: Boiled eggs + whole wheat toast. Lunch: Brown rice + dal + vegetables. Dinner: Soup + salad. Avoid white bread and sweets."
    elif bmi >= 30:
        plan["diet"] = "Small portions 5-6 times a day. High protein (eggs, legumes, tofu). Low carb evenings. No fried food."
    else:
        plan["diet"] = "Balanced plate: half vegetables, quarter lean protein, quarter whole grains. Eat mindfully."

    if bmi >= 35:
        plan["exercise"] = "Low-impact cardio: 30 min swimming or cycling. Aim for 6,000 steps minimum."
    elif bmi >= 30 or glucose >= 140:
        plan["exercise"] = "30-min walk after lunch + 20-min evening walk. Add resistance bands 3x/week to build muscle."
    else:
        plan["exercise"] = "20-30 min jogging or cycling + 2 strength sessions per week."

    plan["evening"] = "Dinner by 7:30 PM. No screens 1 hour before bed. Target 7-9 hours sleep. Try 5-min deep breathing."
    plan["hydration"] = "8-10 glasses of water/day. Start each meal with a glass of water. Avoid sugary drinks."

    return plan


# ─────────────────────────────────────────────
# FEATURE 5  Sample Doctor Cards
# ─────────────────────────────────────────────
SAMPLE_DOCTORS = [
    {"name": "Dr. Priya Sharma",    "specialty": "Endocrinologist & Diabetologist",        "hospital": "Apollo Hospitals, Bangalore",   "rating": 4.9, "experience": "18 years", "phone": "+91 98451 12345", "available": "Mon-Sat, 10AM-5PM"},
    {"name": "Dr. Rajesh Kumar",    "specialty": "Internal Medicine & Diabetes Care",       "hospital": "Fortis Hospital, Bangalore",    "rating": 4.7, "experience": "14 years", "phone": "+91 98765 43210", "available": "Mon-Fri, 9AM-3PM"},
    {"name": "Dr. Ananya Reddy",    "specialty": "Diabetologist & Nutritionist",            "hospital": "Manipal Hospital, Bangalore",   "rating": 4.8, "experience": "11 years", "phone": "+91 80123 67890", "available": "Tue-Sun, 11AM-6PM"},
    {"name": "Dr. Suresh Nair",     "specialty": "Metabolic Disorders Specialist",          "hospital": "Narayana Health, Bangalore",    "rating": 4.6, "experience": "20 years", "phone": "+91 80987 54321", "available": "Mon-Sat, 8AM-2PM"},
    {"name": "Dr. Meena Iyer",      "specialty": "General Physician & Diabetes Prevention", "hospital": "Columbia Asia, Bangalore",      "rating": 4.5, "experience": "9 years",  "phone": "+91 77654 32198", "available": "Mon-Fri, 10AM-4PM"},
]


# ─────────────────────────────────────────────
# ORIGINAL ROUTE — unchanged
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           lr_accuracy=lr_accuracy,
                           rf_accuracy=rf_accuracy)


# ─────────────────────────────────────────────
# MODIFIED ROUTE — /predict
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        fields = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        values = []
        for field in fields:
            val = request.form.get(field, "").strip()
            if val == "":
                return render_template("index.html",
                                       error=f"Please fill in the '{field}' field.",
                                       lr_accuracy=lr_accuracy,
                                       rf_accuracy=rf_accuracy)
            values.append(float(val))

        (pregnancies, glucose, blood_pressure, skin_thickness,
         insulin, bmi, dpf, age) = values

        model_choice = request.form.get("model_choice", "lr")
        features_df  = pd.DataFrame([dict(zip(feature_names, values))])

        if model_choice == "rf":
            prediction  = rf_model.predict(features_df)[0]
            probability = rf_model.predict_proba(features_df)[0][1]
            model_label = "Random Forest"
            model_acc   = rf_accuracy
        else:
            features_scaled = scaler.transform(features_df)
            prediction      = lr_model.predict(features_scaled)[0]
            probability     = lr_model.predict_proba(features_scaled)[0][1]
            model_label     = "Logistic Regression"
            model_acc       = lr_accuracy

        high_risk    = bool(prediction == 1)
        prob_percent = round(probability * 100, 1)

        # Risk level label
        if prob_percent >= 70:
            risk_level = "High"
        elif prob_percent >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Dynamic tips
        tips = build_dynamic_tips(glucose, bmi, age, blood_pressure, high_risk)

        # Emergency alert
        emergency_alert = None
        if prob_percent > 80:
            emergency_alert = (
                "Immediate medical attention recommended! "
                "Your predicted diabetes risk exceeds 80%. "
                "Please consult an endocrinologist or visit the nearest hospital as soon as possible."
            )

        # Daily plan
        daily_plan = build_daily_plan(glucose, bmi)

        return render_template(
            "index.html",
            prediction      = high_risk,
            probability     = prob_percent,
            model_label     = model_label,
            model_acc       = model_acc,
            lr_accuracy     = lr_accuracy,
            rf_accuracy     = rf_accuracy,
            form_data       = dict(zip(fields, values)),
            model_choice    = model_choice,
            tips            = tips,
            risk_level      = risk_level,
            emergency_alert = emergency_alert,
            daily_plan      = daily_plan,
            doctors         = SAMPLE_DOCTORS,
        )

    except ValueError as e:
        return render_template("index.html",
                               error=f"Invalid input: {e}",
                               lr_accuracy=lr_accuracy,
                               rf_accuracy=rf_accuracy)


# ─────────────────────────────────────────────
# FEATURE 3  NEW ROUTE: /download_report
# ─────────────────────────────────────────────
@app.route("/download_report", methods=["POST"])
def download_report():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER

    # Pull posted data
    prediction  = request.form.get("prediction", "False")
    probability = request.form.get("probability", "N/A")
    risk_level  = request.form.get("risk_level", "N/A")
    model_label = request.form.get("model_label", "N/A")
    model_acc   = request.form.get("model_acc", "N/A")
    tips_raw    = request.form.get("tips_joined", "")
    plan_raw    = request.form.get("plan_joined", "")
    glucose     = request.form.get("Glucose", "N/A")
    bmi         = request.form.get("BMI", "N/A")
    age         = request.form.get("Age", "N/A")
    blood_p     = request.form.get("BloodPressure", "N/A")
    pregnancies = request.form.get("Pregnancies", "N/A")
    insulin     = request.form.get("Insulin", "N/A")
    skin        = request.form.get("SkinThickness", "N/A")
    dpf         = request.form.get("DiabetesPedigreeFunction", "N/A")

    tips_list = [t.strip() for t in tips_raw.split("||") if t.strip()]
    plan_list = [p.strip() for p in plan_raw.split("||") if p.strip()]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    risk_hex = {"High": "#dc2626", "Medium": "#d97706", "Low": "#059669"}.get(risk_level, "#374151")

    def style(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_s   = style("t",   fontSize=22, textColor=colors.HexColor("#1a1a2e"),   spaceAfter=4,  fontName="Helvetica-Bold")
    sub_s     = style("s",   fontSize=9,  textColor=colors.HexColor("#6b7280"),   spaceAfter=14)
    h2_s      = style("h2",  fontSize=13, textColor=colors.HexColor("#b45309"),   spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold")
    result_s  = style("res", fontSize=15, textColor=colors.HexColor(risk_hex),    alignment=TA_CENTER, spaceAfter=8, fontName="Helvetica-Bold")
    body_s    = style("b",   fontSize=9.5, leading=15, textColor=colors.HexColor("#1f2937"))
    footer_s  = style("f",   fontSize=8,  textColor=colors.HexColor("#9ca3af"),   alignment=TA_CENTER, spaceBefore=6)

    story = []
    story.append(Paragraph("DiabetesIQ - Health Report", title_s))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}   |   Model: {model_label} (Accuracy: {model_acc}%)",
        sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceAfter=10))

    # Result
    story.append(Paragraph("PREDICTION RESULT", h2_s))
    is_high = prediction in ("True", "1", "true")
    story.append(Paragraph(
        "HIGH RISK DETECTED" if is_high else "LOW RISK DETECTED", result_s))

    result_data = [
        ["Risk Level", risk_level],
        ["Diabetes Probability", f"{probability}%"],
        ["Model Used", model_label],
        ["Model Accuracy", f"{model_acc}%"],
    ]
    rt = Table(result_data, colWidths=[6*cm, 10*cm])
    rt.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#fef3c7"), colors.HexColor("#fffbeb")]),
        ("GRID",      (0,0), (-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ("PADDING",   (0,0), (-1,-1), 8),
    ]))
    story.append(rt)
    story.append(Spacer(1, 8))

    # Patient data
    story.append(Paragraph("PATIENT HEALTH DATA", h2_s))
    pt = Table([
        ["Pregnancies", pregnancies,  "Glucose (mg/dL)", glucose],
        ["Blood Pressure", blood_p,   "Skin Thickness", skin],
        ["Insulin (uU/mL)", insulin,  "BMI (kg/m2)", bmi],
        ["Diabetes Pedigree", dpf,    "Age", age],
    ], colWidths=[5.5*cm, 3.5*cm, 5.5*cm, 3.5*cm])
    pt.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 9.5),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]),
        ("GRID",      (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ("PADDING",   (0,0), (-1,-1), 7),
    ]))
    story.append(pt)
    story.append(Spacer(1, 4))

    # Tips
    story.append(Paragraph("PERSONALISED HEALTH TIPS", h2_s))
    for tip in tips_list:
        story.append(Paragraph(f"  {tip}", body_s))
        story.append(Spacer(1, 3))

    # Daily plan
    story.append(Paragraph("YOUR DAILY ACTION PLAN", h2_s))
    for item in plan_list:
        story.append(Paragraph(f"  {item}", body_s))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))
    story.append(Paragraph(
        "This report is for educational purposes only and is NOT a substitute for professional medical advice. "
        "Please consult a qualified physician for diagnosis and treatment.",
        footer_s))

    doc.build(story)
    buffer.seek(0)

    filename = f"DiabetesIQ_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, as_attachment=True,
                     download_name=filename, mimetype="application/pdf")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
