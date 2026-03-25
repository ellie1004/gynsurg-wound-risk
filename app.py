"""
==========================================================================
GynSurg Wound Complication Risk Calculator
Proof-of-Concept Web Application — For Research Purposes Only

Paper: "Exploratory machine learning analysis for identification of risk
        factors for wound complications after abdominal gynecologic surgery:
        a pilot study with proof-of-concept web application"
Authors: Song YJ, Kim SK*, Yoon HS*
Journal: EJOGRB 2026

IMPORTANT: NOT validated for clinical decision-making.
==========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GynSurg Wound Risk Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1A3E72 0%, #2E86C1 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.8rem !important;
        margin-bottom: 0.3rem !important;
    }
    .main-header p {
        color: #D6EAF8 !important;
        font-size: 0.95rem;
        margin-bottom: 0 !important;
    }
    .disclaimer-banner {
        background-color: #FFF3CD;
        border: 1px solid #FFCB77;
        border-left: 5px solid #F0A500;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        font-size: 0.88rem;
        color: #664D03;
    }
    section[data-testid="stSidebar"] {
        background-color: #F7F9FC;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1A3E72;
        font-size: 1rem;
        border-bottom: 2px solid #2E86C1;
        padding-bottom: 0.3rem;
        margin-top: 1.2rem;
    }
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E5E8EB;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .section-card h3 {
        color: #1A3E72;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-top: 1px solid #E5E8EB;
        margin-top: 2rem;
        color: #808B96;
        font-size: 0.8rem;
        line-height: 1.7;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# i18n — Bilingual Text (Korean / English)
# ══════════════════════════════════════════════════════════════════════════

TEXT = {
    "ko": {
        # Header
        "app_title": "GynSurg 창상 합병증 위험도 계산기",
        "app_name_desc": "GynSurg = Gynecologic Surgery (부인과 수술)",
        "app_subtitle": "개념 증명 웹 애플리케이션 — 연구 목적 전용",
        "disclaimer": (
            "이 도구는 파일럿 연구(n=231, 8건 이벤트)의 탐색적 도구입니다. "
            "임상 사용을 위해 <strong>검증되지 않았습니다</strong>. "
            "AUROC &lt; 0.5, EPV = 0.5. 자세한 제한 사항은 논문을 참조하세요."
        ),
        # Sidebar
        "patient_info": "환자 정보",
        "patient_info_desc": "수술 전 변수를 입력하세요.",
        "demographics": "인구통계",
        "age": "연령",
        "bmi": "체질량지수 (BMI)",
        "comorbidities": "동반질환",
        "hypertension": "고혈압",
        "diabetes": "당뇨병",
        "smoking": "현재/최근 흡연",
        "pulmonary": "폐질환",
        "prior_surgery": "복부 수술 기왕력",
        "diagnosis_section": "진단",
        "primary_dx": "주 진단",
        "dx_options": ["악성 종양 (암)", "양성 자궁근종", "기타 양성"],
        "incision_section": "절개 유형",
        "wound_type": "절개 방법",
        "wt_options": ["하복부 정중절개", "정중절개", "Pfannenstiel"],
        "sc_fat_section": "피하지방 두께 (초음파)",
        "sc_sp": "치골결합 상연",
        "sc_below": "배꼽 2cm 하방",
        "sc_above": "배꼽 2cm 상방",
        "lab_section": "수술 전 검사 수치",
        "albumin": "알부민",
        "hemoglobin": "헤모글로빈",
        "glucose": "혈당",
        "defaults_note": "기본값 = 연구 코호트 평균 (n=231)",
        # Main panel
        "risk_assessment": "위험도 평가",
        "gauge_title": "창상 합병증 추정 위험도",
        "risk_low": "**저위험** — 모집단 평균 위험도(3.5%) 미만",
        "risk_moderate": "**중등도 위험** — 평균 이상, 추가 예방 조치 고려",
        "risk_elevated": "**고위험** — 평균보다 현저히 높음, 강화된 모니터링 권장",
        "predicted_risk": "예측 위험도",
        "vs_baseline": "기저선 대비",
        "study_baseline": "연구 기저선",
        "feature_contrib": "변수별 기여도",
        "contrib_caption": "🔴 빨강 = 위험 증가 &nbsp;&nbsp; 🟢 초록 = 위험 감소 &nbsp;&nbsp; 계수 0인 변수(L1 제거)는 숨김 처리됨.",
        "contrib_empty": "이 환자 프로필에서 모든 변수 기여도가 0에 가깝습니다.",
        "xaxis_contrib": "Log-odds 기여도 (+ 위험 증가, − 위험 감소)",
        # Feature names
        "feat_names": {
            'Age': '연령',
            'BMI': 'BMI',
            'Hypertension': '고혈압',
            'Diabetes': '당뇨병',
            'Smoking': '현재/최근 흡연',
            'Prior_abd_surgery': '복부 수술 기왕력',
            'Pulmonary_disease': '폐질환',
            'SC_fat_SP': '피하지방 — 치골결합 상연',
            'SC_fat_below_umbilicus': '피하지방 — 배꼽 하방',
            'SC_fat_above_umbilicus': '피하지방 — 배꼽 상방',
            'Preop_albumin': '수술 전 알부민',
            'Preop_hemoglobin': '수술 전 헤모글로빈',
            'Preop_glucose': '수술 전 혈당',
            'Dx_malignant': '악성 종양 진단',
            'Dx_benign_myoma': '양성 자궁근종 진단',
            'WT_low_midline': '하복부 정중절개',
            'WT_midline': '정중절개',
        },
        # Patient summary
        "patient_summary": "환자 요약",
        "col_variable": "변수",
        "col_patient": "환자값",
        "col_study": "연구 평균 ± SD",
        "var_names": [
            '연령', 'BMI', '고혈압', '당뇨병', '흡연',
            '복부 수술력', '폐질환', '진단',
            '절개 유형', '피하지방 (SP)', '피하지방 (배꼽 하방)',
            '피하지방 (배꼽 상방)', '알부민', '헤모글로빈', '혈당',
        ],
        "yes": "예", "no": "아니오",
        "study_stats": [
            "49.0 ± 12.8", "24.0 ± 3.6",
            "17.7%", "6.1%", "3.9%", "44.2%", "0.9%",
            "38.5% 악성", "47.6% 하복부 정중",
            "2.4 ± 0.8", "2.4 ± 0.8", "2.6 ± 1.7",
            "4.1 ± 0.5", "11.7 ± 1.7", "106.4 ± 36.1",
        ],
        # Clinical notes
        "clinical_notes": "임상 참고사항",
        "risk_factors_title": "**주요 위험 요인** *(연구 내 전체 ML 모델 기반)*",
        "risk_factors": """
1. **악성 종양 진단** — 합병증군 75% vs 비합병증군 37.2%
2. **수술 전 헤모글로빈** — 높은 Hb가 합병증과 역설적 연관 (교란 가능성)
3. **복부 수술 기왕력** — 50% vs 43.9%
4. **수술 전 알부민** — 낮은 알부민 → 높은 위험 (영양 상태 지표)
5. **수술 복잡도** — 수술 시간, 출혈량
""",
        "limitations_title": "**주요 제한사항**",
        "limitations": """
1. 파일럿 연구: 231명 중 **8건** 창상 합병증
2. EPV(Events-per-variable) = **0.5**, 권장 최소값 ≥ 10 미달
3. 모델 **AUROC < 0.5** — 개인 수준 예측 신뢰도 부족
4. 이 도구는 **연구 시연 목적 전용**
5. 임상 검증을 위해 **다기관 연구(≥ 500명)** 필요
""",
        # Model details
        "model_details": "모델 상세 정보 및 계수",
        "model_desc": (
            "**모델**: L1-정규화 로지스틱 회귀 (Lasso)\n"
            "— **17개 수술 전 변수**, z-score 표준화 후 학습\n\n"
            "**기저 절편**: `log(8 / 223) = {:.4f}` (연구 유병률 3.5%)\n\n"
            "**파이프라인**: `원시값 → z-score → × 계수 → Σ log-odds → + 기저 절편 → 시그모이드 → 확률`"
        ),
        "col_feature": "변수",
        "col_coef": "계수 (β)",
        "col_status": "상태",
        "status_active": "활성",
        "status_eliminated": "L1에 의해 제거됨",
        "coef_caption": "계수는 z-score 표준화된 입력값에 적용됩니다. 계수 0.0은 L1 정규화에 의해 제거된 변수입니다.",
        # Footer
        "footer": (
            "<strong>GynSurg Wound Risk Calculator v1.0</strong> &nbsp;|&nbsp; "
            "개념 증명 애플리케이션<br>"
            "Song YJ, Kim SK, Yoon HS &nbsp;|&nbsp; EJOGRB 2026<br>"
            "Built with Streamlit &nbsp;|&nbsp; "
            "모델: L1-정규화 로지스틱 회귀<br>"
            "⚠️ <strong>임상 사용 불가</strong> — 연구 시연 목적 전용"
        ),
    },
    "en": {
        # Header
        "app_title": "GynSurg Wound Complication Risk Calculator",
        "app_name_desc": "GynSurg = Gynecologic Surgery",
        "app_subtitle": "Proof-of-Concept Web Application — For Research Purposes Only",
        "disclaimer": (
            "This is an exploratory tool from a pilot study "
            "(n=231, 8 events). It is <strong>NOT</strong> validated for clinical use. "
            "AUROC &lt; 0.5 and EPV = 0.5. See the paper for full details on limitations."
        ),
        # Sidebar
        "patient_info": "Patient Information",
        "patient_info_desc": "Enter preoperative variables below.",
        "demographics": "Demographics",
        "age": "Age",
        "bmi": "BMI",
        "comorbidities": "Comorbidities",
        "hypertension": "Hypertension",
        "diabetes": "Diabetes Mellitus",
        "smoking": "Current/Recent Smoking",
        "pulmonary": "Pulmonary Disease",
        "prior_surgery": "Prior Abdominal Surgery",
        "diagnosis_section": "Diagnosis",
        "primary_dx": "Primary Diagnosis",
        "dx_options": ["Malignant (cancer/carcinoma)", "Benign myoma/leiomyoma", "Other benign"],
        "incision_section": "Planned Incision",
        "wound_type": "Wound Type",
        "wt_options": ["Low midline", "Midline", "Pfannenstiel"],
        "sc_fat_section": "Subcutaneous Fat (US)",
        "sc_sp": "Upper margin of SP",
        "sc_below": "2 cm below umbilicus",
        "sc_above": "2 cm above umbilicus",
        "lab_section": "Preoperative Labs",
        "albumin": "Albumin",
        "hemoglobin": "Hemoglobin",
        "glucose": "Glucose",
        "defaults_note": "Default values = study cohort means (n=231)",
        # Main panel
        "risk_assessment": "Risk Assessment",
        "gauge_title": "Estimated Wound Complication Risk",
        "risk_low": "**Low risk** — Below average population risk (3.5%)",
        "risk_moderate": "**Moderate risk** — Above average, consider additional precautions",
        "risk_elevated": "**Elevated risk** — Significantly above average, recommend enhanced monitoring",
        "predicted_risk": "Predicted Risk",
        "vs_baseline": "vs Baseline",
        "study_baseline": "Study Baseline",
        "feature_contrib": "Feature Contributions",
        "contrib_caption": "🔴 Red = increases risk &nbsp;&nbsp; 🟢 Green = decreases risk &nbsp;&nbsp; Zero-coefficient features (L1-eliminated) are hidden.",
        "contrib_empty": "All feature contributions are near zero for this patient profile.",
        "xaxis_contrib": "Contribution to log-odds (+ increases risk, − decreases risk)",
        # Feature names
        "feat_names": {
            'Age': 'Age',
            'BMI': 'BMI',
            'Hypertension': 'Hypertension',
            'Diabetes': 'Diabetes Mellitus',
            'Smoking': 'Current/Recent Smoking',
            'Prior_abd_surgery': 'Prior Abdominal Surgery',
            'Pulmonary_disease': 'Pulmonary Disease',
            'SC_fat_SP': 'SC Fat — Upper SP',
            'SC_fat_below_umbilicus': 'SC Fat — Below Umbilicus',
            'SC_fat_above_umbilicus': 'SC Fat — Above Umbilicus',
            'Preop_albumin': 'Preop Albumin',
            'Preop_hemoglobin': 'Preop Hemoglobin',
            'Preop_glucose': 'Preop Glucose',
            'Dx_malignant': 'Malignant Diagnosis',
            'Dx_benign_myoma': 'Benign Myoma Diagnosis',
            'WT_low_midline': 'Low Midline Incision',
            'WT_midline': 'Midline Incision',
        },
        # Patient summary
        "patient_summary": "Patient Summary",
        "col_variable": "Variable",
        "col_patient": "Patient Value",
        "col_study": "Study Mean ± SD",
        "var_names": [
            'Age', 'BMI', 'Hypertension', 'Diabetes', 'Smoking',
            'Prior Surgery', 'Pulmonary Disease', 'Diagnosis',
            'Wound Type', 'SC Fat (SP)', 'SC Fat (below umb.)',
            'SC Fat (above umb.)', 'Albumin', 'Hemoglobin', 'Glucose',
        ],
        "yes": "Yes", "no": "No",
        "study_stats": [
            "49.0 ± 12.8", "24.0 ± 3.6",
            "17.7%", "6.1%", "3.9%", "44.2%", "0.9%",
            "38.5% malignant", "47.6% low midline",
            "2.4 ± 0.8", "2.4 ± 0.8", "2.6 ± 1.7",
            "4.1 ± 0.5", "11.7 ± 1.7", "106.4 ± 36.1",
        ],
        # Clinical notes
        "clinical_notes": "Clinical Notes",
        "risk_factors_title": "**Key Risk Factors** *(across all ML models in the study)*",
        "risk_factors": """
1. **Malignant diagnosis** — 75% of complication cases vs 37.2% of non-complication cases
2. **Preoperative hemoglobin** — higher Hb paradoxically associated with complications (possible confounding)
3. **Prior abdominal surgery** — 50% vs 43.9%
4. **Preoperative albumin** — lower albumin → higher risk (nutritional marker)
5. **Operative complexity** — longer surgery time, greater blood loss
""",
        "limitations_title": "**Important Limitations**",
        "limitations": """
1. Pilot study: only **8 wound complication events** in 231 patients
2. Events-per-variable (EPV) = **0.5**, far below the recommended ≥ 10
3. Model **AUROC < 0.5** — cannot reliably discriminate individual outcomes
4. This tool is for **research demonstration purposes only**
5. A multicenter study with **≥ 500 patients** is required for clinical validation
""",
        # Model details
        "model_details": "Model Details & Coefficients",
        "model_desc": (
            "**Model**: L1-Regularized Logistic Regression (Lasso)\n"
            "— trained on **17 preoperative variables**, standardized via z-score.\n\n"
            "**Baseline intercept**: `log(8 / 223) = {:.4f}` (study prevalence 3.5%)\n\n"
            "**Pipeline**: `raw value → z-score → × coefficient → Σ log-odds → + baseline → sigmoid → probability`"
        ),
        "col_feature": "Feature",
        "col_coef": "Coefficient (β)",
        "col_status": "Status",
        "status_active": "Active",
        "status_eliminated": "Eliminated by L1",
        "coef_caption": "Coefficients operate on z-score–standardized inputs. A coefficient of 0.0 means the feature was eliminated by L1 regularization.",
        # Footer
        "footer": (
            "<strong>GynSurg Wound Risk Calculator v1.0</strong> &nbsp;|&nbsp; "
            "Proof-of-Concept Application<br>"
            "Song YJ, Kim SK, Yoon HS &nbsp;|&nbsp; EJOGRB 2026<br>"
            "Built with Streamlit &nbsp;|&nbsp; "
            "Model: L1-Regularized Logistic Regression<br>"
            "⚠️ <strong>NOT for clinical use</strong> — research demonstration only"
        ),
    },
}

# ══════════════════════════════════════════════════════════════════════════
# MODEL DATA — DO NOT MODIFY
# ══════════════════════════════════════════════════════════════════════════

LR_COEFFICIENTS = {
    'Age': -0.2955,
    'BMI': 0.0,
    'Hypertension': 0.0,
    'Diabetes': 0.1441,
    'Smoking': 0.2698,
    'Prior_abd_surgery': 0.5169,
    'Pulmonary_disease': 0.0,
    'SC_fat_SP': 0.0142,
    'SC_fat_below_umbilicus': -0.1912,
    'SC_fat_above_umbilicus': 0.0,
    'Preop_albumin': -0.4419,
    'Preop_hemoglobin': 0.8478,
    'Preop_glucose': -0.9009,
    'Dx_malignant': 0.7129,
    'Dx_benign_myoma': 0.0,
    'WT_low_midline': 0.3559,
    'WT_midline': 0.0,
}

POPULATION_STATS = {
    'Age':                    {'mean': 49.0,   'std': 12.8},
    'BMI':                    {'mean': 24.0,   'std': 3.6},
    'Hypertension':           {'mean': 0.177,  'std': 0.383},
    'Diabetes':               {'mean': 0.061,  'std': 0.239},
    'Smoking':                {'mean': 0.039,  'std': 0.194},
    'Prior_abd_surgery':      {'mean': 0.442,  'std': 0.498},
    'Pulmonary_disease':      {'mean': 0.009,  'std': 0.093},
    'SC_fat_SP':              {'mean': 2.4,    'std': 0.8},
    'SC_fat_below_umbilicus': {'mean': 2.4,    'std': 0.8},
    'SC_fat_above_umbilicus': {'mean': 2.6,    'std': 1.7},
    'Preop_albumin':          {'mean': 4.1,    'std': 0.5},
    'Preop_hemoglobin':       {'mean': 11.7,   'std': 1.7},
    'Preop_glucose':          {'mean': 106.4,  'std': 36.1},
    'Dx_malignant':           {'mean': 0.385,  'std': 0.488},
    'Dx_benign_myoma':        {'mean': 0.264,  'std': 0.442},
    'WT_low_midline':         {'mean': 0.476,  'std': 0.500},
    'WT_midline':             {'mean': 0.208,  'std': 0.407},
}

BASELINE_LOG_ODDS = math.log(8 / 223)


# ══════════════════════════════════════════════════════════════════════════
# RISK CALCULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════

def compute_risk(patient_values: dict) -> tuple:
    """
    Pipeline: raw → z-score → coefficient × z → sum log-odds
              → add baseline intercept → logistic sigmoid → probability
    """
    log_odds = 0.0
    contributions = {}
    for feat, raw_val in patient_values.items():
        stats = POPULATION_STATS[feat]
        z = (raw_val - stats['mean']) / stats['std'] if stats['std'] > 0 else 0.0
        coef = LR_COEFFICIENTS[feat]
        c = z * coef
        log_odds += c
        contributions[feat] = c
    total = BASELINE_LOG_ODDS + log_odds
    prob = 1.0 / (1.0 + math.exp(-total))
    return prob, contributions


# ══════════════════════════════════════════════════════════════════════════
# LANGUAGE SELECTOR & HELPER
# ══════════════════════════════════════════════════════════════════════════

lang_col1, lang_col2 = st.columns([6, 1])
with lang_col2:
    lang = st.selectbox(
        "🌐",
        ["한국어", "English"],
        label_visibility="collapsed",
    )
lang_key = "ko" if lang == "한국어" else "en"
t = TEXT[lang_key]


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="main-header">
    <h1>🏥 {t["app_title"]}</h1>
    <p style="font-size:0.8rem; color:#A9CCE3; margin-bottom:0.4rem;">{t["app_name_desc"]}</p>
    <p>{t["app_subtitle"]}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="disclaimer-banner">
    <strong>⚠️ Disclaimer:</strong> {t["disclaimer"]}
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Patient Input
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"## 📋 {t['patient_info']}")
    st.caption(t["patient_info_desc"])

    st.markdown(f"### {t['demographics']}")
    age = st.slider(t["age"], 14, 90, 50, 1, format="%d years")
    bmi = st.slider(t["bmi"], 14.0, 45.0, 24.0, 0.1, format="%.1f kg/m²")

    st.markdown(f"### {t['comorbidities']}")
    htn = st.checkbox(t["hypertension"])
    dm = st.checkbox(t["diabetes"])
    smoking = st.checkbox(t["smoking"])
    pul_disease = st.checkbox(t["pulmonary"])
    prior_surg = st.checkbox(t["prior_surgery"])

    st.markdown(f"### {t['diagnosis_section']}")
    diagnosis = st.selectbox(t["primary_dx"], t["dx_options"])

    st.markdown(f"### {t['incision_section']}")
    wound_type = st.selectbox(t["wound_type"], t["wt_options"])

    st.markdown(f"### {t['sc_fat_section']}")
    sc_sp = st.slider(t["sc_sp"], 0.5, 5.5, 2.4, 0.1, format="%.1f cm")
    sc_below = st.slider(t["sc_below"], 0.2, 6.0, 2.4, 0.1, format="%.1f cm")
    sc_above = st.slider(t["sc_above"], 0.2, 6.0, 2.6, 0.1, format="%.1f cm")

    st.markdown(f"### {t['lab_section']}")
    albumin = st.slider(t["albumin"], 1.0, 5.5, 4.1, 0.1, format="%.1f g/dL")
    hemoglobin = st.slider(t["hemoglobin"], 5.0, 18.0, 11.7, 0.1, format="%.1f g/dL")
    glucose = st.slider(t["glucose"], 50, 400, 106, 1, format="%d mg/dL")

    st.divider()
    st.caption(t["defaults_note"])


# ══════════════════════════════════════════════════════════════════════════
# ENCODE INPUTS & COMPUTE
# ══════════════════════════════════════════════════════════════════════════

# Encoding works by index position (same order in both languages)
dx_idx = t["dx_options"].index(diagnosis)
dx_malig = 1 if dx_idx == 0 else 0
dx_myoma = 1 if dx_idx == 1 else 0

wt_idx = t["wt_options"].index(wound_type)
wt_low = 1 if wt_idx == 0 else 0
wt_mid = 1 if wt_idx == 1 else 0

patient_values = {
    'Age': age,
    'BMI': bmi,
    'Hypertension': int(htn),
    'Diabetes': int(dm),
    'Smoking': int(smoking),
    'Prior_abd_surgery': int(prior_surg),
    'Pulmonary_disease': int(pul_disease),
    'SC_fat_SP': sc_sp,
    'SC_fat_below_umbilicus': sc_below,
    'SC_fat_above_umbilicus': sc_above,
    'Preop_albumin': albumin,
    'Preop_hemoglobin': hemoglobin,
    'Preop_glucose': glucose,
    'Dx_malignant': dx_malig,
    'Dx_benign_myoma': dx_myoma,
    'WT_low_midline': wt_low,
    'WT_midline': wt_mid,
}

risk_prob, contributions = compute_risk(patient_values)
risk_pct = risk_prob * 100
baseline_pct = 3.5


# ══════════════════════════════════════════════════════════════════════════
# MAIN PANEL — Risk Visualization
# ══════════════════════════════════════════════════════════════════════════

col_gauge, col_bar = st.columns([1, 1], gap="large")

# ── Risk Gauge ──
with col_gauge:
    st.markdown(f'<div class="section-card"><h3>⚙️ {t["risk_assessment"]}</h3></div>',
                unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_pct,
        delta={
            'reference': baseline_pct,
            'relative': False,
            'valueformat': '.1f',
            'suffix': '%p',
            'increasing': {'color': '#C0392B'},
            'decreasing': {'color': '#1E8449'},
        },
        number={'suffix': '%', 'valueformat': '.1f'},
        title={'text': t["gauge_title"],
               'font': {'size': 16, 'color': '#2C3E50'}},
        gauge={
            'axis': {
                'range': [0, 30],
                'tickwidth': 1,
                'tickcolor': '#2C3E50',
                'tickvals': [0, 3.5, 5, 10, 15, 20, 25, 30],
                'ticktext': ['0%', '3.5%', '5%', '10%', '15%', '20%', '25%', '30%'],
            },
            'bar': {'color': '#1A3E72', 'thickness': 0.25},
            'bgcolor': '#F0F2F6',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 5], 'color': '#2ecc71'},
                {'range': [5, 10], 'color': '#f1c40f'},
                {'range': [10, 20], 'color': '#e67e22'},
                {'range': [20, 30], 'color': '#e74c3c'},
            ],
            'threshold': {
                'line': {'color': '#C0392B', 'width': 3},
                'thickness': 0.8,
                'value': baseline_pct,
            },
        },
    ))
    fig_gauge.update_layout(
        height=300,
        margin=dict(t=60, b=10, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'sans-serif'},
    )
    st.plotly_chart(fig_gauge, width="stretch")

    if risk_prob < 0.05:
        st.success(t["risk_low"])
    elif risk_prob < 0.10:
        st.warning(t["risk_moderate"])
    else:
        st.error(t["risk_elevated"])

    m1, m2, m3 = st.columns(3)
    m1.metric(t["predicted_risk"], f"{risk_pct:.1f}%")
    m2.metric(t["vs_baseline"], f"{risk_pct - baseline_pct:+.1f}%p")
    m3.metric(t["study_baseline"], f"{baseline_pct}%")

# ── Feature Contribution Chart ──
with col_bar:
    st.markdown(f'<div class="section-card"><h3>📊 {t["feature_contrib"]}</h3></div>',
                unsafe_allow_html=True)

    active = [(f, c) for f, c in contributions.items() if abs(c) > 0.001]
    active.sort(key=lambda x: abs(x[1]), reverse=True)

    if active:
        feat_names = [t["feat_names"].get(f, f) for f, _ in active]
        feat_vals = [c for _, c in active]
        bar_colors = ['#e74c3c' if v > 0 else '#27ae60' for v in feat_vals]

        fig_bar = go.Figure(go.Bar(
            y=feat_names,
            x=feat_vals,
            orientation='h',
            marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in feat_vals],
            textposition='outside',
            textfont={'size': 11},
        ))
        fig_bar.update_layout(
            xaxis_title=t["xaxis_contrib"],
            yaxis=dict(autorange="reversed"),
            height=max(300, len(active) * 35 + 100),
            margin=dict(l=10, r=60, t=20, b=60),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#ECF0F1', zeroline=True,
                       zerolinecolor='#2C3E50', zerolinewidth=1.5),
            font={'family': 'sans-serif'},
        )
        st.plotly_chart(fig_bar, width="stretch")
        st.caption(t["contrib_caption"])
    else:
        st.info(t["contrib_empty"])


# ══════════════════════════════════════════════════════════════════════════
# PATIENT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f'<div class="section-card"><h3>📋 {t["patient_summary"]}</h3></div>',
            unsafe_allow_html=True)

yes, no = t["yes"], t["no"]
summary_data = {
    t["col_variable"]: t["var_names"],
    t["col_patient"]: [
        f"{age} years", f"{bmi:.1f} kg/m²",
        yes if htn else no, yes if dm else no,
        yes if smoking else no, yes if prior_surg else no,
        yes if pul_disease else no,
        diagnosis, wound_type,
        f"{sc_sp:.1f} cm", f"{sc_below:.1f} cm", f"{sc_above:.1f} cm",
        f"{albumin:.1f} g/dL", f"{hemoglobin:.1f} g/dL", f"{glucose} mg/dL",
    ],
    t["col_study"]: t["study_stats"],
}

st.dataframe(
    pd.DataFrame(summary_data),
    width="stretch",
    hide_index=True,
    height=560,
)


# ══════════════════════════════════════════════════════════════════════════
# CLINICAL NOTES
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f'<div class="section-card"><h3>📝 {t["clinical_notes"]}</h3></div>',
            unsafe_allow_html=True)

note_col1, note_col2 = st.columns(2, gap="large")

with note_col1:
    st.markdown(t["risk_factors_title"])
    st.markdown(t["risk_factors"])

with note_col2:
    st.markdown(t["limitations_title"])
    st.markdown(t["limitations"])


# ══════════════════════════════════════════════════════════════════════════
# MODEL DETAILS (collapsible)
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
with st.expander(f"🔬 {t['model_details']}"):
    st.markdown(t["model_desc"].format(BASELINE_LOG_ODDS))

    coef_df = pd.DataFrame([
        {
            t["col_feature"]: t["feat_names"].get(k, k),
            t["col_coef"]: v,
            t["col_status"]: t["status_active"] if v != 0.0 else t["status_eliminated"],
        }
        for k, v in LR_COEFFICIENTS.items()
    ])
    st.dataframe(coef_df, width="stretch", hide_index=True)
    st.caption(t["coef_caption"])


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="app-footer">
    {t["footer"]}
</div>
""", unsafe_allow_html=True)
