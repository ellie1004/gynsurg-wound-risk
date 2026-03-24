# GynSurg Wound Complication Risk Calculator

## Proof-of-Concept Web Application

**Paper:** "Exploratory machine learning analysis for identification of risk factors for wound complications after abdominal gynecologic surgery: a pilot study with proof-of-concept web application"

**Authors:** Song YJ, Kim SK*, Yoon HS*

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py

# 3. Open browser at http://localhost:8501
```

## Deployment (Streamlit Community Cloud)

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your GitHub account
4. Select the repository and `app.py` as the main file
5. Deploy

## File Structure

```
Appendix_B_Streamlit_App/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Disclaimer

This application is a PROOF-OF-CONCEPT for research purposes only.
It is NOT validated for clinical decision-making.
The underlying model was trained on a limited dataset (n=231, 8 events)
with suboptimal discriminative performance (AUROC < 0.5).
A multicenter study with adequate power is needed before any clinical use.

## License

This code is provided as supplementary material for the associated paper.
It may be used for academic and research purposes.
