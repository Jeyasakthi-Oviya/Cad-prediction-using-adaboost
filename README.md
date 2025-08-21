#  Coronary Artery Disease (CAD) Prediction using AdaBoost

This is my **final year project**.  
I built a Machine Learning model (**AdaBoost**) to predict the risk of Coronary Artery Disease (CAD) using clinical features.  
A **Streamlit app** is included so anyone can test it with patient details.

---

##  Dataset
- The dataset (`CAD.csv`) is included in this repository.  
- Original dataset had 55 features → I selected 11 important ones:  
  `Typical_Chest_Pain, Age, HTN, DM, BP, Tinversion, FBS, TG, Atypical, Nonanginal, EF-TTE`  
- Target variable: `Cath`

---

##  How to Run
1. Install requirements:
```bash
pip install -r requirements.txt

## Run the Streamlit app:
2. streamlit run cad_predictor.py
----
##  Repository Structure
cad-prediction-using-adaboost/
├─ cad_predictor.py
├─ own11.ipynb
├─ CAD.csv
├─ adaboost.pkl
├─ scaler.pkl
├─ quantile_transformer.pkl
├─ requirements.txt
└─ README.md


##  Results
- Algorithm: AdaBoost
- Accuracy: 90%
- Output: Normal / Moderate / High risk classification


##  Author
**Jeyasakthi Oviya Jeyapal**
Final Year B.Sc Computer Science Project
