"""
Runs the full analytics + ML pipeline end-to-end, in order:
  1. data_pipeline.py   - load, clean, merge, feature engineer
  2. eda.py             - generate visualization reports
  3. attrition_model.py - train & select best attrition model
  4. predict_risk.py    - score workforce, assign risk, generate recommendations

After this completes, run:  streamlit run app.py
"""

from analytics import data_pipeline, eda, attrition_model, predict_risk

if __name__ == "__main__":
    print("STEP 1/4: Data pipeline (clean, merge, feature engineer)")
    data_pipeline.build_master_dataset()

    print("\nSTEP 2/4: Exploratory Data Analysis")
    eda.run_eda()

    print("\nSTEP 3/4: Training attrition prediction models")
    attrition_model.train_and_select()

    print("\nSTEP 4/4: Scoring workforce risk & generating recommendations")
    predict_risk.run_predictions()

    print("\nAll done. Launch the dashboard with:  streamlit run app.py")
