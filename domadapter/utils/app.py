import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode

###################################
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


st.set_page_config(page_title="Domadapter")

st.title("Qualitative Analysis of Domadapter Models")

c28, c30, c31 = st.columns([1, 6, 1])

global shows

with c30:
    uploaded_files = st.file_uploader(
        "Upload CSVs", type="csv", accept_multiple_files=True, key="1"
    )
    if uploaded_files:
        for file in uploaded_files:
            file.seek(0)
        names = [i.name.split(".")[0] for i in uploaded_files]
        uploaded_data_read = [pd.read_csv(file) for file in uploaded_files]
        shows = pd.concat(
            [
                uploaded_data_read[0]["sentence"],
                uploaded_data_read[1]["label"],
                uploaded_data_read[0]["prediction"],
                uploaded_data_read[1]["prediction"],
            ],
            axis=1,
            keys=["sentence", "gold_label", f"label {names[0]}", f"label {names[1]}"],
        )

        shows[f"{names[0]} filter"] = 0
        shows[f"{names[0]} filter"] = shows[f"label {names[0]}"] == shows["gold_label"]

        shows[f"{names[1]} filter"] = 0
        shows[f"{names[1]} filter"] = shows[f"label {names[1]}"] == shows["gold_label"]

        st.info(
            f"""
                👆 Now you can play around the models' output by filtering where `{names[0]}` model gave right results and `{names[1]}` model didn't do well or vice versa. \n
                label {names[1]}: Label output by `{names[1]}` Model \n
                {names[1]} filter: 1 if `{names[1]}` model gave right results, 0 otherwise. Use it to filter correct, incorrect predictions. \n
                """
        )


gb = GridOptionsBuilder.from_dataframe(shows)
# enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
gridOptions = gb.build()

st.success(
    f"""
        💡 Tip! Hold the shift key when selecting rows to select multiple rows at once!
        """
)

response = AgGrid(
    shows,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=False,
)

df = pd.DataFrame(response["selected_rows"])

st.subheader("Filtered data will appear below 👇 ")
st.text("")

st.table(df)

# calculate confusion matrix and classification report for label and prediction (domain task adapter model)
confusion_matrix_domain = confusion_matrix(
    shows["gold_label"],
    shows[f"label {names[0]}"],
    labels=["entailment", "neutral", "contradiction"],
)
classification_report_domain = classification_report(
    shows["gold_label"],
    shows[f"label {names[0]}"],
    labels=["entailment", "neutral", "contradiction"],
)  # sort of like meso analysis

st.subheader(f"{names[0]} Model Report \n\n ")
st.text(classification_report_domain)

# calculate confusion matrix and classification report for label and prediction (task adapter model)
confusion_matrix_task = confusion_matrix(
    shows["gold_label"],
    shows[f"label {names[1]}"],
    labels=["entailment", "neutral", "contradiction"],
)
classification_report_task = classification_report(
    shows["gold_label"],
    shows[f"label {names[1]}"],
    labels=["entailment", "neutral", "contradiction"],
)

st.subheader(f"{names[1]} Model Report \n\n ")
st.text(classification_report_task)
