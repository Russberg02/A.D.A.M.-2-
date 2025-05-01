import streamlit as st
import pandas as pd
from pickle import load
import pickle
import numpy as np
import math as m
from PIL import Image
import os
from glob import glob


# HEADER
st.header("Advanced corrodeD pipe structurAl integrity systeM (ADAM)")

# IMAGE
htp = "https://www.researchgate.net/profile/Changqing-Gong/publication/313456917/figure/fig1/AS:573308992266241@1513698923813/Schematic-illustration-of-the-geometry-of-a-typical-corrosion-defect.png"
st.image(htp, caption="Fig. 1: Schematic of corrosion defect geometry")

# SIDEBAR INPUTS
st.sidebar.header('User Input Parameters')

def user_input_features():
    pipe_thickness = st.sidebar.number_input('Pipe Thickness t [mm]', value=0.01)
    pipe_diameter = st.sidebar.number_input('Pipe Diameter D [mm]', value=0.01)
    pipe_length = st.sidebar.number_input('Pipe Length L [mm]', value=0.01)
    corrosion_length = st.sidebar.number_input('Corrosion Length Lc [mm]', value=0.01)
    corrosion_depth = st.sidebar.number_input('Corrosion Depth Dc [mm]', value=0.01)
    Sy = st.sidebar.number_input('Yield Stress Sy [MPa]', value=0.01)
    UTS = st.sidebar.number_input('Ultimate Tensile Strength UTS [MPa]', value=0.01)
    Pop_Max = st.sidebar.slider('Max Operating Pressure Pop_Max [MPa]', 0, 50, 10)
    Pop_Min = st.sidebar.slider('Min Operating Pressure Pop_Min [MPa]', 0, 50, 5)
    return pd.DataFrame({'t (mm)': [pipe_thickness], 'D (mm)': [pipe_diameter],
                         'L (mm)': [pipe_length], 'Lc (mm)': [corrosion_length],
                         'Dc (mm)': [corrosion_depth], 'Sy (MPa)': [Sy],
                         'UTS (MPa)': [UTS], 'Pop_Max (MPa)': [Pop_Max], 'Pop_Min (MPa)': [Pop_Min]})

df = user_input_features()

# Extract Inputs
t, D, L, Lc, Dc, Sy, UTS, Pop_Max, Pop_Min = df.iloc[0]

# Input validation
if Pop_Min > Pop_Max:
    st.warning("⚠️ Minimum Operating Pressure is greater than Maximum Operating Pressure!")

# Burst Pressure Calculations
Pvm = 4*t*UTS/(m.sqrt(3)*D)
PTresca = 2*t*UTS/D
M = m.sqrt(1 + 0.8*(L/m.sqrt(D*t)))
Q = m.sqrt(1 + 0.31*(Lc**2)/(D*t))

P_ASME_B31G = (2*t*UTS/D)*(1 - ((2/3)*(Dc/t))/(1 - ((2/3)*(Dc/t))/M)) if L < m.sqrt(20*D*t) else (2*t*UTS/D)*(1 - Dc/t)
P_DnV = (2*UTS*t/D - t)*((1 - (Dc/t)) / (1 - (Dc/(t*Q))))
P_PCORRC = (2*t*UTS/D)*(1 - Dc/t)

# Display Inputs
st.subheader('User Input Summary')
st.write(df.style.format("{:.2f}"))

# Von Mises Stresses for Fatigue
P1max, P2max = Pop_Max*D/(2*t), Pop_Max*D/(4*t)
P1min, P2min = Pop_Min*D/(2*t), Pop_Min*D/(4*t)
P3max = P3min = 0

Svm_max = (1/m.sqrt(2))*((P1max-P2max)**2+(P2max-P3max)**2+(P3max-P1max)**2)**0.5
Svm_min = (1/m.sqrt(2))*((P1min-P2min)**2+(P2min-P3min)**2+(P3min-P1min)**2)**0.5
sigma_a = (Svm_max - Svm_min)/2
sigma_m = (Svm_max + Svm_min)/2
Se = 0.5 * UTS

# Fatigue Criteria
Goodman = (sigma_a/Se) + (sigma_m/UTS)
Soderberg = (sigma_a/Se) + (sigma_m/Sy)
Gerber = (sigma_a/Se) + ((sigma_m/UTS)**2)
Morrow_allow = Se*(1 - sigma_m/UTS)

# Display Burst Pressure Predictions
st.subheader('Burst Pressure Predictions')
burst_df = pd.DataFrame({
    'Model': ['Von Mises', 'Tresca', 'ASME B31G', 'DnV', 'PCORRC'],
    'Burst Pressure [MPa]': [Pvm, PTresca, P_ASME_B31G, P_DnV, P_PCORRC]
})
st.write(burst_df)
st.bar_chart(burst_df.set_index('Model'))

# Fatigue Assessment Table
st.subheader('Fatigue Assessment')
fatigue_df = pd.DataFrame({
    'σa [MPa]': [sigma_a], 'σm [MPa]': [sigma_m], 'Se [MPa]': [Se],
    'Goodman': [Goodman], 'Safe (Goodman)': ["Safe" if Goodman <= 1 else "Not Safe"],
    'Soderberg': [Soderberg], 'Safe (Soderberg)': ["Safe" if Soderberg <= 1 else "Not Safe"],
    'Gerber': [Gerber], 'Safe (Gerber)': ["Safe" if Gerber <= 1 else "Not Safe"],
    'Morrow Allowable σa [MPa]': [Morrow_allow], 'Safe (Morrow)': ["Safe" if sigma_a <= Morrow_allow else "Not Safe"]
})

# Apply formatting only to numeric columns
numeric_cols = fatigue_df.select_dtypes(include=[np.number]).columns
st.write(fatigue_df.style.format({col: "{:.2f}" for col in numeric_cols}))

# Bar Chart for Stress Comparison
stress_df = pd.DataFrame({
    "Stress Type": ["σa", "σm", "Se", "Yield Stress", "UTS"],
    "Value [MPa]": [sigma_a, sigma_m, Se, Sy, UTS]
})
st.subheader('Stress Comparison')
st.bar_chart(stress_df.set_index("Stress Type"))

# Goodman Diagram
st.subheader("Goodman Diagram")
fig, ax = plt.subplots()
ax.plot([0, UTS], [Se, 0], label='Goodman Line', color='r')
ax.plot([0, Sy], [Se, 0], label='Soderberg Line', color='g')
ax.plot([0, UTS], [Se, 0], label='Gerber Curve (approx)', color='b', linestyle='--')
ax.plot(sigma_m, sigma_a, 'ko', label='Operating Point')
ax.set_xlabel("Mean Stress σm [MPa]")
ax.set_ylabel("Alternating Stress σa [MPa]")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# References and Links
st.subheader('Reference')
st.write('Zhu, A comparative study of burst failure models, Journal of Pipeline Science and Engineering (2021).')

st.subheader('Assessment Resources')
st.markdown('[Case Study](https://drive.google.com/file/d/1Ako5uVRPYL5k5JeEQ_Xhl9f3pMRBjCJv/view?usp=sharing)', unsafe_allow_html=True)
st.markdown('[Corroded Pipe Burst Data](https://docs.google.com/spreadsheets/d/1YJ7ziuc_IhU7-MMZOnRmh4h21_gf6h5Z/edit#gid=56754844)', unsafe_allow_html=True)
st.markdown('[Pre-Test](https://forms.gle/wPvcgnZAC57MkCxN8)', unsafe_allow_html=True)
st.markdown('[Post-Test](https://forms.gle/FdiKqpMLzw9ENscA9)', unsafe_allow_html=True)

