import streamlit as st
import pandas as pd
import numpy as np
import math as m
import seaborn as sns
import matplotlib.pyplot as plt

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

# Extract values
t = df['t (mm)'][0]
D = df['D (mm)'][0]
L = df['L (mm)'][0]
Lc = df['Lc (mm)'][0]
Dc = df['Dc (mm)'][0]
Sy = df['Sy (MPa)'][0]
UTS = df['UTS (MPa)'][0]
Pop_Max = df['Pop_Max (MPa)'][0]
Pop_Min = df['Pop_Min (MPa)'][0]

# Von Mises and Tresca for intact pipe
Pvm = 4 * t * UTS / (m.sqrt(3) * D)
PTresca = 2 * t * UTS / D

# Folias factor M
M = m.sqrt(1 + 0.8 * (L / m.sqrt(D * t)))

# ASME B31G
if L < m.sqrt(20 * D * t):
    P_ASME_B31G = (2 * t * UTS / D) * ((1 - (2/3) * (Dc / t)) / (1 - (2/3) * (Dc / t) / M))
else:
    P_ASME_B31G = (2 * t * UTS / D) * (1 - (Dc / t))

# DnV
Q = m.sqrt(1 + 0.31 * (Lc ** 2) / (D * t))
P_DnV = (2 * UTS * t / (D - t)) * ((1 - (Dc / t)) / (1 - (Dc / (t * Q))))

# PCORRC
P_PCORRC = (2 * t * UTS / D) * (1 - Dc / t)

# Display user input
st.subheader('User Input Parameters')
st.write(df)

# Burst Pressures
st.subheader('Calculated Intact Pipe Burst Pressure via Von Mises')
st.write(pd.DataFrame({'Pvm (MPa)': [f"{Pvm:.2f}"]}))

st.subheader('Calculated Corroded Pipe Burst Pressure via ASME_B31G')
st.write(pd.DataFrame({'P_ASME_B31G (MPa)': [f"{P_ASME_B31G:.2f}"]}))

st.subheader('Calculated Corrorded Pipe Burst Pressure via DnV')
st.write(pd.DataFrame({'P_DnV (MPa)': [f"{P_DnV:.2f}"]}))

st.subheader('Calculated Corrorded Pipe Burst Pressure via PCORRC')
st.write(pd.DataFrame({'P_PCORRC (MPa)': [f"{P_PCORRC:.2f}"]}))

# Principal stresses
P1max = Pop_Max * D / (2 * t)
P2max = Pop_Max * D / (4 * t)
P3max = 0

P1min = Pop_Min * D / (2 * t)
P2min = Pop_Min * D / (4 * t)
P3min = 0

# Von Mises stress
Sigma_VM_Pipe_Max_Operating_Pressure = m.sqrt(0.5 * ((P1max - P2max)**2 + (P2max - P3max)**2 + (P3max - P1max)**2))
Sigma_VM_Pipe_Min_Operating_Pressure = m.sqrt(0.5 * ((P1min - P2min)**2 + (P2min - P3min)**2 + (P3min - P1min)**2))

# Fatigue calculations
sigma_a = (Sigma_VM_Pipe_Max_Operating_Pressure - Sigma_VM_Pipe_Min_Operating_Pressure) / 2
sigma_m = (Sigma_VM_Pipe_Max_Operating_Pressure + Sigma_VM_Pipe_Min_Operating_Pressure) / 2
Se = 0.5 * UTS

Goodman_Value = (sigma_a / Se) + (sigma_m / UTS)
Goodman_Safe = Goodman_Value <= 1

Soderberg_Value = (sigma_a / Se) + (sigma_m / Sy)
Soderberg_Safe = Soderberg_Value <= 1

Gerber_Value = (sigma_a / Se) + ((sigma_m / UTS) ** 2)
Gerber_Safe = Gerber_Value <= 1

Morrow_sigma_a_allow = Se * (1 - (sigma_m / UTS))
Morrow_Safe = sigma_a <= Morrow_sigma_a_allow

# Plot using Seaborn
if UTS > 1 and Sy > 1 and Se > 0 and sigma_m >= 0 and sigma_a >= 0:
    sigma_m_range = np.linspace(0, UTS, 500)
    df_plot = pd.DataFrame({
        'sigma_m': sigma_m_range,
        'Goodman': Se * (1 - sigma_m_range / UTS),
        'Soderberg': Se * (1 - sigma_m_range / Sy),
        'Gerber': Se * (1 - (sigma_m_range / UTS) ** 2),
        'Morrow': Se * (1 - sigma_m_range / UTS)
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=df_plot, x='UTS', y='Se', label='Goodman', ax=ax, linestyle='--')

    ax.plot(sigma_m, sigma_a, 'ro', label='Operating Point')
    ax.annotate('Operating Point', (sigma_m, sigma_a), textcoords="offset points", xytext=(10,10), ha='center')

    ax.set_title('Fatigue Failure Criteria')
    ax.set_xlabel('Mean Stress σm (MPa)')
    ax.set_ylabel('Alternating Stress σa (MPa)')
    ax.grid(True)
    ax.legend()
    ax.set_xlim([0, max(UTS, 1)])
    ax.set_ylim([0, max(Se * 1.2, 1)])
    st.pyplot(fig)
else:
    st.warning("Fatigue plot not generated. Please enter valid material strengths and pressure values (e.g., UTS > 100 MPa, Sy > 50 MPa).")

# Output
st.subheader('Fatigue Failure Assessment (Goodman, Soderberg, Gerber, Morrow)')
st.write(pd.DataFrame({
    'Alternating Stress, σa (MPa)': [f"{sigma_a:.2f}"],
    'Mean Stress, σm (MPa)': [f"{sigma_m:.2f}"],
    'Endurance Limit, Se (MPa)': [f"{Se:.2f}"],
    'Goodman Value': [f"{Goodman_Value:.3f}"],
    'Safe (Goodman)': ["Yes" if Goodman_Safe else "No"],
    'Soderberg Value': [f"{Soderberg_Value:.3f}"],
    'Safe (Soderberg)': ["Yes" if Soderberg_Safe else "No"],
    'Gerber Value': [f"{Gerber_Value:.3f}"],
    'Safe (Gerber)': ["Yes" if Gerber_Safe else "No"],
    'Morrow Allowable σa (MPa)': [f"{Morrow_sigma_a_allow:.2f}"],
    'Safe (Morrow)': ["Yes" if Morrow_Safe else "No"]
}))

st.subheader('Von Mises stress of Maximum Operating Pressure')
st.write(pd.DataFrame({'Sigma_VM_Pipe_Max_Operating_Pressure (MPa)': [f"{Sigma_VM_Pipe_Max_Operating_Pressure:.2f}"]}))

st.subheader('Von Mises stress of Minimum Operating Pressure')
st.write(pd.DataFrame({'Sigma_VM_Pipe_Min_Operating_Pressure (MPa)': [f"{Sigma_VM_Pipe_Min_Operating_Pressure:.2f}"]}))

# Reference
st.subheader('Reference')
st.write('Xian-Kui Zhu, A comparative study of burst failure models for assessing remaining strength of corroded pipelines, Journal of Pipeline Science and Engineering 1 (2021) 36 - 50, https://doi.org/10.1016/j.jpse.2021.01.008')

# Assessment links
st.subheader('Assessment')
st.markdown('[Case Study](https://drive.google.com/file/d/1Ako5uVRPYL5k5JeEQ_Xhl9f3pMRBjCJv/view?usp=sharing)', unsafe_allow_html=True)
st.markdown('[Corroded Pipe Burst Data](https://docs.google.com/spreadsheets/d/1YJ7ziuc_IhU7-MMZOnRmh4h21_gf6h5Z/edit?gid=56754844#gid=56754844)', unsafe_allow_html=True)
st.markdown('[Pre-Test](https://forms.gle/wPvcgnZAC57MkCxN8)', unsafe_allow_html=True)
st.markdown('[Post-Test](https://forms.gle/FdiKqpMLzw9ENscA9)', unsafe_allow_html=True)
