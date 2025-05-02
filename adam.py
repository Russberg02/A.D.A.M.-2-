import streamlit as st
import pandas as pd
from pickle import load
import pickle
import numpy as np
import math as m
from PIL import Image
import os
from glob import glob
import matplotlib.pyplot as plt

st.header("Advanced corrodeD pipe structurAl integrity systeM (ADAM)")

st.subheader('Dimensional Parameters')
htp="https://www.researchgate.net/profile/Changqing-Gong/publication/313456917/figure/fig1/AS:573308992266241@1513698923813/Schematic-illustration-of-the-geometry-of-a-typical-corrosion-defect.png"
st.image(htp, caption= "Fig. 1: Schematic illustration of the geometry of a typical corrosion defect.")

st.sidebar.header('User Input Parameters')

def user_input_features():
    pipe_thickness = st.sidebar.number_input('Pipe Thickness, t (mm)', value = 0.01)
    pipe_diameter = st.sidebar.number_input('Pipe Diameter, D (mm)', value = 0.01)
    pipe_length = st.sidebar.number_input('Pipe Length, L (mm)', value = 0.01)
    corrosion_length = st.sidebar.number_input('Corrosion Length, Lc (mm)', value = 0.01)
    corrosion_depth = st.sidebar.number_input('Corrosion Depth, Dc (mm)', value = 0.01)
    Sy = st.sidebar.number_input('Yield Stress, Sy (MPa)', value = 0.01)
    UTS = st.sidebar.number_input('Ultimate Tensile Strength, UTS (MPa)', value = 0.01)
    Maximum_Operating_Pressure = st.sidebar.slider('Maximum Operating Pressure, Pop, Max (MPa)', min_value=0, max_value=50, step=1)
    Minimum_Operating_Pressure = st.sidebar.slider('Minimum Operating Pressure, Pop, Min (MPa)', min_value=0, max_value=50, step=1)

    data = {'t (mm)': pipe_thickness,
            'D (mm)': pipe_diameter,
            'L (mm)': pipe_length,
            'Lc (mm)': corrosion_length,
            'Dc (mm)': corrosion_depth,           
            'UTS (MPa)': UTS,
            'Sy (MPa)': Sy,
            'Pop_Max (MPa)': Maximum_Operating_Pressure,
            'Pop_Min (MPa)': Minimum_Operating_Pressure}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

t=df['t (mm)'].values.item()
D=df['D (mm)'].values.item()
L=df['L (mm)'].values.item()
Lc=df['Lc (mm)'].values.item()
Dc=df['Dc (mm)'].values.item()
UTS=df['UTS (MPa)'].values.item()
Sy=df['Sy (MPa)'].values.item()
Pop_Max=df['Pop_Max (MPa)'].values.item()
Pop_Min=df['Pop_Min (MPa)'].values.item()

st.subheader('Nomenclature')
st.write('t is the pipe thickness; D is the pipe diameter; L is the pipe length (i.e., by default = 1000 mm); Lc is the corrosion length; Dc is the corrosion depth; Sy is the pipe material yield stress; UTS is the pipe material Ultimate Tensile Strength.')

# Calculate burst pressure of intact pipe P Von Mises
Pvm = 4*t*UTS/(m.sqrt(3)*D)

# Calculate burst pressure of intact pipe P Tresca
PTresca = 2*t*UTS/(D)

# Calculate burst pressure of corroded pipe P ASME B31G (2013)
M = m.sqrt(1+0.8*(L/(m.sqrt(D*t)))) #Folias factor

if L < m.sqrt(20*D*t):
    P_ASME_B31G = (2*t*UTS/D)*(1-(2/3)*(Dc/t)/1-(2/3)*(Dc/t)/M)

elif L > m.sqrt(20*D*t):
    P_ASME_B31G = (2*t*UTS/D)*(1-(Dc/t))

# Calculate burst pressure of corroded pipe PDnV
Q = m.sqrt(1+0.31*(Lc)**2/D*t) #Q is the curved fit of FEA results
P_DnV = (2*UTS*t/D-t)*((1-(Dc/t))/(1-(Dc/(t*Q))))

# Calculate burst pressure of corroded pipe P PCORRC Model 
P_PCORRC = (2*t*UTS/D)*(1-Dc/t)

user_input={'t (mm)': "{:.2f}".format(t),
            'D (mm)': "{:.2f}".format(D),
            'L (mm)': "{:.2f}".format(L),
            'Lc (mm)': "{:.2f}".format(Lc),
            'Dc (mm)': "{:.2f}".format(Dc),
            'UTS (MPa)': "{:.2f}".format(UTS),
            'Sy (MPa)': "{:.2f}".format(Sy),
            'Pop_Max (MPa)': "{:.2f}".format(Pop_Max),
            'Pop_Min (MPa)': "{:.2f}".format(Pop_Min)}
user_input_df=pd.DataFrame(user_input, index=[0])
st.subheader('User Input Parameters')
st.write(user_input_df)

# Intact Pipe
calculated_param={'Pvm (MPa)': "{:.2f}".format(Pvm)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Calculated Intact Pipe Burst Pressure via Von Mises')
st.write(calculated_param_df) #Last Output for sigma_VM_Pipe_Min_Operating_Pressure

# Corroded Pipe
calculated_param={'P_ASME_B31G (MPa)': "{:.2f}".format(P_ASME_B31G)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Calculated Corroded Pipe Burst Pressure via ASME_B31G')
st.write(calculated_param_df)

calculated_param={'P_DnV (MPa)': "{:.2f}".format(P_DnV)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Calculated Corrorded Pipe Burst Pressure via DnV')
st.write(calculated_param_df)

calculated_param={'P_PCORRC (MPa)': "{:.2f}".format(P_PCORRC)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Calculated Corrorded Pipe Burst Pressure via PCORRC')
st.write(calculated_param_df)

Pressure = [Pvm, PTresca, P_ASME_B31G, P_DnV, P_PCORRC]
index = ["Pvm (MPa)", "PTresca (MPa)", "P_ASME_B31G (MPa)", "P_DnV (MPa)", "P_PCORRC (MPa)"]
df = pd.DataFrame({"Burst Pressure (MPa)": Pressure}, index=index)

#st.pyplot(df.plot.barh(stacked=True).figure)

# Principle stresses for Maximum Operating Pressure
P1max = Pop_Max*D/(2*t)
P2max = Pop_Max*D/(4*t)
P3max = 0

# Principle stresses for Minimum Operating Pressure
P1min = Pop_Min*D/(2*t)
P2min = Pop_Min*D/(4*t)
P3min = 0

# VM stress Max and Min Operating Pressure
Sigma_VM_Pipe_Max_Operating_Pressure = (1/m.sqrt(2))*((P1max-P2max)**2+(P2max-P3max)**2+(P3max-P1max)**2)**0.5

Sigma_VM_Pipe_Min_Operating_Pressure = 1/m.sqrt(2)*m.sqrt((P1min-P2min)**2+(P2min-P3min)**2+(P3min-P1min)**2)

sigma_a = (Sigma_VM_Pipe_Max_Operating_Pressure - Sigma_VM_Pipe_Min_Operating_Pressure) / 2
sigma_m = (Sigma_VM_Pipe_Max_Operating_Pressure + Sigma_VM_Pipe_Min_Operating_Pressure) / 2
Se = 0.5 * UTS  # Assumed endurance limit

# Goodman Criterion PART 2 Experiment
Goodman_Value = (sigma_a / Se) + (sigma_m / UTS)
Goodman_Safe = Goodman_Value <= 1

# Soderberg Criterion
Soderberg_Value = (sigma_a / Se) + (sigma_m / Sy)
Soderberg_Safe = Soderberg_Value <= 1

# Gerber Criterion
Gerber_Value = (sigma_a / Se) + ((sigma_m / UTS) ** 2)
Gerber_Safe = Gerber_Value <= 1

# Morrow Criterion (more accurate for mean stress effect at high strains)
# Morrow equation: σa = Se*(1 - σm/UTS)
Morrow_sigma_a_allow = Se * (1 - (sigma_m / UTS))
Morrow_Safe = sigma_a <= Morrow_sigma_a_allow



# Display stress parameters first
calculated_param = {
    'Alternating Stress, σa (MPa)': "{:.2f}".format(sigma_a),
    'Mean Stress, σm (MPa)': "{:.2f}".format(sigma_m),
    'Endurance Limit, Se (MPa)': "{:.2f}".format(Se)
}
calculated_param_df = pd.DataFrame(calculated_param, index=[0])
st.subheader('Fatigue Stress Parameters')
st.write(calculated_param_df)

# Display Goodman result
calculated_param = {
    'Goodman Value': "{:.3f}".format(Goodman_Value)
}
calculated_param_df = pd.DataFrame(calculated_param, index=[0])
st.subheader('Calculated Corroded Pipe Burst Pressure via Goodman')
st.write(calculated_param_df)

# Display Soderberg result
calculated_param = {
    'Soderberg Value': "{:.3f}".format(Soderberg_Value)
}
calculated_param_df = pd.DataFrame(calculated_param, index=[0])
st.subheader('Calculated Corroded Pipe Burst Pressure via Soderberg')
st.write(calculated_param_df)

# Display Gerber result
calculated_param = {
    'Gerber Value': "{:.3f}".format(Gerber_Value)
}
calculated_param_df = pd.DataFrame(calculated_param, index=[0])
st.subheader('Calculated Corroded Pipe Burst Pressure via Gerber')
st.write(calculated_param_df)

# Display Morrow result
calculated_param = {
    'Morrow Allowable σa (MPa)': "{:.2f}".format(Morrow_sigma_a_allow)
}
calculated_param_df = pd.DataFrame(calculated_param, index=[0])
st.subheader('Calculated Corroded Pipe Burst Pressure via Morrow')
st.write(calculated_param_df)

calculated_param={'Sigma_VM_Pipe_Max_Operating_Pressure (MPa)': "{:.2f}".format(Sigma_VM_Pipe_Max_Operating_Pressure)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Von Mises stress of Maximum Operating Pressure')
st.write(calculated_param_df)

calculated_param={'Sigma_VM_Pipe_Min_Operating_Pressure (MPa)': "{:.2f}".format(Sigma_VM_Pipe_Min_Operating_Pressure)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Von Mises stress of Minimum Operating Pressure')
st.write(calculated_param_df)

#Stresses = [Sigma_VM_Pipe_Max_Operating_Pressure, Sigma_VM_Pipe_Min_Operating_Pressure, Sy, UTS]
#index = ["Svm_Max (MPa)", "Svm_Min (MPa)", "Yield Stress (MPa)", "UTS (MPa)"]
Stresses = [Sigma_VM_Pipe_Max_Operating_Pressure, Sigma_VM_Pipe_Min_Operating_Pressure, sigma_a, sigma_m, Se, Sy, UTS]
index = ["Svm_Max (MPa)", "Svm_Min (MPa)", "σa (MPa)", "σm (MPa)", "Se (MPa)", "Yield Stress (MPa)", "UTS (MPa)"]
df = pd.DataFrame({"Stresses (MPa)": Stresses}, index=index)

# Graph Layout for Fatigue Failure Criteria
if UTS > 1 and Sy > 1 and Se > 0 and sigma_m >= 0 and sigma_a >= 0:
    sigma_m_range = np.linspace(0, UTS, 500)
    df_plot = pd.DataFrame({
        'sigma_m': sigma_m_range,
        'Goodman': Se * (1 - sigma_m_range / UTS),
        'Soderberg': Se * (1 - sigma_m_range / Sy),

    })

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df_plot['sigma_m'], df_plot['Goodman'], label='Goodman', linestyle='--')
    ax.plot(df_plot['sigma_m'], df_plot['Soderberg'], label='Soderberg', linestyle='-')


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

#st.pyplot(df.plot.barh(color={"Stresses (MPa)": "red"}, stacked=True).figure)

st.subheader('Reference')
st.write('Xian-Kui Zhu, A comparative study of burst failure models for assessing remaining strength of corroded pipelines, Journal of Pipeline Science and Engineering 1 (2021) 36 - 50, https://doi.org/10.1016/j.jpse.2021.01.008')

st.subheader('Assesment')
st.markdown('[Case Study](https://drive.google.com/file/d/1Ako5uVRPYL5k5JeEQ_Xhl9f3pMRBjCJv/view?usp=sharing)', unsafe_allow_html=True)
st.markdown('[Corroded Pipe Burst Data](https://docs.google.com/spreadsheets/d/1YJ7ziuc_IhU7-MMZOnRmh4h21_gf6h5Z/edit?gid=56754844#gid=56754844)', unsafe_allow_html=True)
st.markdown('[Pre-Test](https://forms.gle/wPvcgnZAC57MkCxN8)', unsafe_allow_html=True)
st.markdown('[Post-Test](https://forms.gle/FdiKqpMLzw9ENscA9)', unsafe_allow_html=True)
