import streamlit as st 
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.title("""
Breast Cancer Detector App:
#### Input parameters of the breast to check for abnormality or probability of cancer.         
""")

st.toast("Created by Shriyash Shitole", icon= "🙂")

cancer = datasets.load_breast_cancer()

def inputs():

    st.write("""***""")

    st.write("Input feature values:")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])


    radius_m = col1.text_input("Radius (mean)", value = (28.11 + 6.891)/2)
    radius_e = col1.text_input("Radius (error)", value = (2.873 + 0.112)/2)
    radius_w = col1.text_input("Radius (worst)", value = (36.04 + 7.93)/2)

    texture_m = col2.text_input("Texture (mean)", value = (39.28 + 9.71)/2)
    texture_e = col2.text_input("Texture (error)", value = (4.885 + 0.36)/2)
    texture_w = col2.text_input("Texture (worst)", value = (49.54 + 12.02)/2)

    perimeter_m = col3.text_input("Perimeter (mean)", value = (188.5 + 43.79)/2)
    perimeter_e = col3.text_input("Perimeter (error)", value = (21.98 + 0.757)/2)
    perimeter_w = col3.text_input("Perimeter (worst)", value = (251.2 + 50.41)/2)

    area_m = col4.text_input("Area (mean)", value = (2501.0 + 143.5)/2)
    area_e = col4.text_input("Area (error)", value = (542.2 + 6.802)/2)
    area_w = col4.text_input("Area (worst)", value = (4254.0 + 185.2)/2)

    smoothness_m = col5.text_input("Smoothness (mean)", value = (0.163 + 0.053)/2)
    smoothness_e = col5.text_input("Smoothness (error)", value = (0.031 + 0.002)/2)
    smoothness_w = col5.text_input("Smoothness (worst)", value = (0.223 + 0.071)/2)


    st.write("""
    ***
    """)

    col6, col7, col8, col9, col10 = st.columns([1, 1, 1, 1, 1])

    compactness_m = col6.text_input("Compactness (mean)", value = (0.345 + 0.019)/2)
    compactness_e = col6.text_input("Compactness (error)", value = (0.135 + 0.002)/2)
    compactness_w = col6.text_input("Compactness (worst)", value = (1.058 + 0.027)/2)

    concavity_m = col7.text_input("Concavity (mean)", value = (0.427)/2)
    concavity_e = col7.text_input("Concavity (error)", value = (0.396)/2)
    concavity_w = col7.text_input("Concavity (worst)", value = (1.252)/2)

    concave_points_m = col8.text_input("Concave Points (mean)", value = (0.201)/2)
    concave_points_e = col8.text_input("Concave Points (error)", value = (0.053)/2)
    concave_points_w = col8.text_input("Concave Points (worst)", value = (0.291)/2)

    symmetry_m = col9.text_input("Symmetry (mean)", value = (0.304 + 0.106)/2)
    symmetry_e = col9.text_input("Symmetry (error)", value = (0.079 + 0.008)/2)
    symmetry_w = col9.text_input("Symmetry (worst)", value = (0.664 + 0.156)/2)

    fractal_dim_m = col10.text_input("Fractal Dimention (mean)", value = (0.097 + 0.05)/2)
    fractal_dim_e = col10.text_input("Fractal Dimention (error)", value = (0.03 + 0.001)/2)
    fractal_dim_w = col10.text_input("Fractal Dimention (worst)", value = (0.208 + 0.055)/2)

    st.write("""
    ***
    """)

    udata = {'mean radius': radius_m,
            'mean texture': texture_m,
            'mean perimeter': perimeter_m,
            'mean area': area_m,
            'mean smoothness': smoothness_m,
            'mean compactness': compactness_m,
            'mean concavity': concavity_m,
            'mean concave points': concave_points_m,
            'mean symmetry': symmetry_m,
            'mean fractal dimension': fractal_dim_m,
            'radius error': radius_e,
            'texture error': texture_e,
            'perimeter error': perimeter_e,
            'area error': area_e,
            'smoothness error': smoothness_e,
            'compactness error': compactness_e,
            'concavity error': concavity_e,
            'concave points error': concave_points_e,
            'symmetry error': symmetry_e,
            'fractal dimension error': fractal_dim_e,
            'worst radius': radius_w,
            'worst texture': texture_w,
            'worst perimeter': perimeter_w,
            'worst area': area_w,
            'worst smoothness': smoothness_w,
            'worst compactness': compactness_w,
            'worst concavity': concavity_w,
            'worst concave points': concave_points_w,
            'worst symmetry': symmetry_w,
            'worst fractal dimension': fractal_dim_w}
    
    features = pd.DataFrame(udata, index=["Values Choosen"])
    return features

df = inputs() 

st.write("Selected values by user:")
st.write(df)
st.write("""****""")



# Features -> X, Target -> Y

X = cancer.data
Y = cancer.target

# Fitting

clf = RandomForestClassifier()

clf.fit(X, Y)


# Prediction & Prediciton Probability 

prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)

if prediction[0] == 1:
    st.write("""## :green[Benign] : You are safe!""")
else:
    st.write("""## :red[Malignant] : Cancerous, consult doctor.""")


st.write("""***""")

st.write(f"""
### Probability of :red[Malignant]: {int(prediction_prob[0][0] * 100)}%
### Probability of :green[Benign]: {int(prediction_prob[0][1] * 100)}%        
""")

st.write("""***""")