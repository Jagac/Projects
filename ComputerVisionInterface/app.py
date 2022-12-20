import streamlit as st
import mediapipe as mp 
import cv2
import numpy as np 
import tempfile
import time 
from PIL import Image 


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo.mp4'



st.title('Face Mesh App')


st.markdown (
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350 px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('Parameters')


#resize image so it fits in the page
@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height in None:
        return image

    if width is None:
        r=width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))

    #resize img
    resized = cv2.resize(image,dim,interpolation=inter)
    return resized

app_mode = st.sidebar.selectbox('Choose Mode',
['Image','Real Time Video'])


if app_mode == 'Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
    st.subheader('We are applying Face Mesh on an Image')

    st.sidebar.text('Params for Image')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    max_faces = st.sidebar.number_input('Maximum Number of Faces',value =2,min_value = 1)



    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)


    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:
    


        results = face_mesh.process(image)

        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            #print('face_landmarks:', face_landmarks)

            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        st.subheader('Output Image')

        

        st.image(out_image,use_column_width= True)
    

