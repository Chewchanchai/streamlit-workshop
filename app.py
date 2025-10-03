
import streamlit as st
import joblib
from PIL import Image
import numpy as np

# โหลดโมเดลที่บันทึกไว้
model = joblib.load("svm_image_classifier_model.pkl")

# สร้าง UI ด้วย Streamlit
st.title("Fruit Classifier")
st.write("Upload an image of an apple or orange for classification.")

# อัปโหลดไฟล์รูปภาพ
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# กำหนด label สำหรับผลลัพธ์
class_dict = {0: "Apple", 1: "Orange"}

if uploaded_file is not None:
    # เปิดรูปภาพ
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Resize เป็น 100x100 ตามที่โมเดลเทรนไว้
    image = image.resize((100, 100))

    # แปลงรูปเป็น numpy array
    image_array = np.array(image)

    # Flatten เป็น 1D array และ reshape ให้โมเดลรับได้
    image_array = image_array.flatten().reshape(1, -1)

    # ทำนาย
    prediction = model.predict(image_array)[0]
    prediction_name = class_dict[prediction]

    st.write(f"Prediction: **{prediction_name}**")
