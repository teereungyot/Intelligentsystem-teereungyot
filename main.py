import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import tensorflow.keras.backend as K
K.clear_session()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ตั้งค่า UI ของเว็บ
st.set_page_config(page_title="Develop Machine Learning Models & Neural Network Models", layout="wide")


csv_path_air = "data/Cleaned_AirQualityUCI.csv"
csv_path_student = "data/Cleaned_StudentPerformance.csv"

if os.path.exists(csv_path_air):
    df_air = pd.read_csv(csv_path_air).head(10)
    st.dataframe(df_air)

if os.path.exists(csv_path_student):
    df_student = pd.read_csv(csv_path_student).head(10)
    st.dataframe(df_student)    

# ฝัง CSS ลงในโค้ด


# สร้าง Navbar
page = st.sidebar.radio("## Menu", ["🏠 Homepage", "📖 About Machine Learning", "📖 About Neural Network", "⏳ Explain Dataset", "📊 Machine Learning demo", "🤖 Neural Network demo"])

# หน้าแรก
if page == "🏠 Homepage":
    st.markdown("""
        <div style="text-align: left;margin-top: 30px;">
        <div class="set-text-card">
            <h3 style="text-align: center;"> 🔹 Introduction Intelligent System 🔹 </h3>
            <p style="text-align: center;font-size:20px;margin-top: 30px;"> 📌 เว็บไซต์นี้เป็นส่วนหนึ่งของรายวิชา Intelligent System ใช้เพื่อศึกษาการทำ Machine
            learning & Neural Network และศึกษากระบวนการต่างๆ 📌 </p>
            <p style="font-size:20px;margin-top: 30px;">- Air Quality Dataset เป็นชุดข้อมูลที่ใช้สำหรับวิเคราะห์และพยากรณ์คุณภาพอากาศ
            โดยรวบรวมข้อมูลเกี่ยวกับระดับมลพิษทางอากาศและสภาพอากาศในแต่ละช่วงเวลา </p>
            <p style="font-size:20px;">- Student Performance Dataset เป็นชุดข้อมูลที่ใช้สำหรับวิเคราะห์ปัจจัยที่ส่งผลต่อผลการเรียนของนักเรียน 
            และสามารถใช้เพื่อพยากรณ์คะแนนสอบสุดท้าย (G3) ของนักเรียนได้</p> 
        </div>    
            <div style=" font-size: 18px; margin-top: 30px" class="set-text-card">
                <strong style="text-align: left; ">🔹 แหล่งข้อมูล 🔹</strong>
                <p style="text-align: left; margin-top: 10px"> <a href="https://archive.ics.uci.edu/ml/datasets/Air+Quality">UCI Machine Learning Repository - Air Quality Data</a>  </p>
                <p style="text-align: left;"> <a href="https://archive.ics.uci.edu/ml/datasets/Student+Performance">UCI Machine Learning Repository - Student Performance Data </a></p>
                <b style="text-align: left;">🔹 จัดทำโดย 🔹</b>
                <p style="font-size: 18px;margin-top: 10px"> นายเรืองยศ เธียรทิพย์วิบูล รหัสนักศึกษา 6404062636455 </p>
        </div>
    """, unsafe_allow_html=True)
   

# หน้า About1
elif page == "📖 About Machine Learning":
    st.title("แนวทางการพัฒนาโมเดล Machine Learning (Air Quality Dataset)")
    st.write("""

        ### 1. การเตรียมข้อมูล (Data Preparation)
        #### 1.1 การรวบรวมข้อมูล (Data Collection)
        - ใช้ Air Quality Dataset ซึ่งมีตัวแปรเกี่ยวกับมลพิษทางอากาศ เช่น CO (Carbon Monoxide), NO2 (Nitrogen Dioxide), PM10, PM2.5
        - มีตัวแปรด้านสภาพอากาศ เช่น อุณหภูมิ (Temperature), ความชื้น (Humidity), แรงดันอากาศ (Pressure)

        #### 1.2 การทำความสะอาดข้อมูล (Data Cleaning)
        - ตรวจสอบ Missing Values และใช้ค่าเฉลี่ย (Mean) หรือ Interpolation
        - ลบข้อมูลที่ซ้ำซ้อน
        - แปลงวันที่และเวลาให้อยู่ในรูปแบบ datetime
        - จัดการค่าผิดปกติด้วย Interquartile Range (IQR) หรือ Z-score

        #### 1.3 การแปลงและเลือก Features (Feature Selection & Transformation)
        - ใช้ Standardization หรือ Normalization เพื่อปรับขนาดข้อมูล
        - อาจสร้าง Features ใหม่ เช่น ค่าเฉลี่ยของมลพิษในช่วงเวลาต่างๆ (Moving Average)

        ### 2. ทฤษฎีของอัลกอริทึมที่ใช้
        #### 2.1 Linear Regression
        - Linear Regression ใช้สมการเชิงเส้นในรูปแบบ:
          \[ y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon \]
        - ใช้สำหรับพยากรณ์ค่าต่อเนื่อง เช่น ค่ามลพิษในอนาคต

        #### 2.2 Random Forest Regressor
        - Random Forest เป็นอัลกอริทึมแบบ Ensemble Learning ที่รวมหลาย Decision Trees
        - ทำงานโดยสุ่มตัวอย่างข้อมูลและสุ่ม Features เพื่อสร้างต้นไม้หลายต้น จากนั้นเฉลี่ยผลลัพธ์

        ### 3. ขั้นตอนการพัฒนาโมเดล (Model Development)
        #### 3.1 การแบ่งชุดข้อมูล (Train-Test Split)
        - แบ่งข้อมูลเป็น **Training Set 80%** และ **Test Set 20%**
        - ใช้ `train_test_split` จาก `sklearn.model_selection`

        #### 3.2 การปรับขนาดข้อมูล (Feature Scaling)
        - ใช้ **StandardScaler** เพื่อปรับขนาดข้อมูลให้อยู่ในช่วงที่เหมาะสมสำหรับโมเดล

        #### 3.3 การ Train โมเดล (Model Training)
        **Linear Regression:**
        ```python
        from sklearn.linear_model import LinearRegression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_preds = lr_model.predict(X_test_scaled)
        ```
        **Random Forest Regressor:**
        ```python
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        ```

        #### 3.4 การประเมินผลลัพธ์ (Model Evaluation)
        - ใช้ **Mean Absolute Error (MAE)** และ **Root Mean Squared Error (RMSE)** เป็นตัวชี้วัด
        ```python
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        def evaluate_model(y_true, y_pred, model_name):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"{model_name}: MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        evaluate_model(y_test, lr_preds, "Linear Regression")
        evaluate_model(y_test, rf_preds, "Random Forest")
        ```

        #### 3.5 การปรับแต่งโมเดล (Hyperparameter Tuning)
        - ใช้ **GridSearchCV** หรือ **RandomizedSearchCV** เพื่อหาค่าพารามิเตอร์ที่เหมาะสม
        - สำหรับ **Random Forest** สามารถปรับค่า **n_estimators, max_depth, min_samples_split**

        #### 3.6 การบันทึกโมเดล (Model Saving)
        - ใช้ `joblib` ในการบันทึกโมเดล
        ```python
        import joblib
        joblib.dump(rf_model, "random_forest_model.pkl")
        ```
       ### 4. สรุป
        - ทำ Data Cleaning & Feature Engineering เพื่อลดปัญหาค่าผิดพลาดและ Missing Values
        - ใช้ Linear Regression และ Random Forest ในการพยากรณ์ค่ามลพิษทางอากาศ
        - ใช้ตัวชี้วัด MAE และ RMSE เพื่อวัดความแม่นยำของโมเดล
        - สามารถปรับแต่งโมเดลเพิ่มเติมด้วย Hyperparameter Tuning
    """)
# หน้า About Neural Network
elif page == "📖 About Neural Network":
    st.title("แนวทางการพัฒนาโมเดล Neural Network (Student Performance Dataset)")
    st.write("""
     
        


        ### 1. การเตรียมข้อมูล (Data Preparation)
        #### 1.1 การรวบรวมข้อมูล (Data Collection)
        - ใช้ Student Performance Dataset ซึ่งมีตัวแปรเกี่ยวกับผลการเรียน เช่น คะแนน G1, G2, G3
        - มีตัวแปรที่เกี่ยวข้อง เช่น เวลาเรียน, จำนวนพี่น้อง, สถานภาพครอบครัว, เวลาว่าง

        #### 1.2 การทำความสะอาดข้อมูล (Data Cleaning)
        - ตรวจสอบ Missing Values และใช้ค่าเฉลี่ย (Mean) หรือ Interpolation
        - แปลงข้อมูลประเภท Categorical เป็นตัวเลขโดยใช้ One-Hot Encoding
        - ปรับค่าผิดปกติให้อยู่ในช่วงมาตรฐาน

        ### 2. ทฤษฎีของอัลกอริทึมที่ใช้: Multi-Layer Perceptron (MLP)
        Multi-Layer Perceptron (MLP) เป็น Neural Network แบบ Fully Connected ที่มีโครงสร้างดังนี้:

        - **Input Layer:** รับข้อมูลจาก Features
        - **Hidden Layers:** ทำการเรียนรู้แบบไม่เชิงเส้นโดยใช้ Activation Function เช่น ReLU
        - **Output Layer:** ทำนายค่าผลการเรียน (G3)

        สมการพื้นฐานของ MLP:
        \[ y = f(WX + b) \]
        โดยที่:
        - \( X \) คืออินพุต (Features ของนักเรียน)
        - \( W \) คือน้ำหนักที่ใช้เรียนรู้
        - \( b \) คือค่าคงที่ (Bias)
        - \( f \) คือ Activation Function เช่น ReLU หรือ Sigmoid

        ### 3. ขั้นตอนการพัฒนาโมเดล (Model Development)
        #### 3.1 การแบ่งชุดข้อมูล (Train-Test Split)
        - แบ่งข้อมูลเป็น ชุดฝึกสอน (Training Set) 80% และ ชุดทดสอบ (Test Set) 20%
        - ใช้ `train_test_split` จาก `sklearn.model_selection` เพื่อแบ่งข้อมูลแบบสุ่ม

        #### 3.2 การปรับขนาดข้อมูล (Feature Scaling)
        - ใช้ `StandardScaler` เพื่อปรับขนาดข้อมูลให้อยู่ในช่วงที่เหมาะสมสำหรับ Neural Network
        ```python
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        ```
        
        #### 3.3 การสร้างโมเดล MLP (Neural Network)
        ```python
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)  # ทำนายคะแนน G3
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        ```
        
        #### 3.4 การ Train โมเดล
        ```python
        history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_data=(X_test_scaled, y_test))
        ```
        
        #### 3.5 การประเมินผลลัพธ์ (Model Evaluation)
        ```python
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Neural Network - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        ```
        
        #### 3.6 การบันทึกโมเดล (Model Saving)
        ```python
        model.save("student_performance_nn.keras")
        ```
        
        ### 4. สรุป
        - ทำ Data Cleaning & Feature Engineering เพื่อลดปัญหาค่าผิดพลาดและ Missing Values
        - ใช้ **MLP Neural Network** ในการพยากรณ์ผลการเรียนของนักเรียน (**G3**)
        - ใช้ตัวชี้วัด **MAE และ RMSE** เพื่อตรวจสอบความแม่นยำของโมเดล
        - สามารถปรับแต่งโมเดลเพิ่มเติมด้วย **Hyperparameter Tuning และ Regularization (Dropout, Batch Normalization)**
    """)

# หน้าอธิบาย Dataset
elif page == "⏳ Explain Dataset":
    st.title("⏳ Explain Dataset")
    
    dataset_choice = st.radio("## เลือก Dataset:", ["🌍 Air Quality Dataset", "🎓 Student Performance Dataset"])
    
    if dataset_choice == "🌍 Air Quality Dataset":
        st.subheader("🌍 ข้อมูลมลพิษทางอากาศ (Air Quality Dataset)")
        st.write("""
            Dataset นี้ใช้สำหรับการพยากรณ์ค่ามลพิษทางอากาศ (**CO(GT)**)  
            ข้อมูลถูกรวบรวมจากสถานีวัดอากาศจริงในประเทศอิตาลี ☁  
        """)
        # โหลดตัวอย่างข้อมูล
        df_air = pd.read_csv("./Cleaned_AirQualityUCI.csv").head(10)
        st.dataframe(df_air)
        st.write("""
            **Features หลักที่ใช้:**
            - `CO(GT)`: ค่ามลพิษ CO (หน่วย ppm)
            - `PT08.S1(CO)`: ค่าเซ็นเซอร์มลพิษตัวที่ 1
            - `C6H6(GT)`: ค่ามลพิษ Benzene
            - `T`: อุณหภูมิ (°C)
            - `RH`: ความชื้นสัมพัทธ์ (%)
            - `AH`: Absolute Humidity  
        """)
        st.write("**แหล่งข้อมูล:** [UCI Machine Learning Repository - Air Quality Data](https://archive.ics.uci.edu/ml/datasets/Air+Quality)")

    elif dataset_choice == "🎓 Student Performance Dataset":
        st.subheader("🎓 ข้อมูลผลการเรียน (Student Performance Dataset)")
        st.write("""
            Dataset นี้ใช้สำหรับการพยากรณ์คะแนนสอบ **G3** ของนักเรียน 🎓  
            โดยมีปัจจัยต่างๆ เช่น เพศ, อายุ, จำนวนพี่น้อง, เวลาที่ใช้เรียน ฯลฯ  
        """)
        # โหลดตัวอย่างข้อมูล
        df_student = pd.read_csv("./Cleaned_StudentPerformance.csv").head(10)
        st.dataframe(df_student)
        st.write("""
            **Features หลักที่ใช้:**
            - `G1, G2`: คะแนนสอบก่อนหน้า
            - `studytime`: เวลาที่ใช้เรียน
            - `failures`: จำนวนครั้งที่สอบตก
            - `freetime`: เวลาว่างของนักเรียน
            - `goout`: ความถี่ในการออกไปเที่ยว  
        """)
        st.write("**แหล่งข้อมูล:** [UCI Machine Learning Repository - Student Performance Data](https://archive.ics.uci.edu/ml/datasets/Student+Performance)")

# **Machine Learning (Air Quality Prediction)**
elif page == "📊 Machine Learning demo":
    st.title("📊 Air Quality Prediction using Machine Learning")
    
    # โหลดข้อมูล
    df = pd.read_csv("./Cleaned_AirQualityUCI.csv").drop(columns=["Datetime"])
    target_column = "CO(GT)"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # แบ่งข้อมูล
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ฝึกโมเดล
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # ปุ่มพยากรณ์
    if st.button("Predict Air Quality"):
        random_sample = X_test.sample(n=10, random_state=np.random.randint(1000))
        random_sample_scaled = scaler.transform(random_sample)
        prediction_rf = rf_model.predict(random_sample_scaled)
        prediction_lr = lr_model.predict(random_sample_scaled)

        st.write("**Random Forest Prediction (ppm CO(GT))**", prediction_rf)
        st.write("**Linear Regression Prediction (ppm CO(GT))**", prediction_lr)

        # กราฟเส้น
        fig, ax = plt.subplots()
        ax.plot(range(len(prediction_rf)), prediction_rf, marker='o', linestyle='-', label='Random Forest')
        ax.plot(range(len(prediction_lr)), prediction_lr, marker='s', linestyle='-', label='Linear Regression')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Predicted CO(GT) ppm")
        ax.set_title("Air Quality Prediction Comparison")
        ax.legend()
        st.pyplot(fig)
        
        # กำหนดระดับมลพิษ
        avg_prediction = np.mean(prediction_rf)
        if avg_prediction < 2.0:
            air_quality = "🌿 อากาศดี"
        elif 2.0 <= avg_prediction < 5.0:
            air_quality = "😷 อากาศปานกลาง มีมลพิษเล็กน้อย"
        else:
            air_quality = "🚨 อากาศแย่ มีมลพิษสูง!"
        st.subheader(f"สถานะคุณภาพอากาศ: {air_quality}")

# **Neural Network (Student Score Prediction)**
elif page == "🤖 Neural Network demo":
    st.title("🤖 Student Score Prediction using Neural Network")

    # โหลดข้อมูล
    df = pd.read_csv("./Cleaned_StudentPerformance.csv")
    df = pd.get_dummies(df, drop_first=True)
    target_column = "G3"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # แบ่งข้อมูล
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # โมเดล MLP
    K.clear_session()
    mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
    ])
    mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    mlp_model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)
    
    # ปุ่มพยากรณ์
    if st.button("Predict Student Score"):
        random_sample = X_test.sample(n=10, random_state=np.random.randint(1000))
        random_sample_scaled = scaler.transform(random_sample)
        prediction_mlp = mlp_model.predict(random_sample_scaled).flatten()

        st.write("**MLP Prediction (G3 Score)**", prediction_mlp)

        # กราฟเส้น
        fig, ax = plt.subplots()
        ax.plot(range(len(prediction_mlp)), prediction_mlp, marker='o', linestyle='-', color='green', label='MLP Prediction')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Predicted G3 Score")
        ax.set_title("Student Score Prediction")
        ax.legend()
        st.pyplot(fig)
        
        # กำหนดระดับคะแนน
        avg_prediction = np.mean(prediction_mlp)
        if avg_prediction >= 15:
            score_status = "🎓 คะแนนดีมาก"
        elif 10 <= avg_prediction < 15:
            score_status = "📚 คะแนนปานกลาง"
        else:
            score_status = "⚠️ คะแนนต่ำ ควรปรับปรุง"
        st.subheader(f"สถานะผลการเรียน: {score_status}")


