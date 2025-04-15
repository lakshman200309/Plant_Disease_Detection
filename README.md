# 🌿 Plant Disease Detection System using Deep Learning

This project is an AI-powered web application that allows users to identify plant diseases simply by uploading an image of a plant leaf. It utilizes a deep learning model trained on the PlantVillage dataset to detect 25+ plant diseases and provides users with treatment suggestions and growth tips.

## 🚀 Features

- 🔍 **Automatic Plant Disease Detection**
- 📷 **Image Upload and Analysis**
- 🧠 **Deep Learning with TensorFlow & Keras**
- 🌱 **Treatment and Growth Suggestions**
- 🌐 **Web Interface Built with Flask**
- 📊 **Supports 25+ Leaf Disease Classes**

---

## 🛠️ Tech Stack

| Layer       | Technologies Used                                  |
|-------------|-----------------------------------------------------|
| Frontend    | HTML, CSS (Tailwind), JavaScript                    |
| Backend     | Python, Flask                                       |
| Deep Learning | TensorFlow, Keras, NumPy, OpenCV                 |
| Data        | PlantVillage Dataset + Custom Enhancements          |

---

## 📂 Project Structure

Plant_Disease_Detection/
├── app.py                    # Flask backend application
├── train.py
├── model/plant_disease_model.h5   # Trained CNN model
├── class_indices.json        # Mapping of class indices to disease metadata
|   └── disease_info.json
├── templates/
│   └── index.html            # Frontend interface
|   └── result.html
├── static/                   # (optional) Static files (CSS, JS)
|   └── style.css
├── uploads/                  # Temporarily stores uploaded images
├── README.md                 # Project Documentation


📷 Sample Usage
Go to the home page.

Upload a plant leaf image via the "Upload Image" section.

View the prediction, confidence level, treatment, and care tips.

💻 How to Run Locally
Clone this repository:
git clone https://github.com/lakshman200309/Plant_Disease_Detection.git
cd Plant_Disease_Detection
Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the dependencies:

pip install -r requirements.txt
Run the Flask app:

python app.py
Open your browser and go to http://127.0.0.1:5000

🧪 Supported Plant Diseases (25+)
Some of the diseases this model can detect:

Alternaria Leaf Spot

Anthracnose

Bacterial Leaf Blight

Fusarium Wilt

Mosaic Virus

Downy Mildew

Early Blight

Leaf Curl

Septoria Leaf Spot

Yellow Leaf Curl Virus

And many more...

Each class includes:

✅ Name of the disease

💊 Cure methods (e.g., fungicide recommendations)

🌱 Growth tips to prevent future occurrences

🔮 Future Scope
Mobile App Deployment (Android/iOS)

Real-time detection via camera integration

Geo-tagging for disease mapping

Integration with weather & environmental APIs

Support for additional crops and plant species

📚 Reference
PlantVillage Dataset on Kaggle

Mohanty et al., “Using Deep Learning for Image-Based Plant Disease Detection”, Frontiers in Plant Science, 2016.

Ferentinos K.P., “Deep learning models for plant disease detection”, Computers and Electronics in Agriculture, 2018.

👨‍💻 Author
Tumu Lakshman Prasanna Kumar

💼 AI & ML Enthusiast | Python Developer

## 👤 Connect with Me

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green?style=for-the-badge&logo=google-chrome)](https://lakshman200309.github.io/Personal_Portfolio/)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/tumu-lakshman-prasanna-kumar-a37561270)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/lakshman200309)  
[![Instagram](https://img.shields.io/badge/Instagram-Follow-E4405F?style=for-the-badge&logo=instagram)](https://www.instagram.com/i_m_the_hotstar)  
[![Email](https://img.shields.io/badge/Email-Contact-red?style=for-the-badge&logo=gmail)](mailto:lpkumartumu@gmail.com)  
[![Telegram](https://img.shields.io/badge/Telegram-Chat-2CA5E0?style=for-the-badge&logo=telegram)](https://t.me/+919490200309)

📌 GitHub Repository
🔗 https://github.com/lakshman200309/Plant_Disease_Detection

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
