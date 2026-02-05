# Neural Networks Course – Final Projects  
**University of Semnan | M.Sc. Artificial Intelligence**  

**Course:** Neural Networks  
**Instructor:** Dr. Kourosh Kiani  
**Semester:** Master’s Program  

---

## 📌 Project Overview

This repository contains the **final projects** for the *Neural Networks* course, implemented as two separate parts.  
Both projects focus on **predictive modeling using deep neural networks** on a real-world dataset of **used cars in Iran**.

The main objective is to:
- Understand neural network implementation **from scratch**
- Compare it with **modern deep architectures** such as **ResNet**
- Analyze performance differences between classical CNNs and deep residual networks

---

## 📂 Repository Structure

.
├── part1.ipynb   # CNN implementation from scratch  
├── part2.ipynb   # CNN with ResNet architecture  
└── README.md  

---

## 🧠 Dataset

- **Name:** Iran Used Cars Dataset  
- **Source:** Kaggle  
- **Download Method:** kagglehub  

The dataset is automatically downloaded inside each notebook using:

```python
pip install kagglehub
import kagglehub

path = kagglehub.dataset_download("usefashrfi/iran-used-cars-dataset")
```

---

## 🔹 Part 1 – CNN From Scratch

📄 **File:** `part1.ipynb`

### Description
In this part, a **Convolutional Neural Network (CNN)** is implemented **from scratch** using deep learning frameworks.  
The focus is on understanding:
- Network architecture design  
- Feature extraction using convolution layers  
- Training and evaluation process  

### Key Concepts
- Data preprocessing  
- Convolutional layers  
- Activation functions  
- Loss functions  
- Model training and evaluation  

---

## 🔹 Part 2 – CNN with ResNet

📄 **File:** `part2.ipynb`

### Description
This part extends the previous work by using a **ResNet-based architecture**.  
Residual connections are introduced to:
- Improve gradient flow  
- Enable deeper networks  
- Enhance model performance  

### Key Concepts
- Residual blocks  
- Deep CNN architectures  
- Performance comparison with basic CNN  
- Stability and convergence analysis  

---

## 👨‍🎓 Students

- **Ali Fadaeimanesh** (40311422011)  
- **Ali Ebadi** (40311422016)  
- **Fatemeh Tahaei** (40211403004)  
- **Armin Ahangar** (40311422014)  
- **Farbod Sey**  

---

## ⚙️ Requirements

- Python 3.x  
- Jupyter Notebook  
- Required libraries:
  - TensorFlow / PyTorch  
  - NumPy  
  - Pandas  
  - kagglehub  
  - Matplotlib / Seaborn  

---

## 🚀 How to Run

1. Open each notebook in Jupyter or Google Colab  
2. Run all cells sequentially  
3. Dataset will be downloaded automatically  
4. Training and evaluation results will be displayed inline  

---

## 📝 Notes

- Each notebook is **self-contained**  
- No manual dataset download is required  
- Code is structured for **educational clarity**  

---

## 📜 License

This project is developed **for academic purposes only** as part of the Neural Networks course at the University of Semnan.

---

✨ Thank you for reviewing our project.
