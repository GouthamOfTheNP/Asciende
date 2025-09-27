# Asciende

_Asciende_ is an interactive web app that recognizes American Sign Language (ASL) letters in real time and provides **instant feedback** on how to perfect each sign.  
Built with **Streamlit**, **OpenCV**, and a **custom-trained model**, the app also displays a reference image to help users improve their form.

---

##  Features
- **Live Webcam Recognition** – Detects ASL letters continuously from your camera feed.
- **Correction Feedback** – For every letter you sign, the app shows:
  - Text guidance on how to adjust your hand shape.
  - A sample image from our reference dataset for comparison.
- **Streamlit UI** – Clean, responsive interface that runs entirely in the browser.

---

##  How It Works

1. **Webcam Capture** – OpenCV streams frames directly from your webcam.  
2. **Prediction** – A trained machine learning model identifies the current ASL letter.  
3. **Feedback** – The app provides:  
   - Text instructions on how to improve the sign.  
   - A sample image from `asl_dataset/<Letter>/` for visual guidance.  

---

##  Inspiration & Learnings

This project was inspired by the desire to make **ASL learning more accessible**.  

While building Asciende, we learned about:  
- Real-time image processing pipelines.  
- Training and serving custom computer vision models.  
- Streamlit’s powerful yet simple UI components.  

---

##  Next Steps

- Add support for full words/phrases.
- Instead of Custom Letters, use offical ASL 
- Improve accuracy with additional training data.  


## How to Run:





