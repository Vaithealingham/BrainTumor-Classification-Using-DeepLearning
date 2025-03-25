import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Define class names
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Load the trained model
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))  # Adjust for 4 classes
    model_path = "/content/drive/MyDrive/BrainTumorClsfn/brain_tumor_resnet50.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Define preprocessing (MATCHES TESTING STAGE)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize like in test_transforms
    transforms.ToTensor(),          # Convert to tensor
])

# Streamlit UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan, and the model will classify it.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")  # Convert grayscale to RGB
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    # with torch.no_grad():
    #     output = model(image)
    #     probabilities = torch.nn.functional.softmax(output, dim=1)
    #     confidence, predicted_class = torch.max(probabilities, 1)
    #     prediction = CLASS_NAMES[predicted_class.item()]
    
    with torch.no_grad():
        output = model(image)
        print("Raw Output:", output)  # Debug
        probabilities = torch.nn.functional.softmax(output, dim=1)
        print("Probabilities:", probabilities.numpy())  # Debug
        confidence, predicted_class = torch.max(probabilities, 1)
        print("Predicted Class Index:", predicted_class.item())  # Debug index
        prediction = CLASS_NAMES[predicted_class.item()]

    # Display prediction
    st.success(f"🩺 **Prediction:** {prediction} ({confidence.item()*100:.2f}% confidence)")
