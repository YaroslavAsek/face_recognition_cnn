import torch
from PIL import Image
from torchvision import transforms
from cnn_reco import EmotionCNN, DEVICE, NUM_CLASSES


def pred(image_path: str):
    top_emotion = []
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    model = EmotionCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(r"C:\Recognise_NN\weights\emotion_cnn_best.pth", map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image = Image.open(image_path)
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # add batch dim

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

    for emotion, prob in sorted(zip(EMOTIONS, probs), key=lambda x: -x[1]):
        # print(f"{emotion:<10} {prob*100:5.1f}%  {'█' * int(prob * 30)}")
        top_emotion.append(emotion)
    print(f"Предполагаемая эмоция: {top_emotion[0]}")