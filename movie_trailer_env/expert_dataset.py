import cv2
import numpy as np
import torch
from torchvision import models, transforms

class ExpertDataset:
    def __init__(self, movie_paths, trailer_paths, env):
        self.trajectories = []
        self.env = env
        self.model = models.resnet18(pretrained=True).eval().cuda()  # Utilisation de ResNet pour les embeddings d'images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for movie, trailer in zip(movie_paths, trailer_paths):
            traj = self.extract_traj(movie, trailer)
            self.trajectories.append(traj)

    def get_video_frames(self, video_path):
        # Ouvre la vidéo et récupère toutes les frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def extract_embeddings(self, frames):
        # Convertir les frames en embeddings à l'aide d'un modèle pré-entrainé
        embeddings = []
        for frame in frames:
            frame_tensor = self.transform(frame).unsqueeze(0).cuda()
            with torch.no_grad():
                embedding = self.model(frame_tensor).cpu().numpy()
            embeddings.append(embedding)
        return np.array(embeddings).squeeze()

    def match_trailer_to_movie(self, movie_path, trailer_path):
        # Récupérer les frames du film et du trailer
        movie_frames = self.get_video_frames(movie_path)
        trailer_frames = self.get_video_frames(trailer_path)

        # Extraire les embeddings des frames
        movie_embeddings = self.extract_embeddings(movie_frames)
        trailer_embeddings = self.extract_embeddings(trailer_frames)

        # Calculer les similarités (cosinus) entre les frames du trailer et les frames du film
        trailer_indices = []
        for trailer_embedding in trailer_embeddings:
            similarities = np.dot(movie_embeddings, trailer_embedding)  # Produit scalaire
            best_match_index = np.argmax(similarities)  # Index de la frame du film la plus similaire
            trailer_indices.append(best_match_index)
        
        return trailer_indices

    def extract_traj(self, movie_path, trailer_path):
        env = self.env(movie_path)
        env.reset()
        trailer_indices = self.match_trailer_to_movie(movie_path, trailer_path)
        trajectory = []
        for idx in trailer_indices:
            _, _, _, _ = env.step(idx)
            trajectory.append(idx)
        return trajectory

