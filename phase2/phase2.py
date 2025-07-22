from keras_facenet import FaceNet
import numpy as np
from pathlib import Path


# Load processed faces from .npz file
data = np.load("processed_faces.npz", allow_pickle=True)
trainX, trainy = data['trainx'], data['trainy']
testX, testy = data['testx'], data['testy']


# Function to get embeddings from the FaceNet model
def get_embedding(model, face_image):

    face_image = face_image.astype('float32')
    
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std

    face_image = np.expand_dims(face_image, axis=0)    
    embedding = model.embeddings(face_image)
    return embedding[0]

embedder = FaceNet()
embedded_trainX = [get_embedding(embedder, face) for face in trainX]
embedded_testX = [get_embedding(embedder, face) for face in testX]

embedded_trainX = np.asarray(embedded_trainX)
embedded_testX = np.asarray(embedded_testX)


BASE_DIR = Path(__file__).resolve().parent
np.savez_compressed(BASE_DIR / "face_embeddings.npz", trainX=trainX, trainy=trainy, testX=testX, testy=testy)
print("Embeddings saved to 'face_embeddings.npz'")