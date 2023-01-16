from torch import cuda

# Global variables
TEST_DIR = './Dataset/Test'
TRAIN_DIR = './Dataset/Train'
EPOCHS = 10
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
BATCH_SIZE = 64  # Number of images feed to the network at one time
LEARNING_RATE = 0.001
MODEL_PATH = '../model/modelFinal.pth'
