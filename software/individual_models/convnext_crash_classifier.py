import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import average_precision_score, recall_score, precision_score
import numpy as np
import time
from transformers import ConvNextImageProcessor, ConvNextModel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

#################### Global Parameters ######################
#! To switch between training and testing modes comment the function calls at the end of the file respectively

use_distributed = os.environ.get('USE_DIST', '0') == '1'

#! Uncomment local or cluster paths respectively
# Cluster
model_path = "/cs/home/psyjn4/db/convnext_crash_classifier_convnext_pooled_forward_6_epochs_3_patience_4_step_32_batch_12_frames_flip_768_512_256.pth"
# Local
# model_path = "./convnext_crash_classifier_convnext_pooled_forward_6_epochs_3_patience_4_step_32_batch_12_frames_flip_768_512_256.pth"

hidden_sizes = [768,512,256]
num_epochs = 6
patience = 3
batch_size = 32
feature_size = 768

##############################################################

if use_distributed:
    dist.init_process_group(backend="nccl")
    # For distributed training, set the device according to local_rank
    torch.cuda.set_device(dist.get_rank())
    device = torch.device(f"cuda:{dist.get_rank()}")
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"



# Initialises the model and feature extractor
image_processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224")
# Uses the model in evaluation mode
convnext_model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224").to(device)


class EarlyStopping:
    # patience: How long to wait after last time validation loss improved.
    # delta: Minimum change in the monitored quantity to qualify as an improvement.
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.val_loss_min = -np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, epoch):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            print(f"Epoch: {epoch+1}. Initial performance: {score}")
            return False
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"Epoch: {epoch+1}. No improvement, best: {self.best_score} current: {score}. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            print(f"Epoch: {epoch+1}. Improvement of: {score - self.best_score}")
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            return False


    def save_checkpoint(self, val_loss, model):
        #Saves models score when validation loss decrease.
        self.val_loss_min = val_loss
        torch.save(model.state_dict(), model_path)  

# Model
class CrashClassifier(nn.Module):
    def __init__(self,convnext_model, input_size, hidden_sizes, output_size=1):
        super(CrashClassifier, self).__init__()
        self.convnext_model = convnext_model
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        # Architecture definition
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            #! https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size),
            # Sigmoid to output a value between 0 and 1
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        # Pass through convnext
        output = self.convnext_model(x)
        feature = output.last_hidden_state

        pooled_feature = self.adaptive_pool(feature)
        
        feature = pooled_feature.view(x.size(0), -1)
        # Forward through classifier
        return self.layers(feature)

def load_video(video_path='./datasets/DAD/Dashcam_dataset/videos/testing/positive/000456.mp4'):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        # Breaks the loop if video out of frames (shouldnt happen)
        if not ret:
            break
        # Only starts adding frames when reaching the start frame

        # Converts BGR (OpenCV format) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return (frames,total_frames)

def load_frame(video_path='./datasets/DAD/Dashcam_dataset/videos/testing/positive/000456.mp4', frame_number=89, total_frames=1, flip_horizontal=False):
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # Read frames in a loop
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            if flip_horizontal:
                # Horizontal flip to increase size of testing set
                frame = cv2.flip(frame,1)
            frames.append(frame)
        else:
            print("Error: Could not read frame or end of video reached")
            break

    if len(frames) == total_frames:
        cap.release()
        return frames
    else:
        print("Error: Could not read all frames!")
    cap.release()

class VideoDatasetConvNeXt(Dataset):
    def __init__(self, positive_dir, negative_dir):
        self.positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.mp4')]
        self.negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.mp4')]
        self.all_files = self.positive_files + self.negative_files
        # Labels: 1 for positive, 0 for negative
        self.labels = [1] * len(self.positive_files) + [0] * len(self.negative_files) 

    def __len__(self):
        return len(self.all_files*12)

    def __getitem__(self, idx):
        # Video index in the total list
        video_idx = idx // 12
        # Is the video horizontally flipped
        frame_set_idx = (idx % 12) // 6
        # Frame index within the set
        within_set_idx = idx % 6

        video_path = self.all_files[video_idx]
        label = self.labels[video_idx]
        
        frame_number = 79
        flip_horizontal = False
        # Non flipped videos
        if frame_set_idx == 0:
            frame_number = 79 + within_set_idx * 4
        # Horizontally flipped videos
        else:
            frame_number = 77 + within_set_idx * 4
            flip_horizontal = True

        # Load and process your video; this function should return a tensor
        frame = load_frame(video_path, frame_number=frame_number, total_frames = 1, flip_horizontal=flip_horizontal)
        # Take in the video frame from dataloader and process it for convnext input
        preprocessed_frame = image_processor(images=frame, return_tensors="pt")['pixel_values'][0].to(device)
        #! Extracting features in the model to allow convnext finetuning
        # Apply any additional transformations (e.g., feature extraction)
        # video_tensor = process_frame(frame)
        return preprocessed_frame, label





def train_crash_classifier(batch_size, convnext_model, model_path, device, patience, num_epochs=1, hidden_sizes=[768,512,256]):

    # Loss function and optimisation
    # ViViT feature shape: [1, 3137, 768] or [1, 2409216]
    # ConvNeXt feature shape: [1, 768, 7, 7] or [1, 37632]

    model = CrashClassifier(convnext_model,input_size=feature_size, hidden_sizes=hidden_sizes).to(device)

    # Binary Cross-Entropy Loss for binary classification tasks
    criterion = nn.BCELoss()

    # Using the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    # Training loop
    positive_dir = '/cs/home/psyjn4/db/DAD/training/positive'
    negative_dir = '/cs/home/psyjn4/db/DAD/training/negative_subset'
    # Local Training
    # positive_dir = './datasets/DAD/Dashcam_dataset/videos/training/positive'
    # negative_dir = './datasets/DAD/Dashcam_dataset/videos/training/negative_subset'
    early_stopping = EarlyStopping(patience)

    video_dataset = VideoDatasetConvNeXt(positive_dir, negative_dir)
    if use_distributed:
        sampler = DistributedSampler(video_dataset, shuffle=True)
        dataloader = DataLoader(video_dataset, batch_size=batch_size, sampler=sampler) #, batch_size=4, shuffle=True)
    else:
        dataloader = DataLoader(video_dataset, batch_size=batch_size) #, batch_size=4, shuffle=True)

    no_of_files = len(dataloader)
    done_count = 0
    for epoch in range(num_epochs):
        if use_distributed:
            sampler.set_epoch(epoch)
        # Loads best saved model
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded best saved model.")
        for input, label in dataloader:
            torch.cuda.empty_cache()
            label = label.to(device)
            input = input.to(device)

            # Clears gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input)
            # Reshapes the label to match the output shape
            label = label.float()
            label = label.view_as(outputs)
            # Calculates loss
            loss = criterion(outputs, label)
            # Backpropagates the error
            loss.backward()
            # Updates weights
            optimizer.step()
            done_count += 1
            if done_count % 10 == 0:
                print(f"Done batch: {done_count % no_of_files}/{no_of_files}")
        # Validation set evaluation
        val_loss = validate_crash_classifier(batch_size, model, epoch, device)

        # Updates early stopping calculations & saves model accordingly
        if early_stopping(val_loss, model, epoch):
            print(f"Early stopping triggered on epoch: {epoch+1}")
            break
        print(f"Done epoch: {epoch+1}")

    # Save the model
    torch.save(model.state_dict(), model_path)

def load_crash_classifier(convnext_model, hidden_sizes, model_path):
    # Recreatse the model structure
    #! ConvNeXt for testing
    # [1, 37632] for ConvNeXt
    classifier = CrashClassifier(convnext_model, input_size=feature_size, hidden_sizes=hidden_sizes).to(device)

    # Loads the saved state dict into the model
    classifier.load_state_dict(torch.load(model_path, map_location=device))

    return classifier


def test_crash_classifier(convnext_model, model_path, device, hidden_sizes=[768,512,256]):
    positive_dir = '/cs/home/psyjn4/db/DAD/testing/positive'
    negative_dir = '/cs/home/psyjn4/db/DAD/testing/negative_subset'
    # Local Training
    # positive_dir = './datasets/DAD/Dashcam_dataset/videos/testing/positive'
    # negative_dir = './datasets/DAD/Dashcam_dataset/videos/testing/negative_subset'
    
    video_dataset = VideoDatasetConvNeXt(positive_dir, negative_dir)
    if use_distributed:
        sampler = DistributedSampler(video_dataset, shuffle=False)
        dataloader = DataLoader(video_dataset, batch_size=32, sampler=sampler)
        sampler.set_epoch(0)
    else:
        dataloader = DataLoader(video_dataset, batch_size=32)

    no_of_files = len(dataloader)
    classifier = load_crash_classifier(convnext_model, hidden_sizes, model_path)

    # Switches to eval mode if only using for inference
    classifier.eval()

    # Containers for predictions and labels
    all_preds = []
    all_labels = []
    done_count = 0
    all_outputs = []
    # No need to track gradients
    with torch.no_grad():
        for input, label in dataloader:
            torch.cuda.empty_cache()
            input = input.to(device)
            label = label.to(device)
            
            # Forward pass
            outputs = classifier(input)
            
            # Converts outputs to predictions (0 or 1 based on a 0.5 threshold)
            preds = outputs > 0.5
            all_outputs.extend(outputs.cpu().numpy().flatten())

            # Converts predictions to boolean (0 or 1) and flattens the array
            preds_numeric = preds.cpu().numpy().astype(int).flatten()
            all_preds.extend(preds_numeric)
            all_labels.extend(label.cpu().numpy())

            done_count += 1
            print(f"Done batch: {done_count}/{no_of_files}")            

    print(f"All labels: {all_labels}")
    print(f"All outputs: {all_outputs}")
    # Calculates metrics
    print(f"Average Precision: {average_precision_score(all_labels, all_outputs)}\n")
    print(f"Precision: {precision_score(all_labels, all_preds)}")
    print(f"Recall: {recall_score(all_labels, all_preds)}")
    print(f"Average guess: {np.mean(all_outputs)}")
    # Average guess certainty: 0.0

def validate_crash_classifier(batch_size, model, epoch, device):
    positive_dir = '/cs/home/psyjn4/db/DAD/training/positive_validation'
    negative_dir = '/cs/home/psyjn4/db/DAD/training/negative_validation'
    # Local Training
    # positive_dir = './datasets/DAD/Dashcam_dataset/videos/training/positive_validation'
    # negative_dir = './datasets/DAD/Dashcam_dataset/videos/training/negative_validation'
    
    video_dataset = VideoDatasetConvNeXt(positive_dir, negative_dir)
    if use_distributed:
        sampler = DistributedSampler(video_dataset, shuffle=False)
        dataloader = DataLoader(video_dataset, batch_size=batch_size, sampler=sampler)
        sampler.set_epoch(epoch)
    else:
        dataloader = DataLoader(video_dataset, batch_size=batch_size)

    # Set the model to eval mode, not updating gradients anyways
    model.eval()

    # Containers for predictions and labels
    all_outputs = []
    all_labels = []
    # No need to track gradients
    with torch.no_grad():
        for input, label in dataloader:
            torch.cuda.empty_cache()
            input = input.to(device)
            label = label.to(device)
            # Forward pass
            output = model(input)
            

            all_outputs.extend(output.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy())
    
    # Resets the model to training mode
    model.train()
    ap_score = average_precision_score(all_labels, all_outputs)       
    print(f"Epoch {epoch+1} Validation Average Precision: {ap_score}")
    return ap_score

# Training
start = time.time()
train_crash_classifier(batch_size, convnext_model, model_path, device, patience, num_epochs, hidden_sizes)
end = time.time()
print("Train runtime: ", end-start)

# Training
start = time.time()
test_crash_classifier(convnext_model, model_path, device, hidden_sizes)
end = time.time()
print("Train runtime: ", end-start)
