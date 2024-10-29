import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import average_precision_score, recall_score, precision_score
import numpy as np
import time
from transformers import VivitForVideoClassification, VivitImageProcessor
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


#################### Global Parameters ######################

use_distributed = os.environ.get('USE_DIST', '0') == '1'
testing = os.environ.get('TEST', '0') == '1'
testing_TTA = os.environ.get('TEST_TTA', '0') == '1'
if use_distributed:
    model_path = "/cs/home/psyjn4/db/vivit_crash_classifier_768_128_32_one.pth"
# Local
else:
    model_path = "./vivit_crash_classifier_768_128_32_two.pth"
hidden_sizes = [768,512,128]
num_epochs = 20
patience = 12
batch_size = 1
# Feature dimension for attention pooling taking a [3137x768] output and condensing it to a [768xhidden_input]
feature_size = 768

##############################################################

if use_distributed:
    dist.init_process_group(backend="nccl")
    # For distributed training, set the device according to local_rank
    torch.cuda.set_device(dist.get_rank())
    device = torch.device(f"cuda:{dist.get_rank()}")
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


# Initialises Vivit image processor
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")


# Loads the pre-trained weights into this modified model class
vivit_model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400").to(device)




def freeze_early_layers(model, freeze_until_layer='vivit.encoder.layer.11.layernorm_after.bias'):

    freeze = True
    for name, param in model.named_parameters():
        if name == freeze_until_layer:
            freeze = False
        param.requires_grad = not freeze

# Unfreezes at the start of 4th layer set with [128, 64, 32] hidden config by default
# freeze_early_layers(vivit_model, freeze_until_layer="vivit.encoder.layer.4.attention.attention.query.weight")


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_weights = self.query(x) 
        attn_weights = torch.softmax(attn_weights, dim=1)
        # Weighted sum of features
        weighted_feature_sum = torch.sum(attn_weights * x, dim=1)
        return weighted_feature_sum

# Model
class CrashClassifier(nn.Module):
    def __init__(self, vivit_model, hidden_sizes, input_size, output_size=1):
        super(CrashClassifier, self).__init__()
        self.vivit_model = vivit_model
        self.attention_pooling = AttentionPooling(input_dim=input_size)
        # Architecture definition
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            #! https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size),
            # Sigmoid to output a value between 0 and 1
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        # Pass through vivit
        # last layer classifier removed in vivit_model definitions forward

        output = self.vivit_model(pixel_values=x,output_hidden_states=True)
        features = output.hidden_states[-1]
        # Attention pooling expect [bacth size, sequence length, feature dimension] therefore remove [batch size, feature dimentsion] code
        # Attention pooling
        pooled_features = self.attention_pooling(features)
        # Forward pooled features through classifier
        return self.layers(pooled_features)


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

def load_frame(video_path='./datasets/DAD/Dashcam_dataset/videos/testing/positive/000456.mp4', frame_number=59, total_frames=32, flip_horizontal=False):
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
        print("Error: Could not read 32 frames!")
    cap.release()

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


class VideoDatasetViViT(Dataset):
    def __init__(self, positive_dir, negative_dir):
        self.positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.mp4')]
        self.negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.mp4')]
        self.all_files = self.positive_files + self.negative_files
        # Labels: 1 for positive, 0 for negative
        self.labels = [1] * len(self.positive_files) + [0] * len(self.negative_files) 

    def __len__(self):
        return len(self.all_files*20)


    def __getitem__(self, idx):
        # Video index in the total list
        video_idx = idx // 20
        # Is the video horizontally flipped
        frame_set_idx = (idx % 20) // 10
        # Frame index within the set
        within_set_idx = idx % 10

        video_path = self.all_files[video_idx]
        label = self.labels[video_idx]

        flip_horizontal = False
        frame_number = 59 + within_set_idx

        # Horizontally flipped videos
        if frame_set_idx == 1:
            flip_horizontal = True

        # Load and process your video; this function should return a tensor
        frames = load_frame(video_path, frame_number=frame_number, total_frames = 32, flip_horizontal=flip_horizontal)
        # Take in the video frame from dataloader and process it for vivit input
        processed_frames = image_processor(images=frames, return_tensors="pt")#, padding=True, return_overflowing_tokens=False
        #! Extracting features in the model to allow vivit finetuning        
        pixel_values = processed_frames['pixel_values'].squeeze(0).to(device)
        
        return pixel_values, label
    
class FullVideoDatasetViViT(Dataset):
    def __init__(self, positive_dir, negative_dir):
        self.positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.mp4')]
        self.negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.mp4')]
        self.all_files = self.positive_files + self.negative_files
        # Labels: 1 for positive, 0 for negative
        self.labels = [1] * len(self.positive_files) + [0] * len(self.negative_files) 

    def __len__(self):
        return len(self.all_files*68)


    def __getitem__(self, idx):
        # Video index in the total list
        video_idx = idx // 68

        # Frame index within the set
        within_set_idx = idx % 68

        video_path = self.all_files[video_idx]
        label = self.labels[video_idx]

                # If the video is positive, only label the last 10 frames as a crash
        if label == 1 and within_set_idx < 58:
            label = 0

        # Load and process your video; this function should return a tensor
        frames = load_frame(video_path, frame_number=within_set_idx, total_frames = 32, flip_horizontal=False)
        # Take in the video frame from dataloader and process it for vivit input
        processed_frames = image_processor(images=frames, return_tensors="pt")#, padding=True, return_overflowing_tokens=False
        #! Extracting features in the model to allow vivit finetuning        
        pixel_values = processed_frames['pixel_values'].squeeze(0).to(device)
        
        return pixel_values, label





def train_crash_classifier(feature_size, batch_size, vivit_model, model_path, device, patience, num_epochs=1, hidden_sizes=[256,128,32]):
    
    # Loss function and optimisation
    #* NEED TO CALCULATE FEATURE SIZE
    # ViViT feature shape: [1, 3137, 768] or [1, 2409216]
    # ViViT feature shape: [1, 768, 7, 7] or [1, 37632]

    model = CrashClassifier(vivit_model,hidden_sizes, input_size=feature_size).to(device)

    # Binary Cross-Entropy Loss for binary classification tasks
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    # Using the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    # Training loop
    if use_distributed:
        positive_dir = '/cs/home/psyjn4/db/DAD/training/positive'
        negative_dir = '/cs/home/psyjn4/db/DAD/training/negative_subset'
    # Local Training
    else:
        positive_dir = './datasets/DAD/Dashcam_dataset/videos/training/positive'
        negative_dir = './datasets/DAD/Dashcam_dataset/videos/training/negative_subset'

    early_stopping = EarlyStopping(patience)
    if testing_TTA:
        video_dataset = FullVideoDatasetViViT(positive_dir, negative_dir)
    else:
        video_dataset = VideoDatasetViViT(positive_dir, negative_dir)
    if use_distributed:
        sampler = DistributedSampler(video_dataset, shuffle=True)
        dataloader = DataLoader(video_dataset, batch_size=batch_size, sampler=sampler)
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
            if early_stopping.best_score is None:
                print("Initial model load, doing best score calculations")
                val_loss = validate_crash_classifier(batch_size, model, epoch, device)
                # Updates early stopping best score
                early_stopping(val_loss, model, epoch)
            print("Loaded best saved model.")

        for input, label in dataloader:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            # 829 negative training
            # 455 positive training
            label = label.to(device)
            input = input.to(device)
            # with autocast():
            # Forward pass
            output = model(input)
            # Reshapes the label to match the output shape
            # ValueError: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([1, 1, 1])) is deprecated. Please ensure they have the same size.
            label = label.float()
            label = label.view_as(output)
            
            # Calculates loss
            loss = criterion(output, label)
            # Backpropagates the error
            loss.backward()
            
            # scaler.scale(loss).backward()
            # Updates weights
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            # print(f"Done frame {i}")
            done_count += 1
            if done_count % 1000 == 0:
                print(f"Done batch: {done_count % no_of_files}/{no_of_files}")

        # Validation set evaluation
        val_loss = validate_crash_classifier(batch_size, model, epoch, device)

        # Updates early stopping calculations & saves model accordingly
        if early_stopping(val_loss, model, epoch):
            print(f"Early stopping triggered on epoch: {epoch+1}")
            break
        print(f"Done epoch: {epoch+1}")


        print(f"Epoch {epoch+1}")

    torch.save(model.state_dict(), model_path)

def load_crash_classifier(feature_size, vivit_model, model_path,hidden_sizes):

    # Recreatse the model structure
    #! ViViT for testing
    # [1, 2409216] for ViViT
    classifier = CrashClassifier(vivit_model,hidden_sizes, input_size=feature_size).to(device)

    classifier.load_state_dict(torch.load(model_path, map_location=device))

    return classifier


def test_crash_classifier(feature_size, vivit_model, model_path, device, hidden_sizes=[256,128,32]):
    if use_distributed:
        positive_dir = '/cs/home/psyjn4/db/DAD/testing/positive'
        negative_dir = '/cs/home/psyjn4/db/DAD/testing/negative_subset'
    # Local Training
    else:
        positive_dir = './datasets/DAD/Dashcam_dataset/videos/testing/positive'
        negative_dir = './datasets/DAD/Dashcam_dataset/videos/testing/negative_subset'
    
    if testing_TTA:
        print("Running TTA tests")
        video_dataset = FullVideoDatasetViViT(positive_dir, negative_dir)
    else:
        video_dataset = VideoDatasetViViT(positive_dir, negative_dir)
    
    if use_distributed:
        sampler = DistributedSampler(video_dataset, shuffle=False)
        dataloader = DataLoader(video_dataset, batch_size=1, sampler=sampler)
        sampler.set_epoch(0)
    else:
        dataloader = DataLoader(video_dataset, batch_size=1)

    no_of_files = len(dataloader)

    classifier = load_crash_classifier(feature_size, vivit_model, model_path, hidden_sizes)
    # Switches to eval mode if only using for inference
    classifier.eval()

    # Containers for prediction and label
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
            # with autocast():
            # Forward pass
            outputs = classifier(input)

            preds = outputs > 0.5
            all_outputs.extend(outputs.cpu().numpy().flatten())

            # Converts predictions to boolean (0 or 1) and flattens the array
            preds_numeric = preds.cpu().numpy().astype(int).flatten()
            all_preds.extend(preds_numeric)
            all_labels.extend(label.cpu().numpy())

            done_count += 1
            if done_count % 1000 == 0:
                print(f"Done batch: {done_count % no_of_files}/{no_of_files}")         

    print(f"All labels: {all_labels}")
    print(f"All outputs: {all_outputs}")
    # Calculates metrics
    print(f"Average Precision: {average_precision_score(all_labels, all_outputs)}\n")
    print(f"Precision: {precision_score(all_labels, all_preds)}")
    print(f"Recall: {recall_score(all_labels, all_preds)}")
    print(f"Average guess: {np.mean(all_outputs)}")
    # Average guess certainty: 0.0

def validate_crash_classifier(batch_size, model, epoch, device):
    if use_distributed:
        positive_dir = '/cs/home/psyjn4/db/DAD/training/positive_validation'
        negative_dir = '/cs/home/psyjn4/db/DAD/training/negative_validation'
    # Local Training
    else:
        positive_dir = './datasets/DAD/Dashcam_dataset/videos/training/positive_validation'
        negative_dir = './datasets/DAD/Dashcam_dataset/videos/training/negative_validation'
    
    
    if testing_TTA:
        video_dataset = FullVideoDatasetViViT(positive_dir, negative_dir)
    else:
        video_dataset = VideoDatasetViViT(positive_dir, negative_dir)

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
            # with autocast():    
            # Forward pass
            output = model(input)
                
            all_outputs.extend(output.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy())
    
    # Resets the model to training mode
    model.train()
    ap_score = average_precision_score(all_labels, all_outputs)
    print(f"Epoch {epoch+1} Validation Average Precision: {ap_score}")
    return ap_score



if testing:
    # Testing
    start = time.time()
    test_crash_classifier(feature_size, vivit_model, model_path, device, hidden_sizes)
    end = time.time()
    print("Test runtime: ", end-start)
else:
    # Training
    start = time.time()
    train_crash_classifier(feature_size, batch_size, vivit_model, model_path, device, patience, num_epochs, hidden_sizes)
    end = time.time()
    print("Train runtime: ", end-start)
