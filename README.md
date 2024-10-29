# ConvNeXt-ViViT-Crash-Anticipation
Dissertation "Evaluating the ConvNeXt-ViViT Hybrid Architecture in Predicting Vehicular Crashes from Video Data"

DOI 10.17605/OSF.IO/BNKW5

## ABSTRACT
This dissertation draws inspiration from existing accident anticipation and prevention systems
implemented in vehicles. The research is driven by the aspiration to develop something of
real-world utility that could aid others in their research or improve existing technologies. The
dissertation aims to develop an advanced predictive model that utilises both ConvNeXt and
ViViT architectures to accurately detect vehicular crashes from dashcam footage, advancing
road safety research. The methodology involved the integration of the ConvNeXt and ViViT
models to capture spatial and temporal features from video data. Data preprocessing included
resizing, normalisation and augmentation to manage class imbalances and enhance training.
Training incorporated strategies like layer freezing, early stopping and pooling to optimise
learning. Adam optimiser and Binary Cross-Entropy Loss were utilised to finetune model
adjustments during training. Validation was conducted through a subset of the Dashcam
Accident Dataset to monitor performance and adjust parameters dynamically. The proposed
ConvNeXt-ViViT hybrid architecture achieves state-of-the-art 72.80% AP and 2.03 sec TTA
when maximising AP for the DAD dataset. The architecture demonstrates its ability to
capture spatial and temporal dynamics from its individual models and suggests that
optimising ViViT temporal feature extraction can lead to improvements in TTA.

## Contributors
- Professor Valerio Giuffrida

## Links:
- [OSF](https://osf.io/bnkw5/)
- [ResearchGate](https://www.researchgate.net/publication/385345429_Evaluating_the_ConvNeXt-ViViT_Hybrid_Architecture_in_Predicting_Vehicular_Crashes_from_Video_Data)
