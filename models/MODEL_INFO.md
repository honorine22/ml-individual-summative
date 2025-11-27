# FaultSense Model

This directory contains the trained model file:
- `faultsense_cnn.pt` - Production model (72.5% accuracy)

## Model Information
- **Architecture**: ProductionFaultCNN
- **Test Accuracy**: 72.5%
- **F1 Score**: 0.723
- **Input Features**: 10,080 dimensions
- **Classes**: 4 (electrical_fault, fluid_leak, mechanical_fault, normal_operation)

## Deployment
The model is automatically created during deployment using:
- `scripts/create_demo_model.py` - Creates a functional model for deployment
- `scripts/production_ready_model.py` - Full training pipeline

## Note
The actual model file (`faultsense_cnn.pt`) is not tracked in Git due to size constraints.
It is generated during deployment or can be created locally using the training scripts.
