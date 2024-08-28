
# Parkinson's detection using mri

A brief description of what this project does and who it's for

Dataset Link: https://www.kaggle.com/datasets/irfansheriff/parkinsons-brain-mri-dataset

To detect Parkinson's Disease (PD) using MRI scans and the VGG16 model:

1. **Data Preparation**: Gather and preprocess MRI scans, standardizing their size and enhancing the dataset through augmentation techniques.
   
2. **Feature Extraction with VGG16**: Use the pre-trained VGG16 model, known for its effective feature extraction capabilities, and fine-tune it for PD detection by adding custom classification layers on top.

3. **Model Training**: Train the modified VGG16 model on the MRI dataset using techniques like data augmentation, batch normalization, and dropout to prevent overfitting.

4. **Evaluation**: Assess model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC, and analyze the confusion matrix to fine-tune the model further.

5. **Optimization and Deployment**: Fine-tune the model's layers for optimal performance, and deploy it in a clinical setting for assisting in PD diagnosis. Regularly update the model with new data to enhance its diagnostic accuracy.

This approach leverages VGG16's deep learning capabilities for effective PD detection from MRI scans.

**Steps to run: streamlit run streamlit_front_end.py**



