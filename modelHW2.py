import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



##### Model #####
def model_DNN(input_shape,num_classes, LR, *layers):
      model = tf.keras.Sequential()
      tf.keras.Input(input_shape)
      
      for elt in layers[0:len(layers)]:
            model.add(tf.keras.layers.Dense(elt, activation='relu'))

      model.add(tf.keras.layers.Flatten())

      model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))

      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
      
      return model



def model_ConvNet(input_shape,num_classes, LR, *layers): 
      model = tf.keras.Sequential()

      tf.keras.Input(input_shape)

      for elt in layers[0:len(layers)-1]:
            model.add(tf.keras.layers.Conv2D(elt,(3,3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))

      model.add(tf.keras.layers.Conv2D(layers[-1],(3,3), activation='relu'))

      model.add(tf.keras.layers.Flatten())

      model.add(tf.keras.layers.Dense(64, activation='relu'))  

      model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))

      model.build((None,28,28,1))
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
      
      return model


def model_ResNet(input_shape,num_classes, nbOfBlock, LR): 
      model = tf.keras.Sequential()
      input_layer = tf.keras.layers.Input(input_shape)
      
      # convolutional layer
      x =tf.keras.layers.Conv2D(64,(3,3))(input_layer)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)

      # max pooling layer to halve the size coming from the previous layer
      x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
      layers = [64,64,256]
      for i in range(nbOfBlock):
            if i == 0:
                  x = ResNetBlockConv(x, layers)
                  x = ResNetBlockId(x, layers)  
            else:
                  l = list(np.array(layers) * (i+1))
                  x = ResNetBlockConv(x, l, 2)
                  x = ResNetBlockId(x, l)   


      x = tf.keras.layers.GlobalAveragePooling2D()(x)

      x = tf.keras.layers.Flatten()(x)

      output_layer = tf.keras.layers.Dense(num_classes,activation='softmax')(x)
      model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
      
      return model

def ResNetBlockId(x, filters, activation='relu'):
    # Shortcut connection
    shortcut = x

    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters[0], kernel_size=1, strides=1, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)

    # Second convolutional layer
    x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    
    # Third convolutional layer
    x = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=1, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add the shortcut to the output
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation(activation)(x)
    return x

def ResNetBlockConv(x, filters, strides = 1, activation='relu'):
    # Shortcut connection
    shortcut = x

    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters[0], kernel_size=1, strides=strides, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)

    # Second convolutional layer
    x = tf.keras.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    
    # Third convolutional layer
    x = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=1, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # shortcut path   
    shortcut = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=strides)(shortcut)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Add the shortcut to the output
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation(activation)(x)
    return x


##### Visualization #####
from sklearn.metrics import auc, RocCurveDisplay, roc_curve
from sklearn.metrics import f1_score

def performance_visualization(history):

      train_loss = history.history['loss']
      val_loss = history.history['val_loss']
      train_acc = history.history['accuracy']
      val_acc = history.history['val_accuracy']

      plt.figure(figsize=(12, 4))
      plt.subplot(1, 2, 1)
      plt.plot(train_loss, label='Training Loss')
      plt.plot(val_loss, label='Validation Loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(train_acc, label='Training Accuracy')
      plt.plot(val_acc, label='Validation Accuracy')
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.legend()

      plt.show()
      return

def F1_visualization(test_labels,predicted_labels):
      # Calculate F1 score for each class
      f1_score_model = f1_score(test_labels, predicted_labels, average=None)

      # Plot the F1 scores for each class
      plt.figure(figsize=(10, 6))
      plt.bar(range(len(f1_score_model)), f1_score_model, tick_label=range(len(f1_score_model)))
      plt.xlabel('Class')
      plt.ylabel('F1 Score')
      plt.title('F1 Score for Each Class')
      plt.show()

      # Calculate the macro-average F1 score
      macro_f1 = np.mean(f1_score_model)
      print(f"Macro-Average F1 Score: {macro_f1}")
      return

def ROC_AUC_visualization(test_labels,predicted_labels):
      fig, axes = plt.subplots(3, 4, figsize=(16, 12))


      for class_id in range(10):
            ax = axes[class_id // 4, class_id % 4]

            y_true_class = (test_labels == class_id).astype(int)
            y_pred = (predicted_labels == class_id).astype(int)

            fpr, tpr, thresholds = roc_curve(y_true_class, y_pred)
            roc_auc = auc(fpr, tpr)

            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"ROC curve for {class_id}")
            roc_display.plot(ax=ax)

      plt.suptitle("RestNet ROC Curve/AUC Score")
      plt.tight_layout()

      plt.show()
      return

def feature_visualization(model , layer_num, features, features_labels):
      feature_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer_num].output)
      # Get the feature maps for the input image
      feature_maps = feature_extractor(features)
      fig, axs = plt.subplots(nrows=feature_maps.shape[-1], ncols=len(feature_maps), figsize=(15, 15))
      for i in range (len(feature_maps)):
            axs[0,i].set_title(str(features_labels[i]))
            for j in range(feature_maps.shape[-1]):
                  axs[j,i].imshow(feature_maps[i, :, :, j], cmap='viridis')
                  axs[j,i].axis('off')
      plt.show()
      return

# Save/load model
import os
import shutil

def save_model(modelName,model):
      name=modelName
      if os.path.exists('models/'+name+'.keras'):
            shutil.rmtree('models/'+name+'.keras')    
      model.save('models/'+name+'.keras')     
      return

def load_model(modelPath):
      importedModel = tf.keras.models.load_model('models/'+modelPath)
      return importedModel