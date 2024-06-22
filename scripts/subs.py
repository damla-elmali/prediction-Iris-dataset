#!/usr/bin/env python3

import rospy
from std_msgs.msg import String,Float32MultiArray
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

# Leaky ReLU ve türevi
def leaky_relu(x):
    return x * 0.01 if x < 0 else x

def leaky_relu_derivative(x):
    return 0.01 if x < 0 else 1

# Tahmin fonksiyonu
def predict(input_data, input_hidden_weights, input_hidden_bias, hidden_output_weights, hidden_output_bias):
    hidden_layer_input = np.dot(input_hidden_weights, input_data) + input_hidden_bias
    hidden_layer_output = leaky_relu(hidden_layer_input)

    output_layer_input = np.dot(hidden_output_weights, hidden_layer_output) + hidden_output_bias
    output_layer_output = leaky_relu(output_layer_input)

    return output_layer_output

# Verileri yükle
input_hidden_weights = pd.read_csv('updated_input_hidden_weights.csv')['input_hidden_weights'].values
hidden_output_weights = pd.read_csv('updated_hidden_output_weights.csv')['0'].values
input_hidden_bias = 0.6
hidden_output_bias = 0.7

# Test verileri
test_data = pd.read_csv('input_data_test.csv')

# Tahminler
predictions = []
for index, data in test_data.iterrows():
    prediction = predict(data.values, input_hidden_weights, input_hidden_bias, hidden_output_weights, hidden_output_bias)
    predictions.append(prediction)

# Tahminleri inceleme
predictions = np.array(predictions).flatten()
print("Tahminler:")
print(predictions)

# Tahminlerin sınıflara dönüştürülmesi
predicted_classes = []
for pred in predictions:
    if pred < 1.5:
        predicted_classes.append(1)
    elif pred < 2.5:
        predicted_classes.append(2)
    else:
        predicted_classes.append(3)

print("tahmin sınıflar=", predicted_classes)

# Gerçek sınıflar
true_classes = pd.read_csv('target_test.csv')['target'].values.astype(int)
print("\nGerçek Sınıflar:")
print(true_classes)

# Doğruluk hesaplama
accuracy = accuracy_score(true_classes, predicted_classes)
print("\nDoğruluk:", accuracy)

# ROS Subscriber
def callback(data):
    rospy.loginfo("Received CSV Data: %s", data.data)
    # CSV'den alınan veriyi işle ve sınıflandır
    processed_data = process_data(data.data)
    rospy.loginfo("Processed Data: %s", processed_data)

def process_data(csv_data):
    # CSV verisini işleme kodunu buraya ekle
    # Örnek olarak:
    data_array = np.array(csv_data.split(',')[1:-1], dtype=float)  # Son sütunu almama
    prediction = predict(data_array, input_hidden_weights, input_hidden_bias, hidden_output_weights, hidden_output_bias)
    predicted_class = 1 if prediction < 1.5 else (2 if prediction < 2.5 else 3)
    return predicted_class

def listener():
    rospy.init_node('csv_subscriber_node', anonymous=True)
    rospy.Subscriber('csv_data_topic', String, self.callback)
    rospy.spin()

if __name__ == '__main__':
    listener()


class Subscriber:
    def __init__(self):
        self.a = 2
        rospy.Subscriber('csv_data_topic', String, self.callback)

    
    def callback(self,msg):
        print(msg.data)
        a = msg.data

    

    def ca(self):
        print(self.a)
    
    def al(self,c):
        a = c + 10
        return a




if __name__ == '__main__':
    rospy.init_node('csv_subscriber_node', anonymous=True)
    Subscriber()
    print(Subscriber.al(10))
    rospy.spin()