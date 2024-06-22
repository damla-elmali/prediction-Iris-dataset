#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import os

class CSVSubscriberNode:
    def __init__(self):
        rospy.init_node('csv_subscriber_node', anonymous=True)
        rospy.Subscriber('csv_data_topic', String, self.callback)

        # Proje kök dizinini belirle
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        input_hidden_path = os.path.join(script_dir, 'data_csv', 'updated_input_hidden_weights.csv')
        hidden_output_path = os.path.join(script_dir, 'data_csv', 'updated_hidden_output_weights.csv')
        
        # Verileri yükle
        try:
            self.input_hidden_weights = pd.read_csv(input_hidden_path)['input_hidden_weights'].values
            self.hidden_output_weights = pd.read_csv(hidden_output_path)['0'].values
        except FileNotFoundError as e:
            rospy.logerr(f"File not found: {e.filename}")
            return

        self.input_hidden_bias = 0.6
        self.hidden_output_bias = 0.7

    def leaky_relu(self, x):
        return x * 0.01 if x < 0 else x

    def predict(self, input_data):
        hidden_layer_input = np.dot(self.input_hidden_weights, input_data) + self.input_hidden_bias
        hidden_layer_output = self.leaky_relu(hidden_layer_input)

        output_layer_input = np.dot(self.hidden_output_weights, hidden_layer_output) + self.hidden_output_bias
        output_layer_output = self.leaky_relu(output_layer_input)

        return output_layer_output

    def process_data(self, csv_data):
        # CSV verisini işleme kodunu buraya ekle
        # Örnek olarak:
        data_array = np.array(csv_data.split(',')[1:-1], dtype=float)  # Son sütunu almama
        prediction = self.predict(data_array)
        predicted_class = 1 if prediction < 1.5 else (2 if prediction < 2.5 else 3)
        return predicted_class

    def callback(self, data):
        rospy.loginfo("Received CSV Data: %s", data.data)
        # CSV'den alınan veriyi işle ve sınıflandır
        processed_data = self.process_data(data.data)
        rospy.loginfo("Processed Data: %s", processed_data)

        # Proje kök dizinini belirle
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Gerçek sınıflar
        true_classes_path = os.path.join(script_dir, 'data_csv', 'target_test.csv')
        input_data_test_path = os.path.join(script_dir, 'data_csv', 'input_data_test.csv')

        try:
            true_classes = pd.read_csv(true_classes_path)['target'].values.astype(int)
            input_data_test = pd.read_csv(input_data_test_path)
        except FileNotFoundError as e:
            rospy.logerr(f"File not found: {e.filename}")
            return

        # Tahminler
        predictions = []
        for index, row in input_data_test.iterrows():
            prediction = self.predict(row.values)
            predicted_class = 1 if prediction < 1.5 else (2 if prediction < 2.5 else 3)
            predictions.append(predicted_class)
        # rospy.loginfo("Predicted Classes: %s", predictions)

        # Doğruluk hesaplama
        accuracy = accuracy_score(true_classes, predictions)
        rospy.loginfo("Accuracy: %s", accuracy)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    csv_subscriber = CSVSubscriberNode()
    csv_subscriber.run()

