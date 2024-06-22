#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import pandas as pd
import random
import time
import os

def publish_csv_data():
    rospy.init_node('csv_publisher_node', anonymous=True)
    pub = rospy.Publisher('csv_data_topic', String, queue_size=10)
    rate = rospy.Rate(1)  # 1Hz

    # Proje kök dizinini belirle
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file_path = os.path.join(script_dir, 'data_csv', 'Iris.csv')

    # CSV dosyasını oku
    try:
        csv_data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        rospy.logerr(f"File not found: {csv_file_path}")
        return

    while not rospy.is_shutdown():
        # Rastgele bir veri seç
        random_index = random.randint(0, len(csv_data) - 1)
        random_data = csv_data.iloc[random_index]

        # Veriyi stringe çevir ve yayımla
        data_str = ','.join(map(str, random_data.values))
        pub.publish(data_str)
        rospy.loginfo("Published CSV Data: %s", data_str)
        
        rate.sleep()
    
if __name__ == '__main__':
    try:
        publish_csv_data()
    except rospy.ROSInterruptException:
        pass

