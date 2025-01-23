import cv2
import numpy as np
import os, sys
import face_recognition
from datetime import datetime
import threading
import time
import serial
import serial.tools.list_ports
from typing import List, Tuple, Dict, Any


class Arduino_use:
    def sent_to_arduino(self) -> None:
        """Отправляет сигнал на Arduino"""
        arduino = serial.Serial('COM1', 9600)
        time.sleep(1.8)
        arduino.write(b'0')
        arduino.close()

    def find_arduino(self) -> bool:
        """Ищет Arduino на доступных COM-портах и запускает его, если он найден."""
        ports = serial.tools.list_ports.comports()
        arduino_ports: List[str] = [] # Список для хранения найденных портов Arduino

        for port in ports:
            if 'Arduino' in port.description or 'CH340' in port.description: #добавить автоввод порта
                arduino_ports.append(port.device)

        if arduino_ports:
            self.sent_to_arduino()
            return True
        else:
            print("Arduino не подключен")
            return False

class AttendanceManager:
    def markAttendance(name: str, attendance_dictionary: Dict[str, List[Any]]) -> None:
        """Отмечает посещаемость для распознанного человека."""
        with open('attendance.csv', 'w+') as at_file:
            now = datetime.now()
            dataString = now.strftime("%H:%M:%S")
            if name != "Unknown":
                status = attendance_dictionary[name][1]
                if status == False:
                    attendance_dictionary[name] = [dataString, True]

                arduino = Arduino_use()
                if arduino.find_arduino():
                    arduino.sent_to_arduino()

                for name_at in attendance_dictionary:
                    at_file.writelines(f"{name_at}, {attendance_dictionary[name_at]}\n")


class FaceRecognition:
    def __init__(self) -> None:
        """Инициализирует систему распознавания лиц."""
        self.face_location: List[Tuple[int, int, int, int]] = [] # Список для хранения координат лиц
        self.face_encoding: List[np.ndarray] = [] # Список для хранения векторов кодирования лиц
        self.face_names: List[str] = [] # Список для хранения имен распознанных лиц
        self.known_face_encodings: List[np.ndarray] = [] # Список для хранения векторов кодирования известных лиц
        self.known_face_names: List[str] = [] # Список для хранения имен известных лиц
        self.attendance_dictionary: Dict[str, List[Any]] = {} # Словарь для хранения данных о посещаемости
        for name_students in os.listdir('students_images'):
            name = name_students.split('.')[0]
            self.attendance_dictionary[name] = ['00:00:00', False]

        self.encode_faces()


    def run(self) -> None:
        """Запускает процесс распознавания лиц."""
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Камера не найдена')

        count_frame = 1
        while True:
            ret, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            self.face_location = face_recognition.face_locations(rgb_small_frame)
            self.face_encoding = face_recognition.face_encodings(rgb_small_frame, self.face_location)
            self.face_names = []

            for face_encoding in self.face_encoding:
                name = 'Unknown'
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                match_index_embedding = np.argmin(face_distances)

                if matches[match_index_embedding]:
                    name = self.known_face_names[match_index_embedding]
                    name = name.split(".")[0]
                    self.face_names.append(name)

                count_frame += 1
                if (count_frame % 3) == 0:
                    mark_attendance = threading.Thread(target=AttendanceManager.markAttendance, args=(name,self.attendance_dictionary))
                    mark_attendance.start()
                    mark_attendance.join()

                self.drawing_rectrangle(frame)

            cv2.imshow('Face_rec',frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def drawing_rectrangle(self,frame):
        for (top, right, bottom, left), name in zip(self.face_location, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
    def encode_faces(self) -> None:
        """Кодирует лица из изображений в каталоге 'students_images'."""
        for image in os.listdir('students_images'):
            face_image = face_recognition.load_image_file(f'students_images/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
            print(self.known_face_names)


if __name__ == '__main__':
    FaceRecognition().run()