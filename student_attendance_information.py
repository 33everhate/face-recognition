from typing import Dict, List, Tuple

student_name: str = str(input("Введите имя студента: "))

def attendance_information(name: str) -> Tuple[str, List[str]]:
    """Извлекает информацию о посещаемости студента из файла attendance.csv."""
    attendance_list: Dict[str, List[str]] = {}
    with open('attendance.csv') as attendance_file:
        for line in attendance_file.readlines():
            time_attendance: str = line.split(", ")[1][2:-1] # Извлекаем время посещения
            is_attendance: str = line.split(", ")[2][:-2] # Извлекаем статус посещения
            name_student: str = line.split(", ")[0] # Извлекаем имя студента
            information: List[str] = f'{time_attendance} {is_attendance}'.split(' ')
            attendance_list[name_student] = information

        if name not in attendance_list:
            print("Студент не найден")
            exit()

        return name, attendance_list[name]


if __name__ == "__main__":
    student_information: Tuple[str, List[str]] = attendance_information(student_name)
    is_attendance: str = student_information[1][1]
    time_coming: str = student_information[1][0]

    if is_attendance == "True":
        print(f'Имя студента: {student_name}. Пришел: {is_attendance}. Время: {time_coming}')
    else:
        print(f'Имя студента: {student_name}. Пришел: {is_attendance}')