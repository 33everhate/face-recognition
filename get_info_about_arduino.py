import serial.tools.list_ports
from typing import List


def get_ports() -> None:
    """Выводит информацию о доступных портах."""
    ports = serial.tools.list_ports.comports()  # Получаем список доступных портов.
    for port in ports:
        print(f"Порт: {port.device}")
        print(f"Описание: {port.description}")


def get_serial_parameters(parameter: str) -> List[str]:
    """Возвращает список параметров портов в зависимости от запрошенного параметра(port,description)."""
    ports = serial.tools.list_ports.comports()
    ports_list: List[str] = []
    description_port_list: List[str] = []
    for port in ports:
        ports_list.append(port.device)
        description_port_list.append(port.description)
    if parameter == 'port':
        return ports_list
    elif parameter == 'description':
        return description_port_list
    else:
        return []


if __name__ == "__main__":
    get_ports()