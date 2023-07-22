import sys

class FileResponse:
    def __init__(self):
        self.magic_number = 0
        self.num_items = 0
        self.labels = []

def read_file(file_name):
    response = FileResponse()
    try:
        with open(file_name, "rb") as file:
            response.magic_number = int.from_bytes(file.read(4), byteorder="big")
            response.num_items = int.from_bytes(file.read(4), byteorder="big")

            for _ in range(response.num_items):
                label = int.from_bytes(file.read(1), byteorder="big")
                response.labels.append(label)
    except FileNotFoundError:
        print("Error opening file.")
        sys.exit(1)
    return response

def process_file(file_name):
    response = read_file(file_name)

    print(f"File: {file_name}")
    print(f"Magic Number: {response.magic_number}")
    print(f"Number of Items: {response.num_items}")

    print("Labels:")
    for i, label in enumerate(response.labels):
        print(f"[{i}] {label}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} file1 file2 file3 ...")
        sys.exit(1)

    for file_name in sys.argv[1:]:
        process_file(file_name)

if __name__ == "__main__":
    main()
