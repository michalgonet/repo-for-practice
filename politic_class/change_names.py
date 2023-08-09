import os
import random
import string


def generate_random_name(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def change_filenames_to_random(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder path")
        return

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        if os.path.isfile(old_path):
            extension = '.jpeg'
            base_name = os.path.splitext(filename)
            new_filename = generate_random_name() + extension
            new_path = os.path.join(folder_path, new_filename)
            while os.path.exists(new_path):
                new_filename = generate_random_name()
                new_path = os.path.join(folder_path, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}")


folder_path = "C://Michal//Programming//Repositories_MG//repo-for-practice//politic_class//downloads//pis"
change_filenames_to_random(folder_path)
