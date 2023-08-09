import simple_image_download as simp
import os

import utils

my_downloader = simp.Downloader()
my_downloader.directory = 'downloads/'
my_downloader.extensions = '.jpg'
my_downloader.download('Sławomir_Mentzen', limit=200, verbose=False)


# po_subfolders = os.listdir('downloads/pis')
#
# for sub_folder in po_subfolders:
#     try:
#         utils.move_images_up(f'downloads/pis/{sub_folder}')
#         os.rmdir(f'downloads/pis/{sub_folder}')
#     except:
#         pass
