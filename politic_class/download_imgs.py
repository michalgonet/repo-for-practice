import simple_image_download as simp
import os

import utils

pis_filepath = 'Data/PIS.txt'
po_filepath = 'Data/PO.txt'

pis, po = [],[]

with open(pis_filepath, 'r', encoding='utf-8') as file:
    for line in file:
        l = line.strip()
        pis.append(l.replace(' ', '_'))

with open(po_filepath, 'r', encoding='utf-8') as file:
    for line in file:
        l = line.strip()
        po.append(l.replace(' ', '_'))

# for i in po:
#     my_downloader = simp.Downloader()
#     my_downloader.directory = 'downloads/po/'
#     my_downloader.extensions = '.jpg'
#     my_downloader.download(f'{i}_PO', limit=2, verbose=False)

# for i in pis:
#     my_downloader = simp.Downloader()
#     my_downloader.directory = 'downloads/pis/'
#     my_downloader.extensions = '.jpg'
#     my_downloader.download(f'{i}_PIS', limit=2, verbose=False)



# po_subfolders = os.listdir('downloads/pis')
#
# for sub_folder in po_subfolders:
#     try:
#         utils.move_images_up(f'downloads/pis/{sub_folder}')
#         os.rmdir(f'downloads/pis/{sub_folder}')
#     except:
#         pass