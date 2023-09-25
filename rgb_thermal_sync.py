import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse
import shutil
from datetime import datetime
from pathlib import Path 

def main(args_parser):
    root_path = args_parser.root_path
    print("Datapath: ", root_path)

    if not os.path.exists(root_path):
        print("Specified path for data [", root_path,"] does not exist. Please check the path")
        return

    for path, subdirs, files in os.walk(root_path):
        for subd in subdirs:
            if ("g1d" in subd and "_" in subd) or ("g2d" in subd and "_" in subd):
                datapath = os.path.join(path, subd)
                print("Processing", datapath)

                datapath_thermal = os.path.join(datapath, "baseline_thermal")
                datapath_rgb = os.path.join(datapath, "baseline_rgb")

                if not os.path.exists(datapath_thermal):
                    print("Thermal frames not found at: ", datapath_thermal)
                    continue
                elif not os.path.exists(datapath_rgb):
                    print("Color frames not found at: ", datapath_rgb)
                    continue
                else:
                    pass

                savepath_rgb = os.path.join(datapath, "rgb")
                savepath_therm = os.path.join(datapath, "thermal")
                savepath_rgb_video = os.path.join(datapath, "rgb_thermal_combined_visualization")

                list_files_thermal = os.listdir(datapath_thermal)
                # print(list_files_thermal)
                n_frames_thermal = len(list_files_thermal)
                list_files_thermal = sorted(list_files_thermal)

                list_files_rgb = os.listdir(datapath_rgb)
                # print(list_files_rgb)
                n_frames_rgb = len(list_files_rgb)
                list_files_rgb = sorted(list_files_rgb)
                
                therm_images = []
                rgb_images = []
                therm_time_stamp_list = []
                rgb_time_stamp_list = []

                # Thermal - Obtained FPS
                ts_start_thermal = float(Path(list_files_thermal[0]).stem)
                ts_end_thermal = float(Path(list_files_thermal[-1]).stem)
                total_secs = (ts_end_thermal - ts_start_thermal)/1000.0
                print("Datapath - thermal:", datapath_thermal)
                print("Start time (Thermal):", ts_start_thermal)
                print("End time (Thermal):", ts_end_thermal)
                print("Total Frames (Thermal):", n_frames_thermal)
                print("Total seconds (Thermal):", total_secs)
                fps_thermal = int(np.round(float(n_frames_thermal)/total_secs))
                print("Obtained FPS - Thermal: ", fps_thermal)

                # RGB - Obtained FPS
                ts_start_rgb = float(Path(list_files_rgb[0]).stem)
                ts_end_rgb = float(Path(list_files_rgb[-1]).stem)
                total_secs = (ts_end_rgb - ts_start_rgb)/1000.0
                print('*'*50)
                print("Datapath - rgb:", datapath_rgb)
                print("Start time (RGB):", ts_start_rgb)
                print("End time (RGB):", ts_end_rgb)
                print("Total Frames (RGB):", n_frames_rgb)
                print("Total seconds (RGB):", total_secs)
                fps_rgb = int(np.round(float(n_frames_rgb)/total_secs))
                print("Obtained FPS - RGB: ", fps_rgb)


                for i in range(len(list_files_thermal)):
                    fn = list_files_thermal[i]
                    if os.path.isfile(os.path.join(datapath_thermal, fn)):
                        fname = Path(fn).stem
                        ts = float(fname)
                        therm_images.append(fn)
                        therm_time_stamp_list.append(ts)

                for i in range(len(list_files_rgb)):
                    fn = list_files_rgb[i]
                    if os.path.isfile(os.path.join(datapath_rgb, fn)):
                        fname = Path(fn).stem
                        ts = float(fname)
                        rgb_images.append(fn)
                        rgb_time_stamp_list.append(ts)

                therm_time_stamp_list = np.asarray(therm_time_stamp_list)
                rgb_time_stamp_list = np.asarray(rgb_time_stamp_list)


                if not os.path.exists(savepath_rgb_video):
                    os.makedirs(savepath_rgb_video)

                img_width = 640
                img_height = 512

                if not os.path.exists(savepath_therm):
                    os.makedirs(savepath_therm)

                shutil.move(datapath_rgb, savepath_rgb)

                datapath_thermal_stem = Path(datapath_thermal).stem
                datapath_thermal_old = datapath_thermal.replace(datapath_thermal_stem, "thermal_unaligned")
                shutil.move(datapath_thermal, datapath_thermal_old)
                datapath_thermal = datapath_thermal_old

                for i in range(len(rgb_images)):

                    th_ts = rgb_time_stamp_list[i]
                    diff_ts = np.abs(th_ts - therm_time_stamp_list)
                    therm_indx = int(np.argmin(diff_ts))
                    print(i, '--', therm_indx, ':', min(diff_ts), "\r", end="")

                    save_fn, _ = os.path.splitext(rgb_images[i])
                    _, therm_ext = os.path.splitext(therm_images[therm_indx])

                    # Copy time matched thermal frame 
                    shutil.copyfile(os.path.join(datapath_thermal, therm_images[therm_indx]), os.path.join(savepath_therm, therm_images[therm_indx]))
                    shutil.move(os.path.join(savepath_therm, therm_images[therm_indx]), os.path.join(savepath_therm, save_fn + therm_ext))

                    fpath = os.path.join(datapath_thermal, therm_images[therm_indx])
                    th_img = np.fromfile(fpath, dtype=np.uint16, count=img_width * img_height).reshape(img_height, img_width)
                    th_img = (th_img  * 0.04) - 273.15

                    rgb_img = cv2.imread(os.path.join(savepath_rgb, rgb_images[i]))
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    ax[0].imshow(th_img, cmap='magma')
                    ax[0].axis('off')
                    ax[1].imshow(rgb_img)
                    ax[1].axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(savepath_rgb_video, save_fn + '.jpg'))
                    plt.close()
                    fig.clear()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default=None,
                        type=str, dest='root_path', help='The path with RGB and Thermal frames.')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    main(args_parser=args_parser)
