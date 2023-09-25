# This Python file uses the following encoding: utf-8
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pathlib import Path
import sys
import threading
import time
import datetime
import numpy as np
from copy import deepcopy
import argparse

import cv2
from seg.inference import seg_inference
from utils.flircamera import CameraManager as tcam

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QFile, QObject, Signal, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
from PIL import Image


global thermal_camera_connect_status, rgb_camera_connect_status, acquisition_status, keep_acquisition_thread, acquisition_thread_rgb_live
global recording_status, save_path, num_frames, subdir_path, num_frames_rgb
global capture_thermal, capture_rgb, run_inference, pseudocolor
global use_lock_rgb_capture_with_thermal, enable_rgb_capture_with_lock
global rgb_inf_img, thermal_inf_img

capture_rgb = False
capture_thermal = False
run_inference = True
acquisition_status = False
recording_status = False
thermal_camera_connect_status = False
rgb_camera_connect_status = False
keep_acquisition_thread = True
acquisition_thread_rgb_live = False
use_lock_rgb_capture_with_thermal = False
enable_rgb_capture_with_lock = False

rgb_inf_img = None
thermal_inf_img = None

pseudocolor = 'coolwarm'
save_path = "recorded_frames"
num_frames = 0
num_frames_rgb = 0

class RGBTCam(QWidget):
    def __init__(self, args_parser):
        super(RGBTCam, self).__init__()
        self.load_ui(args_parser)

    def load_ui(self, args_parser):
        self.args_parser = args_parser
        
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "rgbt_form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        
        self.tcamObj = tcam()
        self.pseudocolor_list = ['CoolWarm', 'GNUPlot2', 'Gray', 'Magma',
                                 'Nipy_Spectral', 'Pink', 'Plasma', 'Prism', 'Rainbow', 'Seismic', 'Terrain']

        self.camObj = None
        self.study_name = ""
        self.participant_id = ""
        self.segObj = seg_inference(ckpt_path="seg/ckpt/model.pth")
        self.VOC_COLORMAP = [0, 0, 0, 200, 128, 128, 0, 128, 0, 128,200,128, 128,128,200]

        # input_size = self.configer.get('test', 'data_transformer')['input_size']
        self.seg_img_width = 640 #input_size[0]
        self.seg_img_height = 512 #input_size[1]

        self.ui.checkBox_Th.stateChanged.connect(lambda:self.btnstate(self.ui.checkBox_Th))
        self.ui.checkBox_RGB.stateChanged.connect(lambda:self.btnstate(self.ui.checkBox_RGB))
        self.ui.text_study.textChanged.connect(self.update_study_name)
        self.ui.text_pid.textChanged.connect(self.update_pid)

        self.ui.connectButton_Thermal.pressed.connect(self.scan_and_connect_thermal_camera)
        self.ui.comboBox_RGB_Cam.currentIndexChanged.connect(self.scan_and_connect_rgb_camera)
        self.ui.comboBox_vis.currentIndexChanged.connect(self.update_thermal_pseudocolor)

        self.ui.acquireButton.pressed.connect(self.control_acquisition)
        self.ui.recordButton.pressed.connect(self.control_recording)

        # self.ui.browseButton.pressed.connect(self.browse_recorded_dir)
        self.ui.acquireButton.setEnabled(False)
        self.ui.recordButton.setEnabled(False)

        self.imgAcqLoop = threading.Thread(name='imgAcqLoop', target=thermal_capture_frame_thread, daemon=True, args=(
            self.tcamObj, self.process_inference, self.updatePixmap, self.updateLog))
        self.imgAcqLoop.start()

        ui_file.close()

    def closeEvent(self, event):
        global thermal_camera_connect_status, acquisition_status, keep_acquisition_thread, acquisition_thread_rgb_live
        keep_acquisition_thread = False
        acquisition_thread_rgb_live = False
        print("Please wait while camera is released...")
        time.sleep(0.5)
        if thermal_camera_connect_status and acquisition_status:
            self.tcamObj.release_camera(acquisition_status)

    def btnstate(self, b):
        global capture_thermal, capture_rgb, run_inference

        if b.text() == "Capture Thermal":
            if b.isChecked() == True:
                capture_thermal = True
                self.updateLog(b.text()+" is selected")
                self.ui.connectButton_Thermal.setEnabled(True)
            else:
                capture_thermal = False
                self.updateLog(b.text()+" is deselected")
                self.ui.connectButton_Thermal.setEnabled(False)
				
        if b.text() == "Capture RGB":
            if b.isChecked() == True:
                capture_rgb = True
                self.updateLog(b.text()+" is selected")
                self.ui.comboBox_RGB_Cam.setEnabled(True)
            else:
                capture_rgb = False
                self.updateLog(b.text()+" is deselected")
                self.ui.comboBox_RGB_Cam.setEnabled(False)

        if b.text() == "Run Inference":
            if b.isChecked() == True:
                run_inference = True
                self.updateLog(b.text()+" is selected")
            else:
                run_inference = False
                self.updateLog(b.text()+" is deselected")
            print("Run Inference", run_inference)

        self.enable_acquisition()


    def scan_and_connect_rgb_camera(self, indx):

        global acquisition_thread_rgb_live, rgb_camera_connect_status

        if acquisition_thread_rgb_live:
            acquisition_thread_rgb_live = False
            time.sleep(1)
        try:
            cam_index = int(indx)
            self.camObj = cv2.VideoCapture(cam_index)
            rgb_camera_connect_status = True
        except:
            rgb_camera_connect_status = False

        if rgb_camera_connect_status:
            if not acquisition_thread_rgb_live:
                acquisition_thread_rgb_live = True
                self.imgAcqLoop_rgb = threading.Thread(name='imgAcqLoop_rgb', target=rgb_capture_frame_thread, daemon=True, args=(
                    self.camObj, self.process_inference, self.updateRGBPixmap, self.updateLog))
                self.imgAcqLoop_rgb.start()

        self.enable_acquisition()


    def update_thermal_pseudocolor(self, indx):
        global pseudocolor
        pseudocolor = self.pseudocolor_list[indx].lower()


    def update_study_name(self, text):
        self.study_name = self.ui.text_study.text()


    def update_pid(self, text):
        self.participant_id = self.ui.text_pid.text()


    def scan_and_connect_thermal_camera(self):
        global acquisition_status, thermal_camera_connect_status

        if thermal_camera_connect_status == False:
            if self.tcamObj.get_camera():
                self.cam_serial_number, self.cam_img_width, self.cam_img_height = self.tcamObj.setup_camera()
                if "error" not in self.cam_serial_number.lower():
                    self.ui.connectButton_Thermal.setText("Disconnect Camera")
                    self.updateLog("Camera Serial Number: " + self.cam_serial_number)
                    thermal_camera_connect_status = True
                    self.tcamObj.begin_acquisition()
                    self.img_width = self.seg_img_width
                    self.img_height = self.seg_img_height
                else:
                    self.updateLog("Error Setting Up Camera: " + self.cam_serial_number)

        else:
            # self.ui.acquireButton.setEnabled(False)
            thermal_camera_connect_status = False
            self.tcamObj.end_acquisition()
            self.updateLog('Thermal camera disconnected')
            self.ui.connectButton_Thermal.setText("Scan and Connect Thermal Camera")            

        self.enable_acquisition()


    def enable_acquisition(self):
        
        global capture_rgb, capture_thermal, thermal_camera_connect_status, rgb_camera_connect_status
        enable_acquisition = True
        
        if capture_thermal:
            if thermal_camera_connect_status:
                enable_acquisition = enable_acquisition and True
            else:
                enable_acquisition = False
        
        if capture_rgb:
            if rgb_camera_connect_status:
                enable_acquisition = enable_acquisition and True
            else:
                enable_acquisition = False
            
        if enable_acquisition:
            self.ui.acquireButton.setEnabled(True)
        else:
            self.ui.acquireButton.setEnabled(False)


    def control_acquisition(self):
        global acquisition_status, thermal_camera_connect_status, rgb_camera_connect_status, capture_rgb, capture_thermal

        if acquisition_status == False:
            self.ui.acquireButton.setText('Stop Live Streaming')
            acquisition_status = True
            self.ui.recordButton.setEnabled(True)
            self.updateLog("Acquisition started")

        else:
            acquisition_status = False
            self.ui.recordButton.setEnabled(False)
            self.ui.acquireButton.setText('Start Live Streaming')
            self.updateLog("Acquisition stopped")

    def control_recording(self):
        global save_path, recording_status, subdir_path

        if recording_status == False:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
            subdir_path = os.path.join(save_path, self.study_name, self.participant_id, timestamp)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
            self.ui.recordButton.setText('Stop Recording')
            recording_status = True
            self.updateLog("Recording started")

        else:
            recording_status = False
            self.ui.recordButton.setText('Record Frames')
            self.updateLog("Recording stopped")

    def updatePixmap(self, data_list):
        canvas, width, height = data_list        
        qimg1 = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)
        self.ui.pix_label.setPixmap(QPixmap.fromImage(qimg1))


    def updateRGBPixmap(self, data_list):
        rgb_matrix, rgb_ret = data_list

        if rgb_ret:
            rgbImage = cv2.cvtColor(rgb_matrix, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            qimg2 = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            qimg2 = qimg2.scaled(640, 480, Qt.KeepAspectRatio)
            self.ui.pix_label_rgb.setPixmap(QPixmap.fromImage(qimg2))


    def process_inference(self, data_list):
        global rgb_inf_img, thermal_inf_img
        img_matrix, capture_status, img_type = data_list

        if capture_status:
            if img_type == "rgb":
                rgb_inf_img = img_matrix
            elif img_type == "thermal":
                pass
                # thermal_inf_img = img_matrix

            # if rgb_inf_img != None and thermal_inf_img != None:
            #     pred_seg = self.segObj.run_inference(rgb_inf_img, thermal_inf_img)
            #     rgb_inf_img = None
            #     thermal_inf_img = None

            if np.any(rgb_inf_img) == None:
                pass
            else:
                pred_seg = self.segObj.run_inference(rgb_inf_img)

                pred_seg = Image.fromarray(pred_seg.astype('uint8'))
                pred_seg.putpalette(self.VOC_COLORMAP)
                mask = pred_seg.convert('RGBA')
                mask.putalpha(128)
                original_img = Image.fromarray(rgb_inf_img.astype('uint8')).convert('RGBA')
                original_img = original_img.resize(mask.size)
                result_image = Image.alpha_composite(original_img, mask)

                rgbImage = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                qimg2 = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                qimg2 = qimg2.scaled(640, 480, Qt.KeepAspectRatio)
                self.ui.pix_label_rgb.setPixmap(QPixmap.fromImage(qimg2))

                rgb_inf_img = None


    def updateLog(self, message):
        self.ui.log_label.setText(message)


# Setup a signal slot mechanism, to send data to GUI in a thread-safe way.
class Communicate(QObject):
    data_signal = Signal(list)
    inf_data_signal = Signal(list)
    data_signal_rgb = Signal(list)

    save_signal = Signal(np.ndarray)
    save_signal_rgb = Signal(np.ndarray)

    status_signal = Signal(str)

def save_frame_thermal(thermal_matrix):
    global num_frames, subdir_path
    num_frames += 1
    utc_sec = str((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()).replace('.', '_')
    np.save(os.path.join(subdir_path, f'{num_frames:05d}' + '_' + utc_sec + '.npy'), thermal_matrix)
    

def save_frame_rgb(rgb_matrix):
    global num_frames_rgb, subdir_path, use_lock_rgb_capture_with_thermal

    if not use_lock_rgb_capture_with_thermal:
        num_frames_rgb += 1
    else:
        num_frames_rgb = num_frames

    utc_sec = str((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()).replace('.', '_')
    cv2.imwrite(os.path.join(subdir_path, f'{num_frames_rgb:05d}' + '_' + utc_sec + '.png'), rgb_matrix)


def rgb_capture_frame_thread(camObj, process_inference, updateRGBPixmap, updateLog):

    global acquisition_status, rgb_camera_connect_status, acquisition_thread_rgb_live, run_inference
    global recording_status, use_lock_rgb_capture_with_thermal, enable_rgb_capture_with_lock

    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.status_signal.connect(updateLog)
    mySrc.save_signal_rgb.connect(save_frame_rgb)

    if run_inference:
        mySrc.data_signal_rgb.connect(process_inference)
    else:
        mySrc.data_signal_rgb.connect(updateRGBPixmap)

    while True:
        if acquisition_thread_rgb_live:
            if rgb_camera_connect_status and acquisition_status:
                if not use_lock_rgb_capture_with_thermal or (use_lock_rgb_capture_with_thermal and enable_rgb_capture_with_lock):
                    t1 = time.time()
                    info_str = ""
                    rgb_ret, rgb_matrix = camObj.read()

                    # rgb_matrix = cv2.rotate(rgb_matrix, cv2.ROTATE_180)

                    if rgb_ret:
                        if use_lock_rgb_capture_with_thermal:
                            enable_rgb_capture_with_lock = False
                        rgb_matrix_vis = deepcopy(rgb_matrix)
                        if run_inference:
                            mySrc.data_signal_rgb.emit([rgb_matrix_vis, rgb_ret, "rgb"])
                        else:
                            mySrc.data_signal_rgb.emit([rgb_matrix_vis, rgb_ret])

                        if recording_status:
                            mySrc.save_signal_rgb.emit(rgb_matrix)

                    info_str = "RGB Frame acquisition status: " + str(rgb_ret) + "; " + info_str
                    # time.sleep(0.05)
                    t2 = time.time()
                    t_elapsed = str(t2 - t1)
                    info_str = info_str + "; total_time_per_frame RGB: " + t_elapsed
                    mySrc.status_signal.emit(info_str)
                    # time.sleep(0.05)

            else:
                time.sleep(0.25)
        else:
            mySrc.status_signal.emit("Acquisition thread termination. Please restart the application...")
            break


def thermal_capture_frame_thread(tcamObj, process_inference, updatePixmap, updateLog):

    global acquisition_status, thermal_camera_connect_status, keep_acquisition_thread, run_inference
    global recording_status, pseudocolor, use_lock_rgb_capture_with_thermal, enable_rgb_capture_with_lock

    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.status_signal.connect(updateLog)
    mySrc.save_signal.connect(save_frame_thermal)

    mySrc.inf_data_signal.connect(process_inference)
    mySrc.data_signal.connect(updatePixmap)
 
    while True:
        if keep_acquisition_thread:
            if thermal_camera_connect_status and acquisition_status:
                t1 = time.time()
                info_str = ""
                thermal_matrix, frame_status = tcamObj.capture_frame()
                # thermal_matrix = cv2.rotate(thermal_matrix, cv2.ROTATE_180)

                if frame_status == "valid" and thermal_matrix.size > 0:
                    if use_lock_rgb_capture_with_thermal:
                        enable_rgb_capture_with_lock = True
                    min_temp = np.round(np.min(thermal_matrix), 2)
                    max_temp = np.round(np.max(thermal_matrix), 2)
 
                    if recording_status:
                        mySrc.save_signal.emit(thermal_matrix)
                        info_str = "[Min Temp, Max Temp] = " + str([min_temp, max_temp])

                    if run_inference:
                        inf_therm_img = deepcopy(thermal_matrix)
                        mySrc.inf_data_signal.emit([inf_therm_img, True, "thermal"])

                    fig = Figure(tight_layout=True)
                    canvas = FigureCanvas(fig)
                    ax = fig.add_subplot(111)
                    ax.imshow(thermal_matrix, cmap=pseudocolor)
                    ax.set_axis_off()
                    canvas.draw()
                    width, height = fig.figbbox.width, fig.figbbox.height
                    mySrc.data_signal.emit([canvas, width, height])
                
                info_str = "Frame acquisition status: " + frame_status + "; " + info_str                
                # time.sleep(0.05)
                t2 = time.time()
                t_elapsed = str(t2 - t1)
                info_str = info_str + "; total_time_per_frame: " + t_elapsed
                mySrc.status_signal.emit(info_str)
                # time.sleep(0.05)

            else:
                time.sleep(0.25)
        else:
            mySrc.status_signal.emit("Acquisition thread termination. Please restart the application...")
            break


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--savepath', default=None, nargs='+', type=str,
                        dest='savepath', help='The path to save frames.')

    parser.add_argument('REMAIN', nargs='*')

    args_parser = parser.parse_args()

    app = QApplication([])
    widget = RGBTCam(args_parser=args_parser)
    widget.show()
    sys.exit(app.exec())
